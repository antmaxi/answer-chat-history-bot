"""Turn a question + retrieved windows into a grounded answer with citations.

The whole value of the bot rests on one rule: answer only from the supplied
excerpts, and say so plainly when the history doesn't contain the answer. A bot
that guesses is one people mute after a week.
"""

import re
import sqlite3
import time
from dataclasses import dataclass

from . import config, db, logconfig, retrieve
from .ingest import live
from .llm import LLM, get_llm

SYSTEM = """You answer questions about a group chat, using ONLY the excerpts provided.

Rules:
- Base every claim strictly on the excerpts. Never use outside knowledge or guess.
- If the excerpts don't contain the answer, say exactly: "I couldn't find that in the chat history." Do not speculate.
- Cite the excerpts you used with their [W#] tags, e.g. "You each owe 200 lari [W2]."
- Quote sparingly; prefer to summarize. Keep the answer to a few sentences.
- Answer in the same language as the question."""

CITATION = re.compile(r"\[W(\d+)\]")


@dataclass
class Answer:
    text: str
    hits: list[retrieve.Hit]

    def cited_indices(self) -> list[int]:
        """1-based [W#] tags the model used, bounded to windows that exist.

        Out-of-range tags (a model occasionally invents [W9]) are dropped.
        """
        used = {int(n) for n in CITATION.findall(self.text)}
        return sorted(i for i in used if 1 <= i <= len(self.hits))

    def cited_hits(self) -> list[retrieve.Hit]:
        """The subset of retrieved windows the model actually referenced."""
        idx = set(self.cited_indices())
        return [h for i, h in enumerate(self.hits, 1) if i in idx]

    def source_links(self, limit: int = 3) -> list[tuple[int, retrieve.Hit]]:
        """(index, window) pairs to show as sources.

        Prefer the windows the model cited. If it cited none — some answers read
        naturally without inline tags — fall back to the top retrieved windows,
        so the reader always gets links to the messages the answer drew on.
        """
        idx = self.cited_indices() or list(range(1, min(limit, len(self.hits)) + 1))
        return [(i, self.hits[i - 1]) for i in idx][:limit]

    def all_sources(self) -> list[tuple[int, retrieve.Hit, bool]]:
        """Every retrieved window with its link and whether the model cited it."""
        cited = set(self.cited_indices())
        return [(i, h, i in cited) for i, h in enumerate(self.hits, 1)]

    def sources_block(self) -> str:
        return "\n".join(
            f"[W{i}]{' ✓' if was_cited else ''} {h.when()} · {h.speakers} · {h.link()}"
            for i, h, was_cited in self.all_sources()
        )

    def primary_source(self) -> retrieve.Hit | None:
        """The single most relevant window backing the answer (top cited, else top hit)."""
        pairs = self.source_links(limit=1)
        return pairs[0][1] if pairs else None

    def primary_link(self) -> str | None:
        """Deep link to the first message the answer is grounded in."""
        hit = self.primary_source()
        return hit.link() if hit else None


def build_context(hits: list[retrieve.Hit]) -> str:
    blocks = []
    for i, hit in enumerate(hits, 1):
        header = f"[W{i}] {hit.when()}, {hit.speakers}:"
        blocks.append(f"{header}\n{hit.text}")
    return "\n\n".join(blocks)


def complete_answer(question: str, hits: list[retrieve.Hit], llm: LLM | None = None) -> Answer:
    """LLM call only — no DB. The bot runs this off the SQLite lock."""
    if not hits:
        return Answer("I couldn't find that in the chat history.", [])
    llm = llm or get_llm()
    user = f"Excerpts:\n\n{build_context(hits)}\n\n---\nQuestion: {question}"
    text = llm.complete(SYSTEM, user)
    return Answer(text, hits)


def _record(
    conn: sqlite3.Connection,
    question: str,
    chat_id,
    result: Answer,
    t0: float,
    llm: LLM | None,
    user_id: int | None = None,
) -> None:
    db.log_query(
        conn,
        question=question,
        chat_ids=retrieve.normalize_chat_ids(chat_id),
        window_ids=[h.window_id for h in result.hits],
        cited_ids=[h.window_id for h in result.cited_hits()],
        latency_ms=int((time.monotonic() - t0) * 1000),
        model=getattr(llm, "model", None) or config.ANSWER_MODEL,
        user_id=user_id,
    )


def answer(
    conn: sqlite3.Connection,
    question: str,
    chat_id: retrieve.ChatId = None,
    llm: LLM | None = None,
    *,
    flush: bool = True,
) -> Answer:
    t0 = time.monotonic()
    chats = retrieve.normalize_chat_ids(chat_id)
    if chats is not None and not chats:
        result = Answer("I couldn't find that in the chat history.", [])
        _record(conn, question, chat_id, result, t0, llm)
        return result

    # Live messages are in FTS immediately but only join windows after the tail
    # is re-windowed. Flush first so "what did we just say" sees the open tail.
    if flush:
        live.flush_tail(conn, chat_id)

    hits = retrieve.search(conn, question, chat_id)
    result = complete_answer(question, hits, llm)
    _record(conn, question, chat_id, result, t0, llm)
    return result


def main() -> None:
    import argparse

    from . import db

    logconfig.setup()
    ap = argparse.ArgumentParser(description="Answer a question from indexed history")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--chat-id", type=int, default=None)
    args = ap.parse_args()

    conn = db.connect()
    result = answer(conn, " ".join(args.question), args.chat_id)
    print(result.text)
    link = result.primary_link()
    if link:
        print(f"\nFirst message: {link}")
    sources = result.sources_block()
    if sources:
        print("\nSources:")
        print(sources)


if __name__ == "__main__":
    main()
