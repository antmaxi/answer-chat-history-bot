"""Turn a question + retrieved windows into a grounded answer with citations.

The whole value of the bot rests on one rule: answer only from the supplied
excerpts, and say so plainly when the history doesn't contain the answer. A bot
that guesses is one people mute after a week.
"""

import re
import sqlite3
from dataclasses import dataclass

from . import config, retrieve
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

    def sources_block(self, limit: int = 3) -> str:
        return "\n".join(
            f"[W{i}] {h.when()} · {h.speakers} · {h.link()}"
            for i, h in self.source_links(limit)
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


def answer(
    conn: sqlite3.Connection,
    question: str,
    chat_id: int | None = None,
    llm: LLM | None = None,
) -> Answer:
    hits = retrieve.search(conn, question, chat_id)
    if not hits:
        return Answer("I couldn't find that in the chat history.", [])

    llm = llm or get_llm()
    user = f"Excerpts:\n\n{build_context(hits)}\n\n---\nQuestion: {question}"
    text = llm.complete(SYSTEM, user)
    return Answer(text, hits)


def main() -> None:
    import argparse

    from . import db

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
