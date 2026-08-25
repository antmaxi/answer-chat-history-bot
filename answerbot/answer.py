"""Turn a question + retrieved windows into a grounded answer with citations.

The whole value of the bot rests on one rule: answer only from the supplied
excerpts, and say so plainly when the history doesn't contain the answer. A bot
that guesses is one people mute after a week.
"""

import html as html_lib
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
- Cite the excerpts you used with their [W#] tags, e.g. "You each owe 200 lari [W2]." Do not add URLs; the client turns [W#] into links.
- Format the answer in Markdown: **bold** for names, products, and constraints; *italic* for light emphasis; `code` for exact values or identifiers; hyphen bullets (`- `) when a list is clearer than a paragraph. Do not wrap the whole answer in a fenced code block. Do not use headings or images.
- Quote sparingly; prefer to summarize. Keep the answer to a few sentences.
- When excerpts disagree, prefer the more recent ones unless the question is about an earlier period.
- Answer in the same language as the question."""

CITATION = re.compile(r"\[W(\d+)\]")
# [W3], [W3](url), or [W3] (url) — models sometimes emit a markdown/parenthetical link.
CITATION_MARKUP = re.compile(r"\[W(\d+)\](?:\s*\(\s*https?://[^)]+\s*\))?")

_OUTER_FENCE = re.compile(
    r"\A\s*```(?:markdown|md)?\s*\n(.*)\n```\s*\Z",
    re.DOTALL | re.IGNORECASE,
)
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s)\"<>]+)\)")
_BOLD = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
# Opening * must not start a list item (`* foo`). No newlines, so list markers stay put.
_ITALIC = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
_SLOT = "\x00{}\x00"


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


def markdown_to_html(text: str) -> str:
    """Render a CommonMark subset as Telegram HTML.

    Escapes first so raw tags in the model output cannot inject markup, then
    converts **bold**, *italic*, `code`, fenced blocks, and [text](url) links.
    [W#] citations are left for linkify_citations.
    """
    m = _OUTER_FENCE.fullmatch(text)
    if m:
        text = m.group(1)
    text = html_lib.escape(text, quote=False)
    slots: list[str] = []

    def hold(fragment: str) -> str:
        slots.append(fragment)
        return _SLOT.format(len(slots) - 1)

    text = _FENCE.sub(lambda m: hold(f"<pre>{m.group(1)}</pre>"), text)
    text = _INLINE_CODE.sub(lambda m: hold(f"<code>{m.group(1)}</code>"), text)

    def link(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        if re.fullmatch(r"W\d+", label):
            return m.group(0)
        return f'<a href="{url}">{label}</a>'

    text = _LINK.sub(link, text)
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    for i, fragment in enumerate(slots):
        text = text.replace(_SLOT.format(i), fragment)
    return text


def linkify_citations(text: str, hits: list[retrieve.Hit]) -> str:
    """Turn [W#] (and a trailing markdown/parenthetical URL, if any) into source links."""

    def repl(m: re.Match) -> str:
        i = int(m.group(1))
        if 1 <= i <= len(hits):
            return f'<a href="{hits[i - 1].link()}">[W{i}]</a>'
        return m.group(0)

    return CITATION_MARKUP.sub(repl, text)


def format_answer_body(result: Answer) -> str:
    """LLM Markdown as Telegram HTML, with [W#] turned into t.me links."""
    return linkify_citations(markdown_to_html(result.text), result.hits)


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
