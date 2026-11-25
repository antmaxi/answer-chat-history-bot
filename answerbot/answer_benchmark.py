"""Evaluate answer behavior with a synthetic, keyword-indexed chat."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

from . import answer, retrieve
from .llm import LLM


@dataclass(frozen=True)
class Case:
    name: str
    chat_id: int
    question: str
    expected_time: str | None = None
    supporting_message: int | None = None
    context_messages: tuple[int, ...] = ()


CASES = (
    Case("supported-en", 1, "What time is the design review? Use HH:MM.", "09:30", 1, (1,)),
    Case("supported-ru", 2, "Во сколько экскурсия? Укажи время в формате ЧЧ:ММ.", "14:20", 1, (1,)),
    Case("empty-search", 1, "What is the observatory access code?"),
    Case("related-without-answer", 3, "What time is the planning meeting? Use HH:MM.",
         context_messages=(1,)),
    Case("later-correction", 4, "What time is the launch meeting now? Use HH:MM.",
         "11:00", 2, (1, 2)),
)

_MESSAGES = (
    (1, 1, None, 0, "The design review starts at 09:30."),
    (2, 1, None, 0, "Экскурсия начинается в 14:20."),
    (3, 1, None, 0, "The planning meeting needs a projector. I'll bring one."),
    (4, 1, None, 0, "The launch meeting starts at 10:00."),
    (4, 2, 1, 60, "Correction: the launch meeting moved from 10:00 to 11:00."),
)
_FIXTURE_EPOCH = 1_768_478_400
_CLOCK_TIME = re.compile(r"(?<![\d:])(?:[01]\d|2[0-3]):[0-5]\d(?![\d:])")


def seed(conn: sqlite3.Connection) -> None:
    # The message triggers populate FTS. This benchmark needs no embeddings.
    with conn:
        conn.executemany(
            """INSERT INTO messages (chat_id, msg_id, reply_to, ts, text)
               VALUES (?, ?, ?, ?, ?)""",
            [(chat, msg, parent, _FIXTURE_EPOCH + offset, text)
             for chat, msg, parent, offset, text in _MESSAGES],
        )
        # Keep the case terms below the FTS document-frequency cutoff. These
        # background messages belong to a chat outside every case's scope.
        conn.executemany(
            "INSERT INTO messages (chat_id, msg_id, ts, text) VALUES (5, ?, ?, ?)",
            [(i, _FIXTURE_EPOCH, "The cafeteria menu is posted.") for i in range(1, 151)],
        )


def message_ids(hits: list[retrieve.Hit], chat_id: int) -> set[int]:
    return {
        msg for hit in hits if hit.kind == "chat" and hit.chat_id == chat_id
        for msg in hit.msg_ids
    }


def score(case: Case, result: answer.Answer) -> tuple[str, ...]:
    if case.expected_time is None:
        return () if result.abstain else ("unsupported answer",)
    if result.abstain:
        return ("missed supported answer",)
    failures = []
    # Each fixture asks for a single time in HH:MM. An answer containing
    # both values cannot establish which one the bot chose.
    if set(_CLOCK_TIME.findall(result.text)) != {case.expected_time}:
        failures.append("incorrect fact")
    raw_citations = answer.CITATION.findall(result.text)
    invalid_citation = any(
        not 1 <= int(tag) <= len(result.hits) for tag in raw_citations
    )
    if invalid_citation or case.supporting_message not in message_ids(
        result.cited_hits(), case.chat_id
    ):
        failures.append("missing or incorrect citation")
    return tuple(failures)


def run(conn: sqlite3.Connection, llm: LLM | None = None) -> int:
    """Print the case results. Return 2 for execution errors or 1 for model misses."""
    failures = errors = 0
    for case in CASES:
        expected = "abstain"
        if case.expected_time is not None:
            expected = f"answer {case.expected_time}, cite message {case.supporting_message}"
        print(f"\n{case.name} (chat {case.chat_id}): {case.question}")
        print(f"  Expected: {expected}")
        try:
            hits = retrieve.search(conn, case.question, chat_id=case.chat_id)
        except Exception as exc:
            print(f"  ERROR: fixture search failed ({type(exc).__name__})")
            errors += 1
            continue
        retrieved = message_ids(hits, case.chat_id)
        if (not case.context_messages and hits) or not retrieved.issuperset(case.context_messages):
            print(f"  ERROR: unexpected fixture retrieval (messages {sorted(retrieved)})")
            errors += 1
            continue
        try:
            result = answer.complete_answer(case.question, hits, llm=llm)
        except Exception as exc:
            # Provider exceptions can contain request details. Report the error
            # type without dumping credentials or the request.
            print(f"  ERROR: answer model call failed ({type(exc).__name__})")
            errors += 1
            continue
        problems = score(case, result)
        failures += bool(problems)
        observed = "abstain" if result.abstain else "answer"
        cited = sorted(message_ids(result.cited_hits(), case.chat_id))
        print(f"  Observed: {observed}, cited messages {cited}")
        print(f"  Reply: {result.text}")
        print("  FAIL: " + ", ".join(problems) if problems else "  PASS")
    print(f"\n{len(CASES) - failures - errors}/{len(CASES)} passed, {failures} failed, {errors} errors")
    return 2 if errors else int(bool(failures))
