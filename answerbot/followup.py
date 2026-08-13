"""Turn a follow-up into a standalone search query using recent questions.

No extra LLM call: if the new question is dependent ("how much was it", a reply
to the bot), stitch the previous question in front so retrieval still has the
nouns. The original wording is what the answer model sees.
"""

from __future__ import annotations

import re

# Short, pronoun-heavy, or explicitly continuing. Keep this conservative —
# rewriting a standalone question against the wrong prior hurts more than
# missing a follow-up.
_FOLLOWUP = re.compile(
    r"(?is)^("
    r"what about(\s+\w+){0,3}\??$|"
    r"(and|also)\s+(what|how|who|when|why|the)\b|"
    r"how much (was|is|were|are) (it|that|those|them)\b|"
    r"(who|what|when|where|why|how) (was|is|were|are) (it|that|this)\b|"
    r"the other one\b|"
    r"(and|also|wait)[,!]?\s+"
    r")"
)

_PRONOUN = re.compile(r"(?i)\b(it|that|those|them|this|one)\b")


def looks_like_followup(question: str) -> bool:
    q = question.strip()
    if not q:
        return False
    if _FOLLOWUP.search(q):
        return True
    words = q.split()
    return len(words) <= 5 and bool(_PRONOUN.search(q))


def rewrite(question: str, prior: str | None, *, force: bool = False) -> str:
    """Standalone search string. `force` is for a reply to the bot's own message."""
    q = question.strip()
    prior = (prior or "").strip()
    if not prior:
        return q
    if force or looks_like_followup(q):
        return f"{prior} — follow-up: {q}"
    return q
