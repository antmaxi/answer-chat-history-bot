"""Turn a follow-up into a standalone search query using recent questions.

No extra LLM call: if the new question is dependent ("how much was it", a reply
to the bot, or extra words on an @mention-reply to someone else), stitch the
previous question in front so retrieval still has the nouns. The original
wording is what the answer model sees. A bare @mention as a reply to someone
else uses that message as the question.
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


def strip_mention(text: str, username: str) -> str:
    """Remove `@username` (any case) so the rest of a ping is the question."""
    raw = text or ""
    if not username:
        return raw.strip()
    return re.sub(re.escape(f"@{username}"), "", raw, flags=re.IGNORECASE).strip()


def question_from_mention(
    text: str,
    username: str,
    *,
    reply_text: str | None = None,
    reply_from_bot: bool = False,
) -> str:
    """Question implied by an @mention (or a reply to the bot).

    A bare @mention as a reply to someone else uses that message's text.
    Extra words on the ping stay the question. A reply to the bot is never
    replaced by the bot's previous answer.
    """
    q = strip_mention(text, username)
    if q or reply_from_bot:
        return q
    return (reply_text or "").strip()


def search_prior_for_reply(
    question: str,
    *,
    history_prior: str | None,
    reply_text: str | None,
    reply_from_bot: bool,
) -> tuple[str | None, bool]:
    """`(prior, force)` for `rewrite` when the ask is a Telegram reply.

    A reply to the bot keeps the asker's last question (`force=True`). A reply
    to someone else uses that message as prior only when the ping still has
    its own words — so a bare @mention does not search `q — follow-up: q`.
    """
    if reply_from_bot:
        return history_prior, True
    replied = (reply_text or "").strip()
    if replied and question.strip() != replied:
        return replied, False
    return history_prior, False
