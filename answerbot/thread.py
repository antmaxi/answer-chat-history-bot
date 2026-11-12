"""Grow a conversation thread around a retrieved message.

Index-time windows are still a chronological slice of the chat, so two topics
that run in parallel land in the same bag. Retrieval now ranks *messages*;
this module rebuilds the excerpt from the seed using replies, @mentions,
same-speaker fallback (when a message has no vector), and cosine vs the seed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, TypeVar

from . import config

# Telegram usernames are Latin alphanumerics plus underscore.
_MENTION = re.compile(r"@([A-Za-z0-9_]{3,32})")

Msg = Mapping[str, Any]
MessageKey = TypeVar("MessageKey", int, tuple[int, int])


def mentions_in(text: str | None) -> list[str]:
    """Lowercased @handles in `text` (Telegram username shape)."""
    if not text:
        return []
    return [h.lower() for h in _MENTION.findall(text)]


def resolve_mentions(text: str | None, handles: Mapping[str, int]) -> set[int]:
    """Sender ids mentioned in `text` via `handles` (lowercase handle → id)."""
    return {handles[h] for h in mentions_in(text) if h in handles}


def _mid(msg: Msg) -> int:
    return int(msg["msg_id"])


def _get(msg: Msg, key: str):
    try:
        return msg[key]
    except (KeyError, IndexError):
        return None


def _sender(msg: Msg) -> int | None:
    sid = _get(msg, "sender_id")
    if sid is None:
        return None
    return int(sid)


def _reply_to(msg: Msg) -> int | None:
    rid = _get(msg, "reply_to")
    if rid is None:
        return None
    return int(rid)


def _ts(msg: Msg) -> int:
    return int(msg["ts"])


def _text(msg: Msg) -> str:
    return str(msg["text"] or "")


def expand_thread(
    msgs: Sequence[Msg],
    seed_msg_id: int,
    *,
    mentioned: Mapping[int, set[int]] | None = None,
    cosine: Mapping[int, float] | None = None,
    max_msgs: int | None = None,
    max_chars: int | None = None,
    cosine_min: float | None = None,
    same_speaker_seconds: int | None = None,
) -> list[Msg]:
    """Subset of `msgs` that belongs with `seed_msg_id`, in chronological order.

    `msgs` is the candidate neighbourhood (time radius + reply chain). `mentioned`
    maps msg_id → sender ids @mentioned in that line. `cosine` maps msg_id →
    similarity to the seed; missing keys mean "no vector".
    """
    if not msgs:
        return []
    by_id = {_mid(m): m for m in msgs}
    seed = by_id.get(seed_msg_id)
    if seed is None:
        return []

    max_msgs = config.WINDOW_MAX_MSGS if max_msgs is None else max_msgs
    max_chars = config.WINDOW_MAX_CHARS if max_chars is None else max_chars
    cosine_min = config.THREAD_COSINE_MIN if cosine_min is None else cosine_min
    same_speaker_seconds = (
        config.THREAD_SAME_SPEAKER_SECONDS
        if same_speaker_seconds is None
        else same_speaker_seconds
    )
    mentioned = mentioned or {}
    cosine = cosine or {}

    included: set[int] = {seed_msg_id}

    # Walk reply ancestors first so a late answer still carries the question.
    cur = seed
    hops = 0
    while hops < 20:
        parent = _reply_to(cur)
        if parent is None or parent not in by_id or parent in included:
            break
        included.add(parent)
        cur = by_id[parent]
        hops += 1

    seed_sender = _sender(seed)
    seed_ts = _ts(seed)

    def similar(mid: int) -> bool:
        value = cosine.get(mid)
        return value is not None and isfinite(value) and value >= cosine_min

    def no_vector(mid: int) -> bool:
        return mid not in cosine

    self_cosine = cosine.get(seed_msg_id, 0.0)
    degenerate = (
        not isfinite(self_cosine)
        or self_cosine <= 1e-6
        or not any(mid != seed_msg_id and isfinite(c) for mid, c in cosine.items())
    )

    if cosine_min > 0:
        for mid in by_id:
            if similar(mid):
                included.add(mid)

    # Collapsed / missing embeddings cannot tell two topics apart. Keep the
    # whole candidate burst (already radius-clipped) so a keyword hit still
    # carries the next-line answer.
    if degenerate:
        included.update(by_id)

    def thread_senders() -> set[int]:
        out: set[int] = set()
        for mid in included:
            sid = _sender(by_id[mid])
            if sid is not None:
                out.add(sid)
        return out

    def mentioned_in_thread() -> set[int]:
        out: set[int] = set()
        for mid in included:
            out |= mentioned.get(mid, set())
        return out

    changed = True
    while changed:
        changed = False
        senders = thread_senders()
        pinged = mentioned_in_thread()
        for mid, msg in by_id.items():
            if mid in included:
                continue
            sid = _sender(msg)
            parent = _reply_to(msg)
            ok = False
            if parent is not None and parent in included:
                ok = True
            elif sid is not None and sid in pinged:
                ok = True
            elif mentioned.get(mid, set()) & senders:
                ok = True
            elif (
                seed_sender is not None
                and sid is not None
                and sid == seed_sender
                and abs(_ts(msg) - seed_ts) <= same_speaker_seconds
                and no_vector(mid)
            ):
                # No embedding to tell topics apart — keep this speaker's nearby
                # lines so a "yeah" still has a little context.
                ok = True
            if ok:
                included.add(mid)
                changed = True

    chosen = [by_id[mid] for mid in included if mid in by_id]
    chosen.sort(key=lambda m: (_ts(m), _mid(m)))
    return _cap(chosen, seed_msg_id, max_msgs, max_chars)


def _cap(msgs: list[Msg], seed_msg_id: int, max_msgs: int, max_chars: int) -> list[Msg]:
    """Keep the seed and prefer direct replies over incidental neighbours."""
    if not msgs:
        return []
    max_msgs = max(1, max_msgs)
    max_chars = max(1, max_chars)
    by_id = {_mid(m): m for m in msgs}
    seed = by_id[seed_msg_id]
    seed_ts = _ts(seed)
    parent_id = _reply_to(seed)

    ranked = sorted(
        (m for m in msgs if _mid(m) != seed_msg_id),
        key=lambda m: (
            0 if _mid(m) == parent_id or _reply_to(m) == seed_msg_id else 1,
            abs(_ts(m) - seed_ts),
            abs(_mid(m) - seed_msg_id),
        ),
    )
    kept = {seed_msg_id}
    chars = len(_text(seed))
    for m in ranked:
        if len(kept) >= max_msgs:
            break
        size = len(_text(m))
        if chars + size <= max_chars:
            kept.add(_mid(m))
            chars += size
    return [m for m in msgs if _mid(m) in kept]


def jaccard(a: Sequence[MessageKey], b: Sequence[MessageKey]) -> float:
    """Overlap of two msg_id lists. Empty vs anything is 0."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def dedupe_seeds(
    ranked: Sequence[tuple[MessageKey, float]],
    members: Mapping[MessageKey, Sequence[MessageKey]],
    overlap: float | None = None,
) -> list[MessageKey]:
    """Greedy keep of seed ids whose expanded threads barely overlap.

    `ranked` is (seed_msg_id, score) best-first. `members` maps seed → msg_ids
    in its thread. A later seed is dropped when Jaccard with an already kept
    thread is ≥ `overlap`.
    """
    threshold = config.THREAD_OVERLAP_JACCARD if overlap is None else overlap
    kept: list[MessageKey] = []
    kept_sets: list[set[MessageKey]] = []
    for seed_id, _score in ranked:
        ids = set(members.get(seed_id, ()))
        if not ids:
            continue
        if any(ids & prev and jaccard(ids, prev) >= threshold for prev in kept_sets):
            continue
        kept.append(seed_id)
        kept_sets.append(ids)
    return kept
