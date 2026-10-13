"""Group messages into conversation windows and embed each message.

Windows are still built for incremental rebuilds and stats. Retrieval ranks
*messages* (keyword + vectors); query-time expansion grows a thread around
each hit so parallel topics in the same burst do not share one excerpt.

Each vector is a short passage, not the bare line: the reply parent, a few
earlier lines from the same sender, and — for a short line that is not a
reply — the previous message. Chronological neighbours from other people are
left out so two topics in the same minute do not share a vector.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

from . import config, db, embed, logconfig, people, retrieve, thread

log = logging.getLogger("answerbot")

# Bump when the passage recipe changes. Full reindex sets it; a mismatch with
# existing message_vecs means the corpus is still one-line (or an older recipe).
EMBED_PASSAGE_VERSION = "context-v1"


def render(
    msgs: list[sqlite3.Row],
    names: dict[int, str] | None = None,
    mode: str | None = None,
    aliases: dict[int, int] | None = None,
) -> str:
    """Render messages as a plain transcript (embedded lines and LLM excerpts).

    `names` maps sender id to a public display name. Unresolved people render as
    the stable "User N" in `aliases`, unless SPEAKER_LABEL=export (or mode="export")
    which falls back to the stored contact label. Under SPEAKER_LABEL=id, names
    are dropped entirely for those aliases.
    """
    names = names or {}
    lines = []
    for m in msgs:
        stamp = datetime.fromtimestamp(m["ts"], timezone.utc).strftime("%Y-%m-%d %H:%M")
        who = people.speaker_label(names, m["sender_id"], m["sender"] or "", mode, aliases)
        lines.append(f"[{stamp}] {who}: {m['text']}")
    return "\n".join(lines)


def build_windows(
    msgs: list[sqlite3.Row],
    mentioned: dict[int, set[int]] | None = None,
) -> list[list[sqlite3.Row]]:
    """Split a chronological message list into overlapping conversation windows.

    A window ends when the next message arrives after a long silence, or when
    the window is already long enough. A reply to something inside the current
    window, or an @mention of someone who spoke in it, keeps it open regardless
    of the gap — those are one thread even when they span hours.
    """
    windows: list[list[sqlite3.Row]] = []
    current: list[sqlite3.Row] = []
    chars = 0
    mentioned = mentioned or {}

    for msg in msgs:
        if current:
            gap = msg["ts"] - current[-1]["ts"]
            in_window = {m["msg_id"] for m in current}
            speakers_in_window = {
                m["sender_id"] for m in current if m["sender_id"] is not None
            }
            replies_into_window = msg["reply_to"] in in_window
            mentions_into_window = bool(
                mentioned.get(msg["msg_id"], set()) & speakers_in_window
            )

            too_old = (
                gap > config.WINDOW_GAP_SECONDS
                and not replies_into_window
                and not mentions_into_window
            )
            too_many = len(current) >= config.WINDOW_MAX_MSGS
            too_long = chars >= config.WINDOW_MAX_CHARS

            if too_old or too_many or too_long:
                windows.append(current)
                # Carry the tail forward so an answer split across a boundary
                # is still retrievable as a whole.
                overlap = [] if too_old else current[-config.WINDOW_OVERLAP:]
                current = list(overlap)
                chars = sum(len(m["text"]) for m in current)

        current.append(msg)
        chars += len(msg["text"])

    if current:
        windows.append(current)
    return windows


def _before(msg: sqlite3.Row, ts: int, msg_id: int) -> bool:
    return (int(msg["ts"]), int(msg["msg_id"])) < (ts, msg_id)


def passage_rows(
    target: sqlite3.Row,
    chrono: list[sqlite3.Row],
    by_msg_id: dict[int, sqlite3.Row],
) -> list[sqlite3.Row]:
    """Context lines for `target`, oldest first, with `target` last.

    `chrono` is the neighbourhood already limited to THREAD_SAME_SPEAKER_SECONDS
    of the pending batch (plus nothing after `target` is required — callers
    may pass the whole batch). Reply parents outside that neighbourhood are
    looked up in `by_msg_id`.
    """
    ts = int(target["ts"])
    msg_id = int(target["msg_id"])
    gap = config.THREAD_SAME_SPEAKER_SECONDS
    sender = target["sender_id"]
    earlier = [m for m in chrono if _before(m, ts, msg_id)]

    chosen: dict[int, sqlite3.Row] = {}
    parent_id = target["reply_to"]
    if parent_id:
        parent = by_msg_id.get(int(parent_id))
        if parent is not None and int(parent["msg_id"]) != msg_id:
            chosen[int(parent["msg_id"])] = parent

    n_ctx = config.EMBED_CONTEXT_MSGS
    if sender is not None and n_ctx > 0:
        same = [
            m
            for m in earlier
            if m["sender_id"] == sender and ts - int(m["ts"]) <= gap
        ]
        for m in same[-n_ctx:]:
            chosen[int(m["msg_id"])] = m

    text = target["text"] or ""
    if not parent_id and len(text) < config.EMBED_SHORT_CHARS and earlier:
        prev = earlier[-1]
        if ts - int(prev["ts"]) <= gap:
            chosen[int(prev["msg_id"])] = prev

    ordered = sorted(chosen.values(), key=lambda m: (int(m["ts"]), int(m["msg_id"])))
    return ordered + [target]


def render_passage(
    rows: list[sqlite3.Row],
    names: dict[int, str] | None = None,
    aliases: dict[int, int] | None = None,
) -> str:
    """Render a passage, dropping oldest context until it fits WINDOW_MAX_CHARS."""
    if not rows:
        return ""
    max_chars = max(1, config.WINDOW_MAX_CHARS)
    target = rows[-1]
    context = list(rows[:-1])
    while context:
        text = render(context + [target], names, aliases=aliases)
        if len(text) <= max_chars:
            return text
        context.pop(0)
    text = render([target], names, aliases=aliases)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def passage_texts(
    conn: sqlite3.Connection,
    chat_id: int,
    pending: list[sqlite3.Row],
    names: dict[int, str],
    aliases: dict[int, int],
) -> list[str]:
    """One passage per pending message. Context may sit before the reindex range."""
    if not pending:
        return []
    min_ts = min(int(r["ts"]) for r in pending)
    start = min_ts - config.THREAD_SAME_SPEAKER_SECONDS
    chrono = list(
        conn.execute(
            """SELECT * FROM messages WHERE chat_id=? AND ts>=?
               ORDER BY ts, msg_id""",
            (chat_id, start),
        )
    )
    by_id = {int(r["msg_id"]): r for r in chrono}
    missing = sorted(
        {
            int(r["reply_to"])
            for r in pending
            if r["reply_to"] and int(r["reply_to"]) not in by_id
        }
    )
    if missing:
        placeholders = ",".join("?" * len(missing))
        extra = conn.execute(
            f"SELECT * FROM messages WHERE chat_id=? AND msg_id IN ({placeholders})",
            [chat_id, *missing],
        )
        for row in extra:
            by_id[int(row["msg_id"])] = row
    return [
        render_passage(passage_rows(row, chrono, by_id), names, aliases)
        for row in pending
    ]


def embeddings_are_stale(conn: sqlite3.Connection) -> bool:
    """True when stored vectors exist but were not built with this passage recipe."""
    n = conn.execute("SELECT count(*) FROM message_vecs").fetchone()[0]
    if not n:
        return False
    return db.meta_get(conn, "embed_passage") != EMBED_PASSAGE_VERSION


def warn_stale_embeddings(conn: sqlite3.Connection) -> None:
    if embeddings_are_stale(conn):
        log.warning(
            "message embeddings predate contextual passages; "
            "run a full index: python -m answerbot.index or /reindex full"
        )


def mark_embed_passage(conn: sqlite3.Connection) -> None:
    db.meta_set(conn, "embed_passage", EMBED_PASSAGE_VERSION)


def _chat_ids(conn: sqlite3.Connection, chat_id: retrieve.ChatId = None) -> list[int]:
    chats = retrieve.normalize_chat_ids(chat_id)
    if chats is not None:
        return chats
    return [r[0] for r in conn.execute("SELECT DISTINCT chat_id FROM messages")]


def _watermark(conn: sqlite3.Connection, chat_id: int) -> int:
    row = conn.execute(
        "SELECT last_indexed_msg_id FROM state WHERE chat_id=?", (chat_id,)
    ).fetchone()
    return row[0] if row else 0


def _rebuild_boundary(conn: sqlite3.Connection, chat_id: int) -> int:
    """Where an incremental pass must start: the first message of the last, still
    "open" window. That window is rebuilt together with the new messages, because
    the next message often belongs inside it rather than starting a fresh one.
    Returns 0 when the chat has no windows yet (i.e. index the whole thing)."""
    row = conn.execute(
        "SELECT first_msg FROM windows WHERE chat_id=? ORDER BY last_msg DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    return row[0] if row else 0


@dataclass
class EmbedJob:
    """Windows + message texts ready to embed, produced under the DB lock."""

    chat_id: int
    n_windows: int
    watermark: int
    pending_ids: list[int]
    pending_texts: list[str]


def plan_from(conn: sqlite3.Connection, chat_id: int, from_msg_id: int) -> EmbedJob | None:
    """Rebuild windows from `from_msg_id` onward. Does not encode.

    Returns the texts that still need vectors, so the caller can encode them
    without holding a database lock. `None` if this chat has no messages in range.
    """
    conn.execute("DELETE FROM windows WHERE chat_id=? AND first_msg>=?", (chat_id, from_msg_id))
    conn.commit()

    msgs = conn.execute(
        "SELECT * FROM messages WHERE chat_id=? AND msg_id>=? ORDER BY ts, msg_id",
        (chat_id, from_msg_id),
    ).fetchall()
    if not msgs:
        return None

    names = people.name_map(conn)
    people.ensure_aliases(conn, (m["sender_id"] for m in msgs))
    aliases = people.alias_map(conn)
    handles = people.mention_map(conn)
    mentioned = {
        m["msg_id"]: thread.resolve_mentions(m["text"], handles) for m in msgs
    }
    rows = []
    for g in build_windows(msgs, mentioned):
        speakers = sorted(
            {people.speaker_label(names, m["sender_id"], m["sender"] or "", aliases=aliases)
             for m in g}
        )
        rows.append(
            (chat_id, g[0]["msg_id"], g[-1]["msg_id"], g[0]["ts"], g[-1]["ts"],
             ", ".join(speakers), render(g, names, aliases=aliases))
        )
    conn.executemany(
        """INSERT INTO windows (chat_id, first_msg, last_msg, ts_start, ts_end, speakers, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    # Drop stale message vectors in this range so edits and renamed speakers
    # are re-embedded. Pending is whatever still lacks a vector afterwards.
    conn.execute(
        """DELETE FROM message_vecs WHERE message_id IN (
               SELECT id FROM messages WHERE chat_id=? AND msg_id>=?)""",
        (chat_id, from_msg_id),
    )
    conn.commit()

    pending = conn.execute(
        """SELECT m.* FROM messages m
           LEFT JOIN message_vecs v ON v.message_id = m.id
           WHERE m.chat_id=? AND m.msg_id>=? AND v.message_id IS NULL
           ORDER BY m.ts, m.msg_id""",
        (chat_id, from_msg_id),
    ).fetchall()
    pending_texts = passage_texts(conn, chat_id, pending, names, aliases)
    return EmbedJob(
        chat_id=chat_id,
        n_windows=len(rows),
        watermark=max(m["msg_id"] for m in msgs),
        pending_ids=[r["id"] for r in pending],
        pending_texts=pending_texts,
    )


def apply_job(conn: sqlite3.Connection, job: EmbedJob, vecs) -> int:
    """Write vectors (if any) and advance the watermark."""
    if job.pending_ids and vecs is not None:
        conn.executemany(
            "INSERT OR REPLACE INTO message_vecs (message_id, vec) VALUES (?, ?)",
            [(mid, embed.pack(v)) for mid, v in zip(job.pending_ids, vecs)],
        )
    conn.execute(
        """INSERT INTO state (chat_id, last_indexed_msg_id) VALUES (?, ?)
           ON CONFLICT (chat_id) DO UPDATE SET last_indexed_msg_id=excluded.last_indexed_msg_id""",
        (job.chat_id, job.watermark),
    )
    conn.commit()
    retrieve.invalidate_cache()
    return job.n_windows


def _index_from(conn: sqlite3.Connection, chat_id: int, from_msg_id: int, progress: bool) -> int:
    """(Re)window and embed messages with msg_id >= from_msg_id for one chat.

    The single indexing primitive. Windows at or after the boundary are dropped
    and rebuilt, and message vectors in that range are re-encoded. Everything
    before the boundary is left untouched, so only the changed tail is
    re-embedded. `from_msg_id=0` rebuilds the whole chat. Embedding, the
    expensive step, is what this narrows down.
    """
    job = plan_from(conn, chat_id, from_msg_id)
    if job is None:
        return 0
    vecs = None
    if job.pending_texts:
        vecs = embed.encode_passages(job.pending_texts, progress=progress)
    return apply_job(conn, job, vecs)


def reindex(conn: sqlite3.Connection, chat_id: retrieve.ChatId = None, progress: bool = True) -> dict:
    """Rebuild windows and embeddings from scratch for one chat, or for all."""
    jobs = plan_reindex(conn, chat_id)
    total = 0
    for job in jobs:
        vecs = embed.encode_passages(job.pending_texts, progress=progress) if job.pending_texts else None
        total += apply_job(conn, job, vecs)
    if chat_id is None:
        mark_embed_passage(conn)
    retrieve.invalidate_cache()
    return {"chats": len(jobs), "windows": total}


def _lookback_boundary(conn: sqlite3.Connection, chat_id: int, days: int) -> int | None:
    """First message of the earliest window ending within the last `days`.

    "Now" is the chat's newest message, not wall-clock, so a historical import
    reindexes the tail of its own timeline rather than nothing. Returns None when
    no window falls in the window (e.g. chat quiet longer than `days`)."""
    latest = conn.execute(
        "SELECT max(ts) FROM messages WHERE chat_id=?", (chat_id,)
    ).fetchone()[0]
    if latest is None:
        return None
    cutoff = latest - days * 86400
    return conn.execute(
        "SELECT min(first_msg) FROM windows WHERE chat_id=? AND ts_end>=?", (chat_id, cutoff)
    ).fetchone()[0]


def plan_reindex(conn: sqlite3.Connection, chat_id: retrieve.ChatId = None) -> list[EmbedJob]:
    """Drop and rebuild windows for each chat; return embed jobs (no encoding)."""
    jobs = []
    for cid in _chat_ids(conn, chat_id):
        job = plan_from(conn, cid, 0)
        if job is not None:
            jobs.append(job)
    return jobs


def plan_update(
    conn: sqlite3.Connection,
    chat_id: retrieve.ChatId = None,
    lookback_days: int = 0,
    force: bool = False,
) -> list[EmbedJob]:
    """Decide which tails to rebuild and insert their windows. Does not encode."""
    jobs = []
    for cid in _chat_ids(conn, chat_id):
        new = conn.execute(
            "SELECT count(*) FROM messages WHERE chat_id=? AND msg_id>?",
            (cid, _watermark(conn, cid)),
        ).fetchone()[0]
        if not new and not lookback_days and not force:
            continue

        boundary = _rebuild_boundary(conn, cid)
        if lookback_days:
            lb = _lookback_boundary(conn, cid, lookback_days)
            if lb is not None:
                boundary = min(boundary, lb)

        job = plan_from(conn, cid, boundary)
        if job is not None:
            jobs.append(job)
    return jobs


def update(
    conn: sqlite3.Connection,
    chat_id: retrieve.ChatId = None,
    lookback_days: int = 0,
    progress: bool = False,
    force: bool = False,
) -> dict:
    """Incrementally index only what changed near the tail.

    Re-windows and re-embeds the open tail (new messages since the last pass),
    plus — when `lookback_days` > 0 — every window from the last `lookback_days`
    of the timeline, so recent edits are refreshed. The bulk of the corpus is
    left untouched, so this stays cheap. Safe on a never-indexed chat, where it
    falls back to a full index.

    `lookback_days=0` (the default, used by live ingest) is pure tail and runs
    only when new messages exist. `force=True` rebuilds the open tail even then,
    which is how a live edit of a just-indexed message refreshes its window.
    A positive lookback also runs when only edits, not new messages, are pending.
    """
    warn_stale_embeddings(conn)
    jobs = plan_update(conn, chat_id, lookback_days=lookback_days, force=force)
    total_windows = 0
    for job in jobs:
        vecs = embed.encode_passages(job.pending_texts, progress=progress) if job.pending_texts else None
        total_windows += apply_job(conn, job, vecs)
    return {"chats": len(jobs), "windows": total_windows}


def main() -> None:
    import argparse

    logconfig.setup()
    ap = argparse.ArgumentParser(description="Build conversation windows and embeddings")
    ap.add_argument("--chat-id", type=int, default=None)
    ap.add_argument(
        "--update",
        action="store_true",
        help="incremental: index the new tail plus the last few weeks (fast)",
    )
    ap.add_argument(
        "--lookback-days",
        type=int,
        default=config.UPDATE_LOOKBACK_DAYS,
        help="with --update, also re-window this many recent days to catch edits "
        f"(default {config.UPDATE_LOOKBACK_DAYS}; 0 = tail only)",
    )
    ap.add_argument(
        "--speaker-label",
        choices=["name", "id", "export"],
        default=config.SPEAKER_LABEL,
        help="label speakers by public name (User N fallback), anonymous id, "
        "or export/contact label (default %(default)s)",
    )
    args = ap.parse_args()

    config.SPEAKER_LABEL = args.speaker_label  # honoured by render/_index_from
    conn = db.connect()
    if args.update:
        result = update(conn, args.chat_id, lookback_days=args.lookback_days, progress=True)
        verb = "updated"
    else:
        result = reindex(conn, args.chat_id, progress=True)
        verb = "indexed"
    from . import faq

    faq.update(conn, progress=True)
    print(f"{verb} {result['windows']} windows across {result['chats']} chat(s)")
    print(db.stats(conn))


if __name__ == "__main__":
    main()
