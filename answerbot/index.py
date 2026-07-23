"""Group messages into conversation windows and embed them.

Windows, not individual messages, are the retrieval unit. Chat messages are
short and context-free on their own ("yeah, that one"), so embedding them
individually retrieves noise.
"""

import sqlite3
from datetime import datetime, timezone

from . import config, db, embed, people


def render(msgs: list[sqlite3.Row], names: dict[int, str] | None = None, mode: str | None = None) -> str:
    """Render a window as a plain transcript, which is what gets embedded.

    `names` resolves the export's private contact labels to real display names by
    sender id; anyone unresolved keeps their stored label. Under SPEAKER_LABEL=id
    (or mode="id") names are dropped entirely for anonymous ids.
    """
    names = names or {}
    lines = []
    for m in msgs:
        stamp = datetime.fromtimestamp(m["ts"], timezone.utc).strftime("%Y-%m-%d %H:%M")
        who = people.speaker_label(names, m["sender_id"], m["sender"], mode)
        lines.append(f"[{stamp}] {who}: {m['text']}")
    return "\n".join(lines)


def build_windows(msgs: list[sqlite3.Row]) -> list[list[sqlite3.Row]]:
    """Split a chronological message list into overlapping conversation windows.

    A window ends when the next message arrives after a long silence, or when
    the window is already long enough. A reply to something inside the current
    window keeps it open regardless of the gap — reply chains are one thread of
    conversation even when they span hours.
    """
    windows: list[list[sqlite3.Row]] = []
    current: list[sqlite3.Row] = []
    chars = 0

    for msg in msgs:
        if current:
            gap = msg["ts"] - current[-1]["ts"]
            replies_into_window = msg["reply_to"] in {m["msg_id"] for m in current}

            too_old = gap > config.WINDOW_GAP_SECONDS and not replies_into_window
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


def _chat_ids(conn: sqlite3.Connection, chat_id: int | None) -> list[int]:
    if chat_id is not None:
        return [chat_id]
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


def _index_from(conn: sqlite3.Connection, chat_id: int, from_msg_id: int, progress: bool) -> int:
    """(Re)window and embed messages with msg_id >= from_msg_id for one chat.

    The single indexing primitive. Windows at or after the boundary are dropped
    and rebuilt — their vectors cascade away — and everything before it is left
    untouched, so only the changed tail is re-embedded. `from_msg_id=0` rebuilds
    the whole chat. Embedding, the expensive step, is what this narrows down.
    """
    conn.execute("DELETE FROM windows WHERE chat_id=? AND first_msg>=?", (chat_id, from_msg_id))
    conn.commit()  # vectors for the dropped windows go with them (ON DELETE CASCADE)

    msgs = conn.execute(
        "SELECT * FROM messages WHERE chat_id=? AND msg_id>=? ORDER BY ts, msg_id",
        (chat_id, from_msg_id),
    ).fetchall()
    if not msgs:
        return 0

    names = people.name_map(conn)
    rows = []
    for g in build_windows(msgs):
        speakers = sorted({people.speaker_label(names, m["sender_id"], m["sender"]) for m in g if m["sender"]})
        rows.append(
            (chat_id, g[0]["msg_id"], g[-1]["msg_id"], g[0]["ts"], g[-1]["ts"],
             ", ".join(speakers), render(g, names))
        )
    conn.executemany(
        """INSERT INTO windows (chat_id, first_msg, last_msg, ts_start, ts_end, speakers, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    # Embed exactly the windows that lack a vector. Deriving the work from the
    # data (rather than the boundary) is self-correcting: a crash between window
    # insert and embed just leaves them to be picked up on the next run.
    pending = conn.execute(
        """SELECT w.id, w.text FROM windows w
           LEFT JOIN window_vecs v ON v.window_id = w.id
           WHERE w.chat_id=? AND v.window_id IS NULL ORDER BY w.id""",
        (chat_id,),
    ).fetchall()
    if pending:
        vecs = embed.encode_passages([r["text"] for r in pending], progress=progress)
        conn.executemany(
            "INSERT OR REPLACE INTO window_vecs (window_id, vec) VALUES (?, ?)",
            [(r["id"], embed.pack(v)) for r, v in zip(pending, vecs)],
        )

    conn.execute(
        """INSERT INTO state (chat_id, last_indexed_msg_id) VALUES (?, ?)
           ON CONFLICT (chat_id) DO UPDATE SET last_indexed_msg_id=excluded.last_indexed_msg_id""",
        (chat_id, max(m["msg_id"] for m in msgs)),
    )
    conn.commit()
    return len(rows)


def reindex(conn: sqlite3.Connection, chat_id: int | None = None, progress: bool = True) -> dict:
    """Rebuild windows and embeddings from scratch for one chat, or for all."""
    chat_ids = _chat_ids(conn, chat_id)
    total = sum(_index_from(conn, cid, 0, progress) for cid in chat_ids)
    return {"chats": len(chat_ids), "windows": total}


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


def update(
    conn: sqlite3.Connection,
    chat_id: int | None = None,
    lookback_days: int = 0,
    progress: bool = False,
) -> dict:
    """Incrementally index only what changed near the tail.

    Re-windows and re-embeds the open tail (new messages since the last pass),
    plus — when `lookback_days` > 0 — every window from the last `lookback_days`
    of the timeline, so recent edits are refreshed. The bulk of the corpus is
    left untouched, so this stays cheap. Safe on a never-indexed chat, where it
    falls back to a full index.

    `lookback_days=0` (the default, used by live ingest) is pure tail and runs
    only when new messages exist. A positive lookback also runs when only edits,
    not new messages, are pending.
    """
    total_windows = 0
    touched = 0
    for cid in _chat_ids(conn, chat_id):
        new = conn.execute(
            "SELECT count(*) FROM messages WHERE chat_id=? AND msg_id>?",
            (cid, _watermark(conn, cid)),
        ).fetchone()[0]
        if not new and not lookback_days:
            continue

        boundary = _rebuild_boundary(conn, cid)
        if lookback_days:
            lb = _lookback_boundary(conn, cid, lookback_days)
            if lb is not None:
                boundary = min(boundary, lb)

        total_windows += _index_from(conn, cid, boundary, progress)
        touched += 1
    return {"chats": touched, "windows": total_windows}


def main() -> None:
    import argparse

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
        choices=["name", "id"],
        default=config.SPEAKER_LABEL,
        help="label speakers by real name or anonymous id (default %(default)s)",
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
    print(f"{verb} {result['windows']} windows across {result['chats']} chat(s)")
    print(db.stats(conn))


if __name__ == "__main__":
    main()
