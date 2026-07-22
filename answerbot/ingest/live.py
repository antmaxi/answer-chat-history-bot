"""Append messages arriving over the Bot API and keep the index fresh.

The Bot API only delivers messages sent after the bot joined, and only in
groups where privacy mode is off (set via BotFather). The one-time export seeds
everything before that; this keeps it live.

Re-embedding the whole chat on every message would be wasteful, so new messages
land in `messages` immediately (searchable by keyword at once via the FTS
triggers) and the *tail* is re-windowed in batches once enough have piled up.
"""

import sqlite3

from .. import config, embed, retrieve
from ..index import build_windows, render


def add_message(
    conn: sqlite3.Connection,
    chat_id: int,
    msg_id: int,
    sender: str,
    sender_id: int | None,
    ts: int,
    text: str,
    reply_to: int | None = None,
) -> None:
    conn.execute(
        """INSERT INTO messages (chat_id, msg_id, reply_to, sender_id, sender, ts, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (chat_id, msg_id) DO UPDATE SET text=excluded.text""",
        (chat_id, msg_id, reply_to, sender_id, sender, ts, text),
    )
    conn.commit()


def pending_count(conn: sqlite3.Connection, chat_id: int) -> int:
    """Messages that have arrived since the last window was built."""
    last = conn.execute(
        "SELECT last_indexed_msg_id FROM state WHERE chat_id=?", (chat_id,)
    ).fetchone()
    watermark = last[0] if last else 0
    return conn.execute(
        "SELECT count(*) FROM messages WHERE chat_id=? AND msg_id>?", (chat_id, watermark)
    ).fetchone()[0]


def reindex_tail(conn: sqlite3.Connection, chat_id: int) -> int:
    """Re-window only the messages since the last watermark.

    The last stored window is dropped and rebuilt along with the new messages,
    because a window is a moving boundary — the final one is "open" and the next
    message may well belong inside it rather than starting a fresh window.
    """
    last = conn.execute(
        "SELECT last_indexed_msg_id FROM state WHERE chat_id=?", (chat_id,)
    ).fetchone()
    watermark = last[0] if last else 0

    # Find where the last window began, so we can rebuild from a clean boundary.
    last_window = conn.execute(
        "SELECT id, first_msg FROM windows WHERE chat_id=? ORDER BY last_msg DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    rebuild_from = last_window["first_msg"] if last_window else 0

    msgs = conn.execute(
        "SELECT * FROM messages WHERE chat_id=? AND msg_id>=? ORDER BY ts, msg_id",
        (chat_id, rebuild_from),
    ).fetchall()
    if not msgs:
        return 0

    if last_window:
        conn.execute("DELETE FROM windows WHERE chat_id=? AND first_msg>=?", (chat_id, rebuild_from))

    groups = build_windows(msgs)
    rows = []
    for g in groups:
        speakers = sorted({m["sender"] for m in g if m["sender"]})
        rows.append(
            (chat_id, g[0]["msg_id"], g[-1]["msg_id"], g[0]["ts"], g[-1]["ts"],
             ", ".join(speakers), render(g))
        )
    conn.executemany(
        """INSERT INTO windows (chat_id, first_msg, last_msg, ts_start, ts_end, speakers, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()

    fresh = conn.execute(
        "SELECT id, text FROM windows WHERE chat_id=? AND first_msg>=? ORDER BY id",
        (chat_id, rebuild_from),
    ).fetchall()
    if fresh:
        vecs = embed.encode_passages([r["text"] for r in fresh], progress=False)
        conn.executemany(
            "INSERT OR REPLACE INTO window_vecs (window_id, vec) VALUES (?, ?)",
            [(r["id"], embed.pack(v)) for r, v in zip(fresh, vecs)],
        )

    new_watermark = max(m["msg_id"] for m in msgs)
    conn.execute(
        """INSERT INTO state (chat_id, last_indexed_msg_id) VALUES (?, ?)
           ON CONFLICT (chat_id) DO UPDATE SET last_indexed_msg_id=excluded.last_indexed_msg_id""",
        (chat_id, new_watermark),
    )
    conn.commit()

    # The in-memory vector matrix is now stale for this chat.
    retrieve.invalidate_cache()
    return len(rows)


def maybe_reindex(conn: sqlite3.Connection, chat_id: int) -> bool:
    """Re-window the tail once enough messages have accumulated. Returns True if it ran."""
    if pending_count(conn, chat_id) >= config.LIVE_REINDEX_EVERY:
        reindex_tail(conn, chat_id)
        return True
    return False
