"""Append messages arriving over the Bot API and keep the index fresh.

The Bot API only delivers messages sent after the bot joined, and only in
groups where privacy mode is off (set via BotFather). The one-time export seeds
everything before that; this keeps it live.

Re-embedding the whole chat on every message would be wasteful, so new messages
land in `messages` immediately (searchable by keyword at once via the FTS
triggers) and the *tail* is re-windowed in batches once enough have piled up.
The incremental work itself lives in `index.update` — this module only decides
*when* to run it.
"""

import sqlite3

from .. import config, index, retrieve


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
    """Re-window and embed only the messages since the last watermark."""
    n = index.update(conn, chat_id)["windows"]
    retrieve.invalidate_cache()  # the in-memory vector matrix is now stale
    return n


def maybe_reindex(conn: sqlite3.Connection, chat_id: int) -> bool:
    """Re-window the tail once enough messages have accumulated. Returns True if it ran."""
    if pending_count(conn, chat_id) >= config.LIVE_REINDEX_EVERY:
        reindex_tail(conn, chat_id)
        return True
    return False
