"""Append messages arriving over the Bot API and keep the index fresh.

The Bot API only delivers messages sent after the bot joined, and only in
groups where privacy mode is off (set via BotFather). The one-time export seeds
everything before that; this keeps it live.

Re-embedding the whole chat on every message would be wasteful, so new messages
land in `messages` immediately. They only become retrieval units once they sit
in a window (keyword search joins FTS hits to windows by msg_id range), so the
tail is re-windowed in batches and flushed again just before answering.
"""

from collections.abc import Sequence

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


def flush_tail(conn: sqlite3.Connection, chat_id: int | Sequence[int] | None = None) -> int:
    """Re-window any unindexed tail so live messages are searchable.

    `chat_id` follows retrieve.ChatId: one chat, an allow-list, or None for every
    indexed chat. A no-op when nothing new has arrived.
    """
    chats = retrieve.normalize_chat_ids(chat_id)
    if chats is not None and not chats:
        return 0
    total = 0
    if chats is None:
        total = index.update(conn)["windows"]
    else:
        for cid in chats:
            total += index.update(conn, cid)["windows"]
    retrieve.invalidate_cache()
    return total


def reindex_tail(conn: sqlite3.Connection, chat_id: int) -> int:
    """Re-window and embed only the messages since the last watermark."""
    return flush_tail(conn, chat_id)


def maybe_reindex(conn: sqlite3.Connection, chat_id: int) -> bool:
    """Re-window the tail once enough messages have accumulated. Returns True if it ran."""
    if pending_count(conn, chat_id) >= config.LIVE_REINDEX_EVERY:
        reindex_tail(conn, chat_id)
        return True
    return False


def refresh_if_in_tail(conn: sqlite3.Connection, chat_id: int, msg_id: int) -> bool:
    """Rebuild the open tail if this message sits in (or after) it.

    Live edits update `messages` (and FTS) immediately, but the window
    transcript fed to the LLM stays stale until those windows are rebuilt.
    Older edits are left for a lookback or full reindex — same as CLI `--update`.
    """
    boundary = index._rebuild_boundary(conn, chat_id)
    if msg_id < boundary:
        return False
    index.update(conn, chat_id, force=True)
    retrieve.invalidate_cache()
    return True
