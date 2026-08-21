"""SQLite storage: schema, connection helper, and the FTS triggers."""

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY,
  chat_id    INTEGER NOT NULL,
  msg_id     INTEGER NOT NULL,
  reply_to   INTEGER,
  sender_id  INTEGER,
  sender     TEXT,
  ts         INTEGER NOT NULL,
  text       TEXT NOT NULL,
  UNIQUE (chat_id, msg_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages (chat_id, ts);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  text, content='messages', content_rowid='id', tokenize='unicode61'
);

-- Keep the FTS index in sync with the messages table.
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts (rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts (messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts (messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO messages_fts (rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS windows (
  id        INTEGER PRIMARY KEY,
  chat_id   INTEGER NOT NULL,
  first_msg INTEGER NOT NULL,
  last_msg  INTEGER NOT NULL,
  ts_start  INTEGER NOT NULL,
  ts_end    INTEGER NOT NULL,
  speakers  TEXT,
  text      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_windows_range ON windows (chat_id, first_msg, last_msg);

CREATE TABLE IF NOT EXISTS window_vecs (
  window_id INTEGER PRIMARY KEY REFERENCES windows(id) ON DELETE CASCADE,
  vec       BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS state (
  chat_id             INTEGER PRIMARY KEY,
  last_indexed_msg_id INTEGER NOT NULL DEFAULT 0
);

-- Real display names keyed by the stable telegram user id, so the exporter's
-- private contact labels (stored in messages.sender) can be overridden. Filled
-- from live messages, an API backfill, or a manual mapping — see people.py.
CREATE TABLE IF NOT EXISTS people (
  sender_id    INTEGER PRIMARY KEY,
  display_name TEXT NOT NULL,
  username     TEXT,
  source       TEXT NOT NULL DEFAULT 'live',
  updated_at   INTEGER NOT NULL
);

-- Stable anonymous ordinals for SPEAKER_LABEL=id, and the fallback under
-- SPEAKER_LABEL=name: a sequential "User N" assigned once so it doesn't
-- expose the real telegram id or the exporter's contact labels.
CREATE TABLE IF NOT EXISTS aliases (
  sender_id INTEGER PRIMARY KEY,
  ordinal   INTEGER NOT NULL UNIQUE
);

-- DM: last chat the user chose with /chat (unused; the bot is pinned to one chat).
CREATE TABLE IF NOT EXISTS dm_prefs (
  user_id INTEGER PRIMARY KEY,
  chat_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS query_log (
  id          INTEGER PRIMARY KEY,
  ts          INTEGER NOT NULL,
  question    TEXT NOT NULL,
  chat_ids    TEXT,
  window_ids  TEXT NOT NULL,
  cited_ids   TEXT NOT NULL,
  latency_ms  INTEGER,
  model       TEXT
);
"""


def connect(path: Path | str | None = None, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open the database, creating the schema if it isn't there yet.

    The bot runs DB work in a thread pool (via asyncio.to_thread), so it opens
    with check_same_thread=False and serializes access with its own lock — see
    bot.py. Single-threaded callers (the CLIs) keep the default guard.
    """
    db_path = Path(path or config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    except sqlite3.OperationalError as e:
        raise sqlite3.OperationalError(
            f"cannot open {db_path} ({e}); is the directory writable?"
        ) from e
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    migrate(conn)
    return conn


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> bool:
    """ADD COLUMN if it is missing. Returns True if the column was added.

    Table/column names must be simple identifiers — this is a migration helper,
    not a general SQL builder.
    """
    if not _IDENT.match(table) or not _IDENT.match(column):
        raise ValueError("table and column must be simple identifiers")
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column in cols:
        return False
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.commit()
    return True


def migrate(conn: sqlite3.Connection) -> None:
    """Apply additive schema changes that CREATE IF NOT EXISTS cannot cover.

    New tables go in SCHEMA. New columns on existing tables go here via
    ensure_column, so a DB created on an older commit still opens.
    """
    return


def _iso_utc(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(int(ts), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def stats(conn: sqlite3.Connection) -> dict:
    """Row counts and the indexed message span, for the CLI and /stats."""
    def count(table: str) -> int:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    first_ts, last_ts = conn.execute("SELECT MIN(ts), MAX(ts) FROM messages").fetchone()
    return {
        "messages": count("messages"),
        "windows": count("windows"),
        "embedded": count("window_vecs"),
        "chats": count("(SELECT DISTINCT chat_id FROM messages)"),
        "first_message": _iso_utc(first_ts),
        "last_message": _iso_utc(last_ts),
    }


def log_query(
    conn: sqlite3.Connection,
    *,
    question: str,
    chat_ids: list[int] | None,
    window_ids: list[int],
    cited_ids: list[int],
    latency_ms: int,
    model: str | None,
) -> None:
    """Append one answered question. No-op when QUERY_LOG is off."""
    if not config.QUERY_LOG:
        return
    conn.execute(
        """INSERT INTO query_log (ts, question, chat_ids, window_ids, cited_ids, latency_ms, model)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            int(time.time()),
            question,
            json.dumps(chat_ids) if chat_ids is not None else None,
            json.dumps(window_ids),
            json.dumps(cited_ids),
            latency_ms,
            model,
        ),
    )
    conn.commit()


def get_dm_chat(conn: sqlite3.Connection, user_id: int) -> int | None:
    row = conn.execute("SELECT chat_id FROM dm_prefs WHERE user_id=?", (user_id,)).fetchone()
    return int(row[0]) if row else None


def set_dm_chat(conn: sqlite3.Connection, user_id: int, chat_id: int | None) -> None:
    if chat_id is None:
        conn.execute("DELETE FROM dm_prefs WHERE user_id=?", (user_id,))
    else:
        conn.execute(
            """INSERT INTO dm_prefs (user_id, chat_id) VALUES (?, ?)
               ON CONFLICT (user_id) DO UPDATE SET chat_id=excluded.chat_id""",
            (user_id, chat_id),
        )
    conn.commit()


def remap_chat_id(conn: sqlite3.Connection, old: int, new: int) -> int:
    """Move every row from `old` chat_id to `new`. Returns how many messages moved.

    Stitches a Telegram Desktop export (bare positive id) onto the Bot API id.
    If both already exist, overlapping message ids keep the destination row
    (live text wins); windows from both sides are kept so embeddings survive.
    """
    if old == new:
        return 0
    existed = conn.execute(
        "SELECT count(*) FROM messages WHERE chat_id=?", (old,)
    ).fetchone()[0]
    if not existed:
        return 0

    dest_msgs = conn.execute(
        "SELECT count(*) FROM messages WHERE chat_id=?", (new,)
    ).fetchone()[0]
    if dest_msgs == 0:
        conn.execute("UPDATE messages SET chat_id=? WHERE chat_id=?", (new, old))
    else:
        conn.execute(
            """INSERT INTO messages (chat_id, msg_id, reply_to, sender_id, sender, ts, text)
               SELECT ?, msg_id, reply_to, sender_id, sender, ts, text
               FROM messages WHERE chat_id=?
               ON CONFLICT (chat_id, msg_id) DO NOTHING""",
            (new, old),
        )
        conn.execute("DELETE FROM messages WHERE chat_id=?", (old,))

    conn.execute("UPDATE windows SET chat_id=? WHERE chat_id=?", (new, old))
    conn.execute("UPDATE dm_prefs SET chat_id=? WHERE chat_id=?", (new, old))

    old_wm = conn.execute(
        "SELECT last_indexed_msg_id FROM state WHERE chat_id=?", (old,)
    ).fetchone()
    new_wm = conn.execute(
        "SELECT last_indexed_msg_id FROM state WHERE chat_id=?", (new,)
    ).fetchone()
    if old_wm and not new_wm:
        conn.execute("UPDATE state SET chat_id=? WHERE chat_id=?", (new, old))
    elif old_wm and new_wm:
        conn.execute(
            "UPDATE state SET last_indexed_msg_id=? WHERE chat_id=?",
            (max(old_wm[0], new_wm[0]), new),
        )
        conn.execute("DELETE FROM state WHERE chat_id=?", (old,))
    else:
        conn.execute("DELETE FROM state WHERE chat_id=?", (old,))

    conn.commit()
    return existed
