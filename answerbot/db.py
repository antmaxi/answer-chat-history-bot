"""SQLite storage: schema, connection helper, and the FTS triggers."""

import sqlite3
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
"""


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the database, creating the schema if it isn't there yet."""
    conn = sqlite3.connect(path or config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def stats(conn: sqlite3.Connection) -> dict:
    """Row counts, for the CLI and the bot's /stats command."""
    def count(table: str) -> int:
        return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    return {
        "messages": count("messages"),
        "windows": count("windows"),
        "embedded": count("window_vecs"),
        "chats": count("(SELECT DISTINCT chat_id FROM messages)"),
    }
