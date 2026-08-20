"""Parse a Telegram Desktop JSON export into the messages table.

Produce the export from Telegram Desktop: chat menu -> Export chat history,
format "JSON", which writes a result.json. Media can be left out entirely; we
only read text.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .. import db, logconfig

# Service messages (joins, pins, calls) carry no conversational content.
SKIP_TYPES = {"service"}

# Telegram Desktop writes the bare channel/group id. The Bot API encodes the
# peer type in the number: `-100<id>` for supergroups/channels, `-id` for
# basic groups. Personal chats already match.
_CHANNEL_TYPES = {
    "private_supergroup",
    "public_supergroup",
    "private_channel",
    "public_channel",
}
_BASIC_GROUP_TYPES = {"private_group", "public_group", "group"}


def bot_api_chat_id(export: dict) -> int:
    """Map a Desktop JSON `id` onto the chat id the Bot API uses."""
    raw = int(export["id"])
    if raw < 0:
        return raw
    typ = str(export.get("type") or "")
    if typ in _CHANNEL_TYPES or "supergroup" in typ or typ.endswith("channel"):
        return int(f"-100{raw}")
    if typ in _BASIC_GROUP_TYPES:
        return -raw
    return raw


def desktop_ids_for(bot_api_id: int) -> list[int]:
    """Desktop-export ids that might already be stored for this Bot API chat."""
    text = str(bot_api_id)
    aliases: list[int] = []
    if text.startswith("-100") and len(text) > 4:
        aliases.append(int(text[4:]))
    elif bot_api_id < 0:
        aliases.append(-bot_api_id)
    return aliases


def bot_api_candidates(stored_id: int) -> list[int]:
    """Ids to try when resolving a stored chat against the Bot API.

    The stored value itself is first (already-correct, or a personal chat).
    Positive Desktop leftovers then try the supergroup and basic-group forms.
    """
    if stored_id < 0:
        return [stored_id]
    return [stored_id, int(f"-100{stored_id}"), -stored_id]


def flatten_text(raw) -> str:
    """Telegram stores text as a string, or a list of plain strings and entity dicts."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def parse_sender_id(raw) -> int | None:
    """from_id looks like "user123456" or "channel123456"; we want the digits."""
    if raw is None:
        return None
    digits = "".join(c for c in str(raw) if c.isdigit())
    return int(digits) if digits else None


def parse_ts(msg: dict) -> int | None:
    """Prefer date_unixtime; fall back to parsing the ISO date."""
    if "date_unixtime" in msg:
        try:
            return int(msg["date_unixtime"])
        except (TypeError, ValueError):
            pass
    if "date" in msg:
        try:
            return int(datetime.fromisoformat(msg["date"]).timestamp())
        except (TypeError, ValueError):
            pass
    return None


def iter_messages(export: dict, chat_id: int) -> Iterator[tuple]:
    """Yield insertable rows, skipping anything without usable text."""
    for msg in export.get("messages", []):
        if msg.get("type") in SKIP_TYPES:
            continue

        text = flatten_text(msg.get("text", "")).strip()
        if not text:
            # Media with no caption, stickers, polls: nothing to search on.
            continue

        ts = parse_ts(msg)
        msg_id = msg.get("id")
        if ts is None or msg_id is None:
            continue

        sender = msg.get("from") or msg.get("actor") or "Unknown"

        yield (
            chat_id,
            int(msg_id),
            msg.get("reply_to_message_id"),
            parse_sender_id(msg.get("from_id")),
            sender,
            ts,
            text,
        )


def load_data(conn: sqlite3.Connection, export: dict, source: str = "export") -> dict:
    """Insert an already-parsed export. Safe to re-run: rows are upserted."""
    if export.get("id") is None:
        raise ValueError(f"{source}: no chat id in export — is this a result.json?")

    raw_id = int(export["id"])
    chat_id = bot_api_chat_id(export)
    if raw_id != chat_id:
        db.remap_chat_id(conn, raw_id, chat_id)

    rows = list(iter_messages(export, chat_id))
    before = conn.execute("SELECT count(*) FROM messages").fetchone()[0]

    conn.executemany(
        """INSERT INTO messages (chat_id, msg_id, reply_to, sender_id, sender, ts, text)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT (chat_id, msg_id) DO UPDATE SET text=excluded.text""",
        rows,
    )
    conn.commit()

    after = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    return {
        "chat_id": int(chat_id),
        "chat_name": export.get("name", "?"),
        "in_file": len(export.get("messages", [])),
        "usable": len(rows),
        "inserted": after - before,
    }


def load(conn: sqlite3.Connection, path: Path | str) -> dict:
    """Read an export file into the database. Safe to re-run: rows are upserted."""
    with open(path, encoding="utf-8") as fh:
        export = json.load(fh)
    return load_data(conn, export, source=str(path))


def main() -> None:
    import argparse

    logconfig.setup()
    ap = argparse.ArgumentParser(description="Load a Telegram JSON export")
    ap.add_argument("path", help="path to result.json")
    args = ap.parse_args()

    conn = db.connect()
    result = load(conn, args.path)
    print(
        f"{result['chat_name']} (chat {result['chat_id']}): "
        f"{result['in_file']} messages in file, {result['usable']} with text, "
        f"{result['inserted']} new"
    )


if __name__ == "__main__":
    main()
