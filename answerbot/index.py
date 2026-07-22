"""Group messages into conversation windows and embed them.

Windows, not individual messages, are the retrieval unit. Chat messages are
short and context-free on their own ("yeah, that one"), so embedding them
individually retrieves noise.
"""

import sqlite3
from datetime import datetime, timezone

from . import config, db, embed


def render(msgs: list[sqlite3.Row]) -> str:
    """Render a window as a plain transcript, which is what gets embedded."""
    lines = []
    for m in msgs:
        stamp = datetime.fromtimestamp(m["ts"], timezone.utc).strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{stamp}] {m['sender']}: {m['text']}")
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


def reindex(conn: sqlite3.Connection, chat_id: int | None = None, progress: bool = True) -> dict:
    """Rebuild windows and embeddings from scratch for one chat, or for all."""
    if chat_id is None:
        chat_ids = [r[0] for r in conn.execute("SELECT DISTINCT chat_id FROM messages")]
    else:
        chat_ids = [chat_id]

    total_windows = 0
    for cid in chat_ids:
        conn.execute(
            "DELETE FROM window_vecs WHERE window_id IN (SELECT id FROM windows WHERE chat_id=?)",
            (cid,),
        )
        conn.execute("DELETE FROM windows WHERE chat_id=?", (cid,))

        msgs = conn.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY ts, msg_id", (cid,)
        ).fetchall()
        if not msgs:
            continue

        groups = build_windows(msgs)
        rows = []
        for g in groups:
            speakers = sorted({m["sender"] for m in g if m["sender"]})
            rows.append(
                (
                    cid,
                    g[0]["msg_id"],
                    g[-1]["msg_id"],
                    g[0]["ts"],
                    g[-1]["ts"],
                    ", ".join(speakers),
                    render(g),
                )
            )

        conn.executemany(
            """INSERT INTO windows (chat_id, first_msg, last_msg, ts_start, ts_end, speakers, text)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()

        stored = conn.execute(
            "SELECT id, text FROM windows WHERE chat_id=? ORDER BY id", (cid,)
        ).fetchall()
        vecs = embed.encode_passages([r["text"] for r in stored], progress=progress)
        conn.executemany(
            "INSERT OR REPLACE INTO window_vecs (window_id, vec) VALUES (?, ?)",
            [(r["id"], embed.pack(v)) for r, v in zip(stored, vecs)],
        )

        last = conn.execute(
            "SELECT max(msg_id) FROM messages WHERE chat_id=?", (cid,)
        ).fetchone()[0]
        conn.execute(
            """INSERT INTO state (chat_id, last_indexed_msg_id) VALUES (?, ?)
               ON CONFLICT (chat_id) DO UPDATE SET last_indexed_msg_id=excluded.last_indexed_msg_id""",
            (cid, last),
        )
        conn.commit()
        total_windows += len(rows)

    return {"chats": len(chat_ids), "windows": total_windows}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Build conversation windows and embeddings")
    ap.add_argument("--chat-id", type=int, default=None)
    args = ap.parse_args()

    conn = db.connect()
    result = reindex(conn, args.chat_id)
    print(f"indexed {result['windows']} windows across {result['chats']} chat(s)")
    print(db.stats(conn))


if __name__ == "__main__":
    main()
