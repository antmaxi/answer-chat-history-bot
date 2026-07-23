"""Resolve people's real display names, overriding the export's private labels.

A Telegram Desktop export stores each message's `from` as the name the exporting
account had saved — so contacts appear under that account's private labels, which
then leak into embeddings, prompts, and answers shown to everyone. This module
keeps a `people` table, keyed by the stable telegram user id, mapping to a real
display name from one of three sources (most to least trusted):

  manual  a mapping you supply and edit by hand (--import)
  api     resolved via the Bot API (the bot's /resolve command)
  live    picked up automatically from incoming messages as people post

A more trusted name is never clobbered by a less trusted one. Indexing renders
windows through this map (see index.render), so resolved names replace the
labels after a reindex.
"""

import sqlite3
import time

# Higher wins; a live name won't overwrite one you set by hand.
_TRUST = {"live": 0, "api": 1, "manual": 2}


def record(
    conn: sqlite3.Connection,
    sender_id: int | None,
    display_name: str | None,
    username: str | None = None,
    source: str = "live",
) -> bool:
    """Upsert a name unless a more trusted one is already stored. Returns True if written."""
    if sender_id is None or not display_name:
        return False
    row = conn.execute("SELECT source FROM people WHERE sender_id=?", (sender_id,)).fetchone()
    if row and _TRUST.get(source, 0) < _TRUST.get(row["source"], 0):
        return False
    conn.execute(
        """INSERT INTO people (sender_id, display_name, username, source, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT (sender_id) DO UPDATE SET
             display_name=excluded.display_name,
             username=excluded.username,
             source=excluded.source,
             updated_at=excluded.updated_at""",
        (sender_id, display_name.strip(), username, source, int(time.time())),
    )
    return True


def name_map(conn: sqlite3.Connection) -> dict[int, str]:
    """All resolved names, for use during indexing."""
    return {
        r["sender_id"]: r["display_name"]
        for r in conn.execute("SELECT sender_id, display_name FROM people")
    }


def resolve(names: dict[int, str], sender_id: int | None, fallback: str) -> str:
    """Resolved name for a sender, or the export label if we don't have one."""
    if sender_id is not None:
        name = names.get(sender_id)
        if name:
            return name
    return fallback


# --- Local, no-API workflow: dump a template, edit it, load it back ----------

def build_template(conn: sqlite3.Connection) -> list[dict]:
    """Every distinct sender with its current label and message count, busiest
    first — edit the `name` values and feed the file back with --import."""
    rows = conn.execute(
        """SELECT m.sender_id AS sender_id,
                  count(*) AS messages,
                  COALESCE(p.display_name, max(m.sender)) AS name,
                  p.source AS source
           FROM messages m
           LEFT JOIN people p ON p.sender_id = m.sender_id
           WHERE m.sender_id IS NOT NULL
           GROUP BY m.sender_id
           ORDER BY messages DESC""",
    ).fetchall()
    return [
        {"sender_id": r["sender_id"], "messages": r["messages"], "name": r["name"],
         "source": r["source"] or "export-label"}
        for r in rows
    ]


def load_mapping(conn: sqlite3.Connection, items: list[dict], source: str = "manual") -> int:
    """Apply a list of {sender_id, name[, username]} entries."""
    written = 0
    for it in items:
        if record(conn, it.get("sender_id"), it.get("name"), it.get("username"), source):
            written += 1
    conn.commit()
    return written


def main() -> None:
    import argparse
    import json

    from . import db

    ap = argparse.ArgumentParser(description="Manage real display names for chat members")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--template", metavar="FILE", help="write an editable name template (JSON)")
    g.add_argument("--import", dest="import_", metavar="FILE", help="load a name mapping (JSON)")
    g.add_argument("--stats", action="store_true", help="show how many names are resolved")
    args = ap.parse_args()

    conn = db.connect()

    if args.template:
        rows = build_template(conn)
        with open(args.template, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
        print(f"wrote {len(rows)} people to {args.template} — edit the \"name\" fields, then "
              f"reload with --import, then reindex")
    elif args.import_:
        with open(args.import_, encoding="utf-8") as fh:
            items = json.load(fh)
        n = load_mapping(conn, items)
        print(f"applied {n} names. Run `python -m answerbot.index` to rewrite history with them.")
    else:
        total = conn.execute(
            "SELECT count(DISTINCT sender_id) FROM messages WHERE sender_id IS NOT NULL"
        ).fetchone()[0]
        by_src = dict(conn.execute("SELECT source, count(*) FROM people GROUP BY source").fetchall())
        resolved = sum(by_src.values())
        print(f"people in chat: {total}")
        print(f"resolved names: {resolved} ({by_src})")
        print(f"still on export labels: {total - resolved}")


if __name__ == "__main__":
    main()
