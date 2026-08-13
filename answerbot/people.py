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

import re
import sqlite3
import time

from . import config

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


def ensure_aliases(conn: sqlite3.Connection, sender_ids) -> None:
    """Assign a persistent ordinal to any sender that lacks one.

    Ordinals are handed out in ascending id order and never reused, so a given
    person keeps the same "User N" across reindexes without that N revealing
    their telegram id."""
    ids = {s for s in sender_ids if s is not None}
    if not ids:
        return
    have = {r[0] for r in conn.execute("SELECT sender_id FROM aliases")}
    missing = sorted(ids - have)
    if not missing:
        return
    start = conn.execute("SELECT COALESCE(MAX(ordinal), 0) FROM aliases").fetchone()[0] + 1
    conn.executemany(
        "INSERT OR IGNORE INTO aliases (sender_id, ordinal) VALUES (?, ?)",
        [(sid, start + i) for i, sid in enumerate(missing)],
    )
    conn.commit()


def alias_map(conn: sqlite3.Connection) -> dict[int, int]:
    return {r["sender_id"]: r["ordinal"] for r in conn.execute("SELECT sender_id, ordinal FROM aliases")}


def known_speakers(conn: sqlite3.Connection) -> list[str]:
    """Display names we might see in a 'what did X say' question, longest first."""
    names: set[str] = set()
    for (n,) in conn.execute("SELECT display_name FROM people WHERE display_name != ''"):
        names.add(n)
    for (n,) in conn.execute(
        "SELECT DISTINCT sender FROM messages WHERE sender IS NOT NULL AND sender != ''"
    ):
        names.add(n)
    return sorted(names, key=lambda s: (-len(s), s.lower()))


_SPEAKER_CUE = re.compile(
    r"(?i)\b(did|said|says|told|tell|from|by|according to)\b"
)


def parse_speaker(question: str, names: list[str]) -> str | None:
    """If the question asks what a known person said, return that name.

    Longest name wins so 'Anna Maria' is preferred over 'Anna'. Names shorter
    than 3 characters are ignored — they collide with common words.
    """
    if not _SPEAKER_CUE.search(question):
        return None
    q = question.lower()
    for name in names:
        if len(name) < 3:
            continue
        n = re.escape(name.lower())
        if re.search(
            rf"(?i)\b(?:what |who )?did {n} (?:say|tell|ask)\b"
            rf"|\b{n} (?:said|says|told|wrote)\b"
            rf"|\b(?:from|by|according to) {n}\b",
            q,
        ):
            return name
    return None


def speaker_label(
    names: dict[int, str],
    sender_id: int | None,
    fallback: str,
    mode: str | None = None,
    aliases: dict[int, int] | None = None,
) -> str:
    """How a speaker is shown, honouring SPEAKER_LABEL.

    In "id" mode nobody's name appears at all — not resolved names, not export
    labels — only a stable anonymous "User N" from the aliases table (assigned by
    ensure_aliases), so the label never exposes the real telegram id.
    """
    if (mode or config.SPEAKER_LABEL) == "id":
        if sender_id is None:
            return "User unknown"
        ordinal = (aliases or {}).get(sender_id)
        return f"User {ordinal}" if ordinal is not None else f"User #{sender_id}"
    return resolve(names, sender_id, fallback)


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
