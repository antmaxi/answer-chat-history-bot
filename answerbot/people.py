"""Resolve people's real display names, overriding the export's private labels.

A Telegram Desktop export stores each message's `from` as the name the exporting
account had saved — so contacts appear under that account's private labels. This
module keeps a `people` table, keyed by the stable telegram user id, mapping to
a real public display name from one of three sources (most to least trusted):

  manual  a mapping you supply and edit by hand (--import)
  api     resolved via the Bot API (the bot's /resolve command)
  live    picked up automatically from incoming messages as people post

A more trusted name is never clobbered by a less trusted one. Indexing renders
windows through this map (see index.render). Unresolved people are shown as
stable "User N" aliases unless SPEAKER_LABEL=export, which opts back into the
exporter's contact labels.
"""

import html
import re
import sqlite3
import time

from . import config, i18n, logconfig

# Higher wins; a live name won't overwrite one you set by hand.
_TRUST = {"live": 0, "api": 1, "manual": 2}


def clean_display_name(name: str | None) -> str:
    """Keep emoji, ZWJ sequences, and styled unicode; only trim ASCII whitespace.

    Do not NFKC-normalize: Telegram names may use mathematical-bold letters and
    similar styled alphabets, which compatibility folding would flatten.
    """
    if not name:
        return ""
    return name.strip(" \t\r\n\f\v")


def name_from_user(user) -> str | None:
    """Public display name from a Bot API User, including emoji in first/last name.

    Custom premium emoji in names are not given as entity ids on User; the API
    supplies a fallback unicode emoji in first_name/last_name, which we keep.
    """
    if user is None:
        return None
    first = getattr(user, "first_name", None) or ""
    last = getattr(user, "last_name", None) or ""
    combined = " ".join(p for p in (first, last) if p)
    if not combined:
        combined = getattr(user, "full_name", None) or ""
    return clean_display_name(combined) or None


def clear_miss(conn: sqlite3.Connection, sender_id: int | None) -> None:
    if sender_id is None:
        return
    conn.execute("DELETE FROM resolve_misses WHERE sender_id=?", (sender_id,))


def mark_miss(conn: sqlite3.Connection, sender_id: int | None, reason: str = "left") -> None:
    """Remember a failed getChatMember so the next /resolve pass can skip it."""
    if sender_id is None:
        return
    conn.execute(
        """INSERT INTO resolve_misses (sender_id, reason, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT (sender_id) DO UPDATE SET
             reason=excluded.reason,
             updated_at=excluded.updated_at""",
        (sender_id, reason, int(time.time())),
    )


def miss_reason_from_error(message: str) -> str:
    text = (message or "").lower()
    if "inaccessible" in text or "chat_admin_required" in text:
        return "hidden"
    if "forbidden" in text:
        return "forbidden"
    if "not found" in text or "participant" in text or "invalid user" in text:
        return "left"
    return "error"


def pending_ids(
    conn: sqlite3.Connection, chat_id: int, retry_misses: bool = False
) -> list[int]:
    """Sender ids that still need an API name lookup, in a stable order.

    Anyone already in `people` is skipped. Failed lookups stay in
    `resolve_misses` and are skipped unless `retry_misses` is true.
    """
    miss_clause = "" if retry_misses else (
        "AND NOT EXISTS (SELECT 1 FROM resolve_misses r WHERE r.sender_id = m.sender_id)"
    )
    rows = conn.execute(
        f"""SELECT DISTINCT m.sender_id FROM messages m
            WHERE m.chat_id=? AND m.sender_id IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM people p WHERE p.sender_id = m.sender_id)
              {miss_clause}
            ORDER BY m.sender_id""",
        (chat_id,),
    )
    return [r[0] for r in rows]


def lookup_stats(conn: sqlite3.Connection, chat_id: int | None = None) -> dict[str, int]:
    """How many senders are named, skipped as API misses, or still pending."""
    if chat_id is None:
        msg_where = "m.sender_id IS NOT NULL"
        params: tuple = ()
    else:
        msg_where = "m.chat_id=? AND m.sender_id IS NOT NULL"
        params = (chat_id,)
    total = conn.execute(
        f"SELECT count(DISTINCT m.sender_id) FROM messages m WHERE {msg_where}",
        params,
    ).fetchone()[0]
    resolved = conn.execute(
        f"""SELECT count(DISTINCT m.sender_id) FROM messages m
            JOIN people p ON p.sender_id = m.sender_id
            WHERE {msg_where}""",
        params,
    ).fetchone()[0]
    missed = conn.execute(
        f"""SELECT count(DISTINCT m.sender_id) FROM messages m
            JOIN resolve_misses r ON r.sender_id = m.sender_id
            WHERE {msg_where}
              AND NOT EXISTS (SELECT 1 FROM people p WHERE p.sender_id = m.sender_id)""",
        params,
    ).fetchone()[0]
    total = int(total or 0)
    resolved = int(resolved or 0)
    missed = int(missed or 0)
    return {
        "total": total,
        "resolved": resolved,
        "missed": missed,
        "pending": total - resolved - missed,
    }


def record(
    conn: sqlite3.Connection,
    sender_id: int | None,
    display_name: str | None,
    username: str | None = None,
    source: str = "live",
) -> bool:
    """Upsert a name unless a more trusted one is already stored. Returns True if written."""
    display_name = clean_display_name(display_name)
    if sender_id is None or not display_name:
        return False
    row = conn.execute("SELECT source FROM people WHERE sender_id=?", (sender_id,)).fetchone()
    if row and _TRUST.get(source, 0) < _TRUST.get(row["source"], 0):
        clear_miss(conn, sender_id)
        return False
    conn.execute(
        """INSERT INTO people (sender_id, display_name, username, source, updated_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT (sender_id) DO UPDATE SET
             display_name=excluded.display_name,
             username=excluded.username,
             source=excluded.source,
             updated_at=excluded.updated_at""",
        (sender_id, display_name, username, source, int(time.time())),
    )
    clear_miss(conn, sender_id)
    return True


def name_map(conn: sqlite3.Connection) -> dict[int, str]:
    """All resolved names, for use during indexing."""
    return {
        r["sender_id"]: r["display_name"]
        for r in conn.execute("SELECT sender_id, display_name FROM people")
    }


def resolve(names: dict[int, str], sender_id: int | None, fallback: str) -> str:
    """Resolved name for a sender, or `fallback` if we don't have one."""
    if sender_id is not None:
        name = names.get(sender_id)
        if name:
            return name
    return fallback


def alias_label(sender_id: int | None, aliases: dict[int, int] | None = None) -> str:
    """Stable anonymous "User N", never the real telegram id when an ordinal exists."""
    if sender_id is None:
        return "User unknown"
    ordinal = (aliases or {}).get(sender_id)
    return f"User {ordinal}" if ordinal is not None else f"User #{sender_id}"


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


_WHO_ALIAS = re.compile(r"(?i)^(?:user\s*#?\s*|\#)(\d+)$")
_WHO_ID = re.compile(r"^-?\d+$")


def parse_who_arg(arg: str) -> tuple[str, int] | None:
    """Parse `/who` input as ('id', telegram_id) or ('alias', ordinal)."""
    text = (arg or "").strip()
    if not text:
        return None
    m = _WHO_ALIAS.match(text)
    if m:
        n = int(m.group(1))
        return ("alias", n) if n > 0 else None
    if _WHO_ID.match(text):
        n = int(text)
        return ("id", n) if n != 0 else None
    return None


def sender_id_for_alias(conn: sqlite3.Connection, ordinal: int) -> int | None:
    row = conn.execute("SELECT sender_id FROM aliases WHERE ordinal=?", (ordinal,)).fetchone()
    return int(row["sender_id"]) if row else None


def whois(conn: sqlite3.Connection, sender_id: int) -> dict:
    """Local facts about a sender: public name, @username, User N alias, export label."""
    person = conn.execute(
        "SELECT display_name, username, source FROM people WHERE sender_id=?",
        (sender_id,),
    ).fetchone()
    alias = conn.execute(
        "SELECT ordinal FROM aliases WHERE sender_id=?", (sender_id,)
    ).fetchone()
    export = conn.execute(
        """SELECT sender FROM messages
           WHERE sender_id=? AND sender IS NOT NULL AND sender != ''
           ORDER BY ts DESC LIMIT 1""",
        (sender_id,),
    ).fetchone()
    messages = conn.execute(
        "SELECT count(*) FROM messages WHERE sender_id=?", (sender_id,)
    ).fetchone()[0]
    display_name = person["display_name"] if person else None
    export_name = export["sender"] if export else None
    if export_name and display_name and export_name == display_name:
        export_name = None
    return {
        "sender_id": sender_id,
        "display_name": display_name,
        "username": (person["username"] or None) if person else None,
        "source": person["source"] if person else None,
        "alias": int(alias["ordinal"]) if alias else None,
        "export_name": export_name,
        "messages": int(messages or 0),
    }


def format_who(lang: str, info: dict) -> str:
    """HTML body for the admin `/who` reply. Names are escaped for parse_mode."""
    lines = [i18n.t(lang, "who_id", id=info["sender_id"])]
    name = info.get("display_name")
    if name:
        lines.append(i18n.t(lang, "who_name", name=html.escape(name)))
    username = info.get("username")
    if username:
        handle = str(username).lstrip("@")
        lines.append(i18n.t(lang, "who_username", username=html.escape(handle)))
    alias = info.get("alias")
    if alias:
        lines.append(i18n.t(lang, "who_alias", n=alias))
    export_name = info.get("export_name")
    if export_name and not name:
        lines.append(i18n.t(lang, "who_export", name=html.escape(export_name)))
    source = info.get("source")
    if source:
        lines.append(i18n.t(lang, "who_source", source=source))
    messages = int(info.get("messages") or 0)
    if messages:
        lines.append(i18n.t(lang, "who_messages", n=messages))
    return "\n".join(lines)


def who_has_local_info(info: dict) -> bool:
    return bool(
        info.get("display_name")
        or info.get("username")
        or info.get("alias")
        or info.get("export_name")
        or info.get("messages")
    )


def known_speakers(conn: sqlite3.Connection) -> list[str]:
    """Display names we might see in a 'what did X say' question, longest first.

    Export/contact labels are included only under SPEAKER_LABEL=export, matching
    what actually appears in window speaker fields.
    """
    names: set[str] = set()
    for (n,) in conn.execute("SELECT display_name FROM people WHERE display_name != ''"):
        names.add(n)
    if config.SPEAKER_LABEL == "export":
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
    fallback: str = "",
    mode: str | None = None,
    aliases: dict[int, int] | None = None,
) -> str:
    """How a speaker is shown, honouring SPEAKER_LABEL.

    name     resolved public name, else "User N". Contact/export labels are ignored.
    id       always "User N" — not resolved names, not export labels.
    export   resolved public name, else the stored export/contact label (opt-in).
    """
    mode = (mode or config.SPEAKER_LABEL or "name").strip().lower()
    if mode == "id":
        return alias_label(sender_id, aliases)
    resolved = resolve(names, sender_id, "")
    if resolved:
        return resolved
    if mode == "export" and fallback:
        return fallback
    return alias_label(sender_id, aliases)


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

    logconfig.setup()
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
        s = lookup_stats(conn)
        by_src = dict(conn.execute("SELECT source, count(*) FROM people GROUP BY source").fetchall())
        print(f"people in chat: {s['total']}")
        print(f"resolved names: {s['resolved']} ({by_src})")
        print(f"skipped (left / not found): {s['missed']}")
        print(f"pending API lookup: {s['pending']}")
        print(f"unresolved (User N unless SPEAKER_LABEL=export): {s['total'] - s['resolved']}")


if __name__ == "__main__":
    main()
