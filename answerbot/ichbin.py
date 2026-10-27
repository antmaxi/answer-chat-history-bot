"""Similar people from #ichbin intros.

Each sender who posted the tag gets one profile: those messages joined, capped,
and embedded with the same local model as chat search. Recommendations are
nearest neighbors (cosine as a dot product of normalized vectors). This is not
part of question answering — nothing here is retrieved for /ask.
"""

from __future__ import annotations

import html
import logging
import re
import sqlite3
import time
from dataclasses import dataclass

import numpy as np

from . import config, embed, i18n

log = logging.getLogger("answerbot")

# `#ichbin` as its own token. `#IchBin` matches; `#ichbinhier` and `my#ichbin` do not.
TAG_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])#ichbin\b")
# Telegram usernames are 5–32 characters, starting with a letter.
_USERNAME_RE = re.compile(r"@?([A-Za-z][A-Za-z0-9_]{4,31})\Z")
_EXCERPT_CHARS = 140


def has_tag(text: str | None) -> bool:
    return bool(TAG_RE.search(text or ""))


def parse_username(text: str | None) -> str | None:
    """First token as a Telegram @username, without the @, or None."""
    raw = (text or "").strip()
    if not raw:
        return None
    token = raw.split()[0]
    match = _USERNAME_RE.fullmatch(token)
    if not match:
        return None
    return match.group(1).lower()


def cap_tail(text: str, limit: int) -> str:
    """Keep the latest `limit` characters. `limit <= 0` means no cap.

    When the cut lands in the middle of an earlier message, drop that partial
    line. A single message longer than `limit` has no newline in the tail, so
    the raw tail is kept.
    """
    if limit <= 0 or len(text) <= limit:
        return text
    tail = text[-limit:]
    newline = tail.find("\n")
    if newline != -1 and newline < len(tail) - 1:
        rest = tail[newline + 1 :]
        if rest:
            return rest
    return tail


def message_link(chat_id: int, msg_id: int) -> str:
    """t.me deep link. Supergroup ids (`-100…`) drop that prefix, same as Hit.link."""
    raw = str(int(chat_id))
    raw = raw[4:] if raw.startswith("-100") else raw.lstrip("-")
    return f"https://t.me/c/{raw}/{int(msg_id)}"


def corpus_includes(chat_id: int) -> bool:
    """True when a live message in this chat belongs in the #ichbin corpus.

    The main chat when `TELEGRAM_CHAT_ID` is set; otherwise every chat.
    """
    if config.TELEGRAM_CHAT_ID is None:
        return True
    return int(chat_id) == int(config.TELEGRAM_CHAT_ID)


def forward_sender(origin) -> tuple[str, int | None]:
    """Classify a Telegram forward origin.

    Returns `('user', sender_id)`, `('hidden', None)`, `('other', None)`, or
    `('none', None)`. Duck-typed: aiogram sets `type` to `user`, `hidden_user`,
    `chat`, or `channel`.
    """
    if origin is None:
        return "none", None
    kind = getattr(origin, "type", None)
    if kind is None:
        name = type(origin).__name__
        if name == "MessageOriginUser":
            kind = "user"
        elif name == "MessageOriginHiddenUser":
            kind = "hidden_user"
    if kind == "user":
        user = getattr(origin, "sender_user", None)
        uid = getattr(user, "id", None) if user is not None else None
        if uid is None:
            return "other", None
        return "user", int(uid)
    if kind == "hidden_user":
        return "hidden", None
    return "other", None


@dataclass(frozen=True)
class ProfileJob:
    sender_id: int
    text: str
    msg_count: int
    chat_id: int
    msg_id: int
    embed: bool


@dataclass(frozen=True)
class PersonCard:
    sender_id: int
    name: str
    username: str | None
    text: str
    chat_id: int
    msg_id: int
    msg_count: int


@dataclass(frozen=True)
class Neighbor:
    card: PersonCard
    cosine: float


@dataclass(frozen=True)
class Recommend:
    target: PersonCard
    neighbors: tuple[Neighbor, ...]


def _corpus_chats(chat_id: int | None, all_chats: bool) -> list[int] | None:
    """Chat ids to read, or None for every chat already in the database."""
    if all_chats:
        return None
    if chat_id is not None:
        return [int(chat_id)]
    if config.TELEGRAM_CHAT_ID is not None:
        return [int(config.TELEGRAM_CHAT_ID)]
    return None


def _matching_rows(
    conn: sqlite3.Connection,
    chats: list[int] | None,
    sender_id: int | None,
) -> list[sqlite3.Row]:
    clauses = ["sender_id IS NOT NULL", "instr(lower(text), '#ichbin') > 0"]
    params: list[int] = []
    if chats is not None:
        clauses.append(f"chat_id IN ({','.join('?' * len(chats))})")
        params.extend(chats)
    if sender_id is not None:
        clauses.append("sender_id=?")
        params.append(int(sender_id))
    sql = (
        "SELECT sender_id, chat_id, msg_id, ts, text FROM messages WHERE "
        + " AND ".join(clauses)
        + " ORDER BY sender_id, ts, msg_id"
    )
    return list(conn.execute(sql, params))


def _build_profiles(
    conn: sqlite3.Connection,
    chats: list[int] | None,
    sender_id: int | None,
) -> dict[int, tuple[str, int, int, int]]:
    """sender_id → (text, msg_count, newest chat_id, newest msg_id)."""
    grouped: dict[int, list[sqlite3.Row]] = {}
    for row in _matching_rows(conn, chats, sender_id):
        if not has_tag(row["text"]):
            continue
        grouped.setdefault(int(row["sender_id"]), []).append(row)
    built: dict[int, tuple[str, int, int, int]] = {}
    limit = config.ICHBIN_MAX_CHARS
    for sid, msgs in grouped.items():
        parts = [(m["text"] or "").strip() for m in msgs]
        parts = [part for part in parts if part]
        if not parts:
            continue
        text = cap_tail("\n".join(parts), limit)
        if not text:
            continue
        last = msgs[-1]
        built[sid] = (text, len(msgs), int(last["chat_id"]), int(last["msg_id"]))
    return built


def _existing_profiles(
    conn: sqlite3.Connection,
    chats: list[int] | None,
    sender_id: int | None,
) -> dict[int, sqlite3.Row]:
    clauses: list[str] = []
    params: list[int] = []
    if sender_id is not None:
        clauses.append("sender_id=?")
        params.append(int(sender_id))
    elif chats is not None:
        clauses.append(f"chat_id IN ({','.join('?' * len(chats))})")
        params.extend(chats)
    sql = "SELECT sender_id, text, msg_count, chat_id, msg_id FROM ichbin_profiles"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    return {int(row["sender_id"]): row for row in conn.execute(sql, params)}


def plan_sync(
    conn: sqlite3.Connection,
    chat_id: int | None = None,
    sender_id: int | None = None,
    all_chats: bool = False,
) -> tuple[list[ProfileJob], list[int]]:
    """Profiles to write and sender ids to drop. Does not embed.

    `all_chats` ignores `TELEGRAM_CHAT_ID` and reads every chat in the database.
    Otherwise an explicit `chat_id` wins, then the configured main chat.
    A profile stored for a chat outside this corpus is left alone.
    """
    chats = _corpus_chats(chat_id, all_chats)
    chat_set = set(chats) if chats is not None else None
    built = _build_profiles(conn, chats, sender_id)
    existing = _existing_profiles(conn, chats, sender_id)
    jobs: list[ProfileJob] = []
    for sid in sorted(built):
        text, msg_count, newest_chat, newest_msg = built[sid]
        stored = existing.get(sid)
        if stored is None or stored["text"] != text:
            embed_this = True
        elif (
            int(stored["msg_count"]) != msg_count
            or int(stored["chat_id"]) != newest_chat
            or int(stored["msg_id"]) != newest_msg
        ):
            embed_this = False
        else:
            continue
        jobs.append(
            ProfileJob(
                sender_id=sid,
                text=text,
                msg_count=msg_count,
                chat_id=newest_chat,
                msg_id=newest_msg,
                embed=embed_this,
            )
        )
    delete_ids: list[int] = []
    for sid, row in existing.items():
        if sid in built:
            continue
        if chat_set is not None and int(row["chat_id"]) not in chat_set:
            continue
        delete_ids.append(sid)
    delete_ids.sort()
    return jobs, delete_ids


def apply_sync(
    conn: sqlite3.Connection,
    jobs: list[ProfileJob],
    vecs,
    delete_ids: list[int],
) -> dict[str, int]:
    """Write planned profiles. `vecs` aligns with jobs where `embed` is true."""
    embed_jobs = [job for job in jobs if job.embed]
    if embed_jobs and (vecs is None or len(vecs) != len(embed_jobs)):
        raise ValueError(
            f"ichbin embed count {0 if vecs is None else len(vecs)} "
            f"does not match {len(embed_jobs)} jobs"
        )
    now = int(time.time())
    vec_i = 0
    for job in jobs:
        conn.execute(
            """INSERT INTO ichbin_profiles
                 (sender_id, text, msg_count, chat_id, msg_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(sender_id) DO UPDATE SET
                 text=excluded.text,
                 msg_count=excluded.msg_count,
                 chat_id=excluded.chat_id,
                 msg_id=excluded.msg_id,
                 updated_at=excluded.updated_at""",
            (job.sender_id, job.text, job.msg_count, job.chat_id, job.msg_id, now),
        )
        if job.embed:
            conn.execute(
                "INSERT OR REPLACE INTO ichbin_vecs (sender_id, vec) VALUES (?, ?)",
                (job.sender_id, embed.pack(np.asarray(vecs[vec_i], dtype=np.float32))),
            )
            vec_i += 1
    if delete_ids:
        conn.executemany(
            "DELETE FROM ichbin_profiles WHERE sender_id=?",
            [(sid,) for sid in delete_ids],
        )
    conn.commit()
    return {"embedded": vec_i, "written": len(jobs), "deleted": len(delete_ids)}


def sync(
    conn: sqlite3.Connection,
    chat_id: int | None = None,
    sender_id: int | None = None,
    all_chats: bool = False,
) -> dict[str, int]:
    """Plan, embed, and apply. Embedding runs before the write, like index.update."""
    jobs, delete_ids = plan_sync(conn, chat_id=chat_id, sender_id=sender_id, all_chats=all_chats)
    embed_jobs = [job for job in jobs if job.embed]
    vecs = embed.encode_passages([job.text for job in embed_jobs]) if embed_jobs else None
    stats = apply_sync(conn, jobs, vecs, delete_ids)
    log.info(
        "ichbin sync written=%s embedded=%s removed=%s",
        stats["written"],
        stats["embedded"],
        stats["deleted"],
    )
    return stats


def has_profile(conn: sqlite3.Connection, sender_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM ichbin_profiles WHERE sender_id=?", (int(sender_id),)
    ).fetchone()
    return row is not None


def _username(conn: sqlite3.Connection, sender_id: int) -> str | None:
    row = conn.execute(
        "SELECT username FROM people WHERE sender_id=?", (int(sender_id),)
    ).fetchone()
    if row is None or not row["username"]:
        return None
    handle = str(row["username"]).lstrip("@").strip()
    return handle or None


def display_name(conn: sqlite3.Connection, sender_id: int) -> str:
    """Public name, else the stored sender label, else User N. Never a raw telegram id."""
    person = conn.execute(
        "SELECT display_name, username FROM people WHERE sender_id=?",
        (int(sender_id),),
    ).fetchone()
    if person is not None and person["display_name"]:
        return str(person["display_name"])
    sender = conn.execute(
        """SELECT sender FROM messages
           WHERE sender_id=? AND sender IS NOT NULL AND trim(sender) != ''
           ORDER BY ts DESC LIMIT 1""",
        (int(sender_id),),
    ).fetchone()
    if sender is not None and sender["sender"]:
        return str(sender["sender"])
    alias = conn.execute(
        "SELECT ordinal FROM aliases WHERE sender_id=?", (int(sender_id),)
    ).fetchone()
    if alias is not None:
        return f"User {int(alias['ordinal'])}"
    if person is not None and person["username"]:
        return "@" + str(person["username"]).lstrip("@")
    return "User"


def _card(conn: sqlite3.Connection, sender_id: int, row) -> PersonCard:
    return PersonCard(
        sender_id=int(sender_id),
        name=display_name(conn, sender_id),
        username=_username(conn, sender_id),
        text=str(row["text"]),
        chat_id=int(row["chat_id"]),
        msg_id=int(row["msg_id"]),
        msg_count=int(row["msg_count"]),
    )


def sender_id_for_username(conn: sqlite3.Connection, username: str) -> int | None:
    """Resolve a handle. If several rows share it, prefer one with a profile."""
    handle = username.lstrip("@").lower()
    if not handle:
        return None
    row = conn.execute(
        """SELECT p.sender_id FROM people p
           LEFT JOIN ichbin_profiles i ON i.sender_id = p.sender_id
           WHERE p.username IS NOT NULL AND lower(ltrim(p.username, '@')) = ?
           ORDER BY (i.sender_id IS NOT NULL) DESC, p.updated_at DESC
           LIMIT 1""",
        (handle,),
    ).fetchone()
    return int(row["sender_id"]) if row else None


def resolve_username(
    conn: sqlite3.Connection, raw: str
) -> tuple[str, int | None, str | None]:
    """`('ok', sender_id, handle)`, `('unknown', None, handle)`, or `('bad', None, None)`."""
    handle = parse_username(raw)
    if handle is None:
        return "bad", None, None
    sender_id = sender_id_for_username(conn, handle)
    if sender_id is None:
        return "unknown", None, handle
    return "ok", sender_id, handle


def _rank(
    conn: sqlite3.Connection,
    sender_id: int,
    target_vec: np.ndarray,
    k: int,
    cosine_min: float,
) -> list[Neighbor]:
    rows = conn.execute(
        """SELECT p.sender_id, p.text, p.msg_count, p.chat_id, p.msg_id, v.vec
           FROM ichbin_profiles p
           JOIN ichbin_vecs v ON v.sender_id = p.sender_id
           WHERE p.sender_id != ?""",
        (int(sender_id),),
    ).fetchall()
    scored: list[tuple[float, int, sqlite3.Row]] = []
    for row in rows:
        vec = embed.unpack(row["vec"])
        if vec.shape != target_vec.shape:
            continue
        score = float(np.dot(target_vec, vec))
        if score < cosine_min:
            continue
        scored.append((score, int(row["sender_id"]), row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    limit = max(0, k)
    return [
        Neighbor(_card(conn, sid, row), score) for score, sid, row in scored[:limit]
    ]


def recommend(
    conn: sqlite3.Connection,
    sender_id: int,
    *,
    k: int | None = None,
    cosine_min: float | None = None,
) -> Recommend | None:
    """The person's profile and the nearest other profiles, or None if they have none."""
    profile = conn.execute(
        "SELECT text, msg_count, chat_id, msg_id FROM ichbin_profiles WHERE sender_id=?",
        (int(sender_id),),
    ).fetchone()
    if profile is None:
        return None
    target = _card(conn, sender_id, profile)
    if k is None:
        k = config.ICHBIN_TOP_K
    if cosine_min is None:
        cosine_min = config.ICHBIN_COSINE_MIN
    vec_row = conn.execute(
        "SELECT vec FROM ichbin_vecs WHERE sender_id=?", (int(sender_id),)
    ).fetchone()
    if vec_row is None:
        return Recommend(target, ())
    neighbors = _rank(conn, sender_id, embed.unpack(vec_row["vec"]), k, cosine_min)
    return Recommend(target, tuple(neighbors))


def excerpt(text: str, limit: int = _EXCERPT_CHARS) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1].rstrip() + "…"


def _person_header(card: PersonCard) -> str:
    name = html.escape(card.name)
    link = message_link(card.chat_id, card.msg_id)
    header = f'<a href="{link}">{name}</a>'
    if card.username:
        header += " @" + html.escape(card.username.lstrip("@"))
    return header


def format_similar(lang: str, rec: Recommend) -> str:
    """HTML reply: their full #ichbin text first, then similar people."""
    lines = [
        _person_header(rec.target),
        "",
        html.escape(rec.target.text),
        "",
        i18n.t(lang, "similar_heading"),
    ]
    if not rec.neighbors:
        lines.append(i18n.t(lang, "similar_none"))
    else:
        for neighbor in rec.neighbors:
            card = neighbor.card
            line = "• " + _person_header(card) + " — " + html.escape(excerpt(card.text))
            lines.append(line)
    return "\n".join(lines)


def main() -> None:
    import argparse

    from . import db, logconfig

    logconfig.setup()
    ap = argparse.ArgumentParser(description="Index #ichbin intros for similar-people search")
    ap.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="only this chat (default: every chat already in the database)",
    )
    args = ap.parse_args()
    conn = db.connect()
    if args.chat_id is None:
        stats = sync(conn, all_chats=True)
    else:
        stats = sync(conn, chat_id=args.chat_id)
    n = conn.execute("SELECT count(*) FROM ichbin_profiles").fetchone()[0]
    print(
        f"ichbin profiles: {n} "
        f"(embedded {stats['embedded']}, updated {stats['written']}, removed {stats['deleted']})"
    )


if __name__ == "__main__":
    main()
