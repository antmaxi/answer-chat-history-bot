"""Hybrid retrieval: BM25 over messages, cosine over window embeddings, fused with RRF.

Keyword search catches names, dates and exact terms; vector search catches
paraphrase. Reciprocal Rank Fusion merges the two rankings without needing the
scores to be on a comparable scale.
"""

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from . import config, embed, people
from .timerange import TimeRange, parse_time_range

# None = every chat; a tuple of ids is an allow-list (empty means no chats).
# values: window ids, vectors, (ts_start, ts_end) per window
_vec_cache: dict[tuple[int, ...] | None, tuple[list[int], np.ndarray, np.ndarray]] = {}
_df_cache: dict[str, int] = {}
_KEEP_STEM_MIN = 4
_KEEP_INFLECT_MIN = 5

# A single chat, an allow-list, or None for “no restriction” (CLI default).
# An empty sequence is *not* None: it means the caller may see zero chats.
ChatId = int | Sequence[int] | None


def normalize_chat_ids(chat_id: ChatId) -> list[int] | None:
    """None → all chats; a list (possibly empty) → only those chats.

    Empty is an allow-list that matches nothing, never a synonym for “all”.
    """
    if chat_id is None:
        return None
    if isinstance(chat_id, (str, bytes)):
        raise TypeError(f"chat_id must be int or a sequence of ints, not {type(chat_id).__name__}")
    if isinstance(chat_id, Sequence):
        return [int(c) for c in chat_id]
    return [int(chat_id)]


@dataclass
class Hit:
    window_id: int
    chat_id: int
    first_msg: int
    last_msg: int
    ts_start: int
    ts_end: int
    speakers: str
    text: str
    score: float
    cosine: float = 0.0

    def when(self) -> str:
        return datetime.fromtimestamp(self.ts_start, timezone.utc).strftime("%Y-%m-%d")

    def link(self) -> str:
        """Deep link to the first message of the window."""
        raw = str(self.chat_id)
        raw = raw[4:] if raw.startswith("-100") else raw.lstrip("-")
        return f"https://t.me/c/{raw}/{self.first_msg}"


def _doc_frequency(conn: sqlite3.Connection, token: str) -> int:
    """How many messages contain this token."""
    if token in _df_cache:
        return _df_cache[token]
    n = conn.execute(
        "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?", (f'"{token}"',)
    ).fetchone()[0]
    _df_cache[token] = n
    return n


def matches_keep_stem(token: str, stem: str) -> bool:
    """True if `token` is `stem` or an inflected/prefixed form of it."""
    token, stem = token.lower(), stem.lower()
    if not stem:
        return False
    if token == stem:
        return True
    if len(stem) >= _KEEP_STEM_MIN and (
        token.startswith(stem) or stem.startswith(token)
    ):
        return True
    n = min(len(token), len(stem))
    return n >= _KEEP_INFLECT_MIN and token[: n - 1] == stem[: n - 1]


def is_kept_term(token: str, stems: Sequence[str] | None = None) -> bool:
    """Protected from DF stopwording (place names, etc.)."""
    if stems is None:
        stems = config.STOPWORD_KEEP
    return any(matches_keep_stem(token, stem) for stem in stems)


# How many DF-band terms /stats a b will list (Telegram 4096; rest are counted).
TERM_DF_LIST_LIMIT = 200


def _ensure_fts_vocab(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_vocab "
        "USING fts5vocab(messages_fts, row)"
    )


def term_df_band(
    conn: sqlite3.Connection,
    lo_pct: float,
    hi_pct: float,
    *,
    limit: int = TERM_DF_LIST_LIMIT,
) -> tuple[int, list[tuple[str, int, float]], int]:
    """Terms whose message DF is in [lo_pct, hi_pct] percent of messages.

    Same definition as fts_query stopwording: messages containing the token
    over total messages. Skips 1-character tokens. Returns
    (message_count, [(term, df, pct), ...] most common first, match_count).
    The list is capped at `limit`; match_count is how many matched before the cap.
    """
    if lo_pct > hi_pct:
        lo_pct, hi_pct = hi_pct, lo_pct
    n = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    if not n:
        return 0, [], 0
    _ensure_fts_vocab(conn)
    # SQLite length() is bytes; over-fetch and drop 1-character tokens in Python
    # so the list matches fts_query (Cyrillic "и" is one character, two bytes).
    fetch = (max(limit, 0) + 64) if limit else 10_000_000
    rows = conn.execute(
        """
        SELECT term, doc FROM messages_fts_vocab
        WHERE length(term) > 1
          AND doc * 100.0 / ? >= ? AND doc * 100.0 / ? <= ?
        ORDER BY doc DESC, term COLLATE NOCASE
        LIMIT ?
        """,
        (n, lo_pct, n, hi_pct, fetch),
    ).fetchall()
    terms: list[tuple[str, int, float]] = []
    for term, doc in rows:
        if len(term) <= 1:
            continue
        terms.append((term, int(doc), 100.0 * int(doc) / n))
        if limit and len(terms) >= limit:
            break
    match_count = conn.execute(
        """
        SELECT count(*) FROM messages_fts_vocab
        WHERE length(term) > 1
          AND doc * 100.0 / ? >= ? AND doc * 100.0 / ? <= ?
        """,
        (n, lo_pct, n, hi_pct),
    ).fetchone()[0]
    match_count = max(int(match_count), len(terms))
    return n, terms, match_count


def fts_query(conn: sqlite3.Connection, question: str) -> str:
    """Turn free text into a safe, informative FTS5 OR-query.

    Two things to get right. First, user questions contain quotes, parentheses
    and the words AND/OR/NOT, all of which are FTS5 syntax — quoting each token
    individually sidesteps the lot.

    Second, an unfiltered OR-query matches every window in the corpus on
    stopwords alone ("how", "the", "for"), which turns the keyword ranking into
    noise that then dilutes the vector ranking during fusion. So drop terms that
    appear in a large fraction of messages. Deriving that from the corpus rather
    than a hardcoded English stopword list keeps it working on any language.
    """
    tokens = re.findall(r"\w+", question, flags=re.UNICODE)
    tokens = [t.lower() for t in tokens if len(t) > 1]
    if not tokens:
        return ""

    total = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    if total:
        cutoff = total * config.STOPWORD_DF_RATIO
        informative = [
            t
            for t in tokens
            if is_kept_term(t) or _doc_frequency(conn, t) <= cutoff
        ]
        # If every term is common, the question is all stopwords — keep them
        # rather than returning nothing and losing the keyword arm entirely.
        tokens = informative or tokens

    return " OR ".join(f'"{t}"' for t in tokens)


def keyword_search(
    conn: sqlite3.Connection,
    question: str,
    chat_id: ChatId,
    limit: int,
    time_range: TimeRange | None = None,
) -> list[int]:
    """Best-matching messages, mapped up to the windows that contain them."""
    query = fts_query(conn, question)
    if not query:
        return []
    chats = normalize_chat_ids(chat_id)
    if chats is not None and not chats:
        return []

    # bm25() is only available where the FTS table is queried directly, so the
    # ranking happens in a CTE and the join to windows happens outside it.
    # Filter by chat inside the CTE so a busy foreign chat cannot crowd out the
    # allow-list, and so we never even rank messages the caller must not see.
    params: list = [query]
    if chats is None:
        ranked = """
            SELECT rowid AS mid, bm25(messages_fts) AS rank
            FROM messages_fts
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        """
    else:
        placeholders = ",".join("?" * len(chats))
        ranked = f"""
            SELECT messages_fts.rowid AS mid, bm25(messages_fts) AS rank
            FROM messages_fts
            JOIN messages scoped ON scoped.id = messages_fts.rowid
            WHERE messages_fts MATCH ?
              AND scoped.chat_id IN ({placeholders})
            ORDER BY rank
            LIMIT ?
        """
        params.extend(chats)
    # Many messages collapse into few windows, so over-fetch inside the CTE.
    params.append(limit * 20)

    sql = f"""
        WITH ranked AS ({ranked})
        SELECT w.id, min(ranked.rank) AS rank
        FROM ranked
        JOIN messages m ON m.id = ranked.mid
        JOIN windows  w ON w.chat_id = m.chat_id
                       AND m.msg_id BETWEEN w.first_msg AND w.last_msg
    """
    if time_range is not None:
        sql += " WHERE w.ts_end >= ? AND w.ts_start <= ?"
        params.extend([time_range.start, time_range.end])
    sql += " GROUP BY w.id ORDER BY rank LIMIT ?"
    params.append(limit)

    return [r[0] for r in conn.execute(sql, params)]


def _vectors(conn: sqlite3.Connection, chat_id: ChatId) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Load and cache the vector matrix. Small enough to keep in memory."""
    chats = normalize_chat_ids(chat_id)
    empty_times = np.zeros((0, 2), dtype=np.int64)
    if chats is not None and not chats:
        return [], np.zeros((0, config.EMBED_DIM), dtype=np.float32), empty_times

    key = None if chats is None else tuple(sorted(chats))
    if key in _vec_cache:
        return _vec_cache[key]

    sql = """SELECT v.window_id, v.vec, w.ts_start, w.ts_end
             FROM window_vecs v JOIN windows w ON w.id = v.window_id"""
    params: list = []
    if chats is not None:
        placeholders = ",".join("?" * len(chats))
        sql += f" WHERE w.chat_id IN ({placeholders})"
        params.extend(chats)
    sql += " ORDER BY v.window_id"

    ids, blobs, starts, ends = [], [], [], []
    for row in conn.execute(sql, params):
        ids.append(row[0])
        blobs.append(embed.unpack(row[1]))
        starts.append(row[2])
        ends.append(row[3])

    matrix = np.vstack(blobs) if blobs else np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    times = np.column_stack([starts, ends]) if ids else empty_times
    _vec_cache[key] = (ids, matrix, times)
    return ids, matrix, times


def invalidate_cache() -> None:
    """Call after reindexing inside a long-running process."""
    _vec_cache.clear()
    _df_cache.clear()


def _vector_scores(
    conn: sqlite3.Connection,
    question: str,
    chat_id: ChatId,
    time_range: TimeRange | None = None,
    query_vec: np.ndarray | None = None,
) -> tuple[list[int], np.ndarray]:
    """Cosine of `question` against every (optionally time-filtered) window vector."""
    ids, matrix, times = _vectors(conn, chat_id)
    if not ids:
        return [], np.zeros(0, dtype=np.float32)
    if time_range is not None:
        mask = (times[:, 1] >= time_range.start) & (times[:, 0] <= time_range.end)
        if not np.any(mask):
            return [], np.zeros(0, dtype=np.float32)
        ids = [i for i, keep in zip(ids, mask) if keep]
        matrix = matrix[mask]
    # Vectors are normalized at write time, so a dot product is the cosine.
    if query_vec is None:
        query_vec = embed.encode_query(question)
    scores = matrix @ query_vec
    return ids, scores


def vector_search(
    conn: sqlite3.Connection,
    question: str,
    chat_id: ChatId,
    limit: int,
    time_range: TimeRange | None = None,
) -> list[int]:
    ids, scores = _vector_scores(conn, question, chat_id, time_range)
    if not ids:
        return []
    top = np.argsort(-scores)[:limit]
    return [ids[i] for i in top]


def recency_weight(ts_end: int, now_ts: int, half_life_days: float) -> float:
    """Exponential decay: 1.0 today, 0.5 at one half-life, approaching 0.

    half_life_days <= 0 disables (always 1.0). Future timestamps are treated as now.
    """
    if half_life_days <= 0:
        return 1.0
    age_days = max(0.0, (now_ts - ts_end) / 86400.0)
    return 0.5 ** (age_days / half_life_days)


def cap_hits(
    hits: list[Hit],
    min_k: int | None = None,
    max_k: int | None = None,
    cosine_min: float | None = None,
) -> list[Hit]:
    """Shorten a ranked hit list to MIN_K..MAX_K using cosine as a stop signal.

    Always keep the first min_k (so a thin question still has context). After
    that, stop at the first window whose cosine is below cosine_min. Never
    return more than max_k. cosine_min <= 0 disables the cutoff.
    """
    if not hits:
        return []
    min_k = config.MIN_K if min_k is None else min_k
    max_k = config.MAX_K if max_k is None else max_k
    cosine_min = config.COSINE_MIN if cosine_min is None else cosine_min
    max_k = max(1, max_k)
    min_k = min(max(0, min_k), max_k)
    kept: list[Hit] = []
    for hit in hits:
        if len(kept) >= max_k:
            break
        if len(kept) >= min_k and cosine_min > 0 and hit.cosine < cosine_min:
            break
        kept.append(hit)
    return kept


def search(
    conn: sqlite3.Connection,
    question: str,
    chat_id: ChatId = None,
    top_k: int | None = None,
    now: datetime | None = None,
    time_range: TimeRange | None = None,
    speaker: str | None = None,
    query_vec: np.ndarray | None = None,
) -> list[Hit]:
    chats = normalize_chat_ids(chat_id)
    if chats is not None and not chats:
        return []
    allowed = None if chats is None else set(chats)

    now = now or datetime.now(timezone.utc)
    if time_range is None:
        time_range = parse_time_range(question, now)
    if speaker is None:
        speaker = people.parse_speaker(question, people.known_speakers(conn))

    explicit_k = top_k is not None
    top_k = top_k or config.TOP_K
    pool = top_k * 4
    if time_range or speaker:
        pool = top_k * 8

    vec_ids, vec_scores = _vector_scores(
        conn, question, chat_id, time_range, query_vec=query_vec
    )
    cosine = {wid: float(s) for wid, s in zip(vec_ids, vec_scores)}
    if vec_ids:
        vec_ranking = [vec_ids[i] for i in np.argsort(-vec_scores)[:pool]]
    else:
        vec_ranking = []

    # Vector search carries more weight: most questions are paraphrases of what
    # was actually said. Keyword search earns its place on names, numbers and
    # exact strings like passwords, where embeddings are weak.
    rankings = [
        (keyword_search(conn, question, chat_id, pool, time_range), config.WEIGHT_KEYWORD),
        (vec_ranking, config.WEIGHT_VECTOR),
    ]

    fused: dict[int, float] = {}
    for ranking, weight in rankings:
        for rank, window_id in enumerate(ranking):
            fused[window_id] = fused.get(window_id, 0.0) + weight / (config.RRF_K + rank + 1)

    if fused and config.RECENCY_HALF_LIFE_DAYS > 0 and time_range is None:
        now_ts = int(now.timestamp())
        placeholders = ",".join("?" * len(fused))
        ts_end = {
            r[0]: r[1]
            for r in conn.execute(
                f"SELECT id, ts_end FROM windows WHERE id IN ({placeholders})",
                list(fused),
            )
        }
        for wid, score in fused.items():
            fused[wid] = score * recency_weight(
                ts_end.get(wid, now_ts), now_ts, config.RECENCY_HALF_LIFE_DAYS
            )

    best = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    if not best:
        return []

    placeholders = ",".join("?" * len(best))
    rows = {
        r["id"]: r
        for r in conn.execute(
            f"SELECT * FROM windows WHERE id IN ({placeholders})", [wid for wid, _ in best]
        )
    }

    hits = []
    for window_id, score in best:
        r = rows.get(window_id)
        if r is None:
            continue
        if allowed is not None and r["chat_id"] not in allowed:
            continue
        if time_range is not None and not time_range.overlaps(r["ts_start"], r["ts_end"]):
            continue
        if speaker and speaker.lower() not in (r["speakers"] or "").lower():
            continue
        hits.append(
            Hit(
                window_id=r["id"],
                chat_id=r["chat_id"],
                first_msg=r["first_msg"],
                last_msg=r["last_msg"],
                ts_start=r["ts_start"],
                ts_end=r["ts_end"],
                speakers=r["speakers"] or "",
                text=r["text"],
                score=score,
                cosine=cosine.get(window_id, 0.0),
            )
        )
    if not explicit_k:
        hits = cap_hits(hits)
    return hits
