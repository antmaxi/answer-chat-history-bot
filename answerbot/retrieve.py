"""Hybrid retrieval: BM25 over messages, cosine over window embeddings, fused with RRF.

Keyword search catches names, dates and exact terms; vector search catches
paraphrase. Reciprocal Rank Fusion merges the two rankings without needing the
scores to be on a comparable scale.
"""

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from . import config, embed

_vec_cache: dict[int, tuple[list[int], np.ndarray]] = {}
_df_cache: dict[str, int] = {}


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
        informative = [t for t in tokens if _doc_frequency(conn, t) <= cutoff]
        # If every term is common, the question is all stopwords — keep them
        # rather than returning nothing and losing the keyword arm entirely.
        tokens = informative or tokens

    return " OR ".join(f'"{t}"' for t in tokens)


def keyword_search(conn: sqlite3.Connection, question: str, chat_id: int | None, limit: int) -> list[int]:
    """Best-matching messages, mapped up to the windows that contain them."""
    query = fts_query(conn, question)
    if not query:
        return []

    # bm25() is only available where the FTS table is queried directly, so the
    # ranking happens in a CTE and the join to windows happens outside it.
    sql = """
        WITH ranked AS (
            SELECT rowid AS mid, bm25(messages_fts) AS rank
            FROM messages_fts
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        )
        SELECT w.id, min(ranked.rank) AS rank
        FROM ranked
        JOIN messages m ON m.id = ranked.mid
        JOIN windows  w ON w.chat_id = m.chat_id
                       AND m.msg_id BETWEEN w.first_msg AND w.last_msg
    """
    # Many messages collapse into few windows, so over-fetch inside the CTE.
    params: list = [query, limit * 20]
    if chat_id is not None:
        sql += " WHERE m.chat_id = ?"
        params.append(chat_id)
    sql += " GROUP BY w.id ORDER BY rank LIMIT ?"
    params.append(limit)

    return [r[0] for r in conn.execute(sql, params)]


def _vectors(conn: sqlite3.Connection, chat_id: int | None) -> tuple[list[int], np.ndarray]:
    """Load and cache the vector matrix. Small enough to keep in memory."""
    key = chat_id if chat_id is not None else -1
    if key in _vec_cache:
        return _vec_cache[key]

    sql = "SELECT v.window_id, v.vec FROM window_vecs v JOIN windows w ON w.id = v.window_id"
    params: list = []
    if chat_id is not None:
        sql += " WHERE w.chat_id = ?"
        params.append(chat_id)
    sql += " ORDER BY v.window_id"

    ids, blobs = [], []
    for row in conn.execute(sql, params):
        ids.append(row[0])
        blobs.append(embed.unpack(row[1]))

    matrix = np.vstack(blobs) if blobs else np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    _vec_cache[key] = (ids, matrix)
    return ids, matrix


def invalidate_cache() -> None:
    """Call after reindexing inside a long-running process."""
    _vec_cache.clear()
    _df_cache.clear()


def vector_search(conn: sqlite3.Connection, question: str, chat_id: int | None, limit: int) -> list[int]:
    ids, matrix = _vectors(conn, chat_id)
    if not ids:
        return []

    # Vectors are normalized at write time, so a dot product is the cosine.
    scores = matrix @ embed.encode_query(question)
    top = np.argsort(-scores)[:limit]
    return [ids[i] for i in top]


def search(
    conn: sqlite3.Connection,
    question: str,
    chat_id: int | None = None,
    top_k: int | None = None,
) -> list[Hit]:
    top_k = top_k or config.TOP_K
    pool = top_k * 4

    # Vector search carries more weight: most questions are paraphrases of what
    # was actually said. Keyword search earns its place on names, numbers and
    # exact strings like passwords, where embeddings are weak.
    rankings = [
        (keyword_search(conn, question, chat_id, pool), config.WEIGHT_KEYWORD),
        (vector_search(conn, question, chat_id, pool), config.WEIGHT_VECTOR),
    ]

    fused: dict[int, float] = {}
    for ranking, weight in rankings:
        for rank, window_id in enumerate(ranking):
            fused[window_id] = fused.get(window_id, 0.0) + weight / (config.RRF_K + rank + 1)

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
            )
        )
    return hits
