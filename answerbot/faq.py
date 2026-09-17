"""Community FAQ wiki (ru-ch/faq) as a second retrieval source.

The GitHub Pages site has no search API. Markdown under docs/ and inbox/ is
fetched, chunked by heading, and stored with FTS + the same local embeddings
as chat messages. Hits become retrieve.Hit rows with kind="faq" so citations
open the article URL instead of a Telegram message.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import sqlite3
import tarfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, embed, logconfig
from .retrieve import Hit

log = logging.getLogger("answerbot")

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
_KEEP_DIRS = ("docs", "inbox")
_USER_AGENT = "answer-chat-history-bot"
_HTTP_TIMEOUT = 60.0

_vec_ids: list[int] | None = None
_vec_matrix: np.ndarray | None = None
_df_cache: dict[str, int] = {}


@dataclass
class Chunk:
    heading: str
    text: str


def invalidate() -> None:
    global _vec_ids, _vec_matrix
    _vec_ids = None
    _vec_matrix = None
    _df_cache.clear()


def page_url(path: str, site: str | None = None) -> str:
    """docs/Быт.md → https://ru-ch.github.io/faq/docs/Быт.html"""
    base = (site or config.FAQ_SITE).rstrip("/")
    rel = path[:-3] + ".html" if path.endswith(".md") else path
    return f"{base}/{rel}"


def page_title(path: str) -> str:
    return Path(path).stem


def _content_sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_markdown(
    markdown: str,
    *,
    title: str,
    max_chars: int | None = None,
) -> list[Chunk]:
    """Split on # / ## / ###. Heading trail is prepended so a subsection still
    carries its parent titles (e.g. recycling → Zürich)."""
    max_chars = config.WINDOW_MAX_CHARS if max_chars is None else max_chars
    levels = {1: "", 2: "", 3: ""}
    buf: list[str] = []
    chunks: list[Chunk] = []

    def trail() -> str:
        return " · ".join(levels[i] for i in (1, 2, 3) if levels[i])

    def speakers(heading: str) -> str:
        if not heading:
            return title
        if heading == title or heading.startswith(f"{title} · "):
            return heading
        return f"{title} · {heading}"

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        heading = trail()
        if not body and not heading:
            return
        label = speakers(heading)
        text = f"{label}\n\n{body}".strip() if body else label
        if not text.strip():
            return
        for piece in _split_chars(text, max_chars):
            chunks.append(Chunk(heading=label, text=piece))

    for line in markdown.replace("\r\n", "\n").split("\n"):
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            levels[level] = m.group(2).strip()
            for lv in range(level + 1, 4):
                levels[lv] = ""
            continue
        buf.append(line)
    flush()
    return chunks


def _split_chars(text: str, max_chars: int) -> list[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []
    parts = re.split(r"\n\s*\n", text)
    out: list[str] = []
    cur = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if not cur:
            cur = piece
            continue
        if len(cur) + 2 + len(piece) <= max_chars:
            cur = f"{cur}\n\n{piece}"
            continue
        out.append(cur)
        cur = piece
    if cur:
        out.append(cur)
    # A single paragraph can still exceed the cap; hard-split as a last resort.
    hard: list[str] = []
    for block in out:
        if len(block) <= max_chars:
            hard.append(block)
            continue
        for i in range(0, len(block), max_chars):
            hard.append(block[i : i + max_chars])
    return hard


def _faq_rel(name: str) -> str | None:
    """faq-master/docs/Быт.md → docs/Быт.md. One directory deep only."""
    parts = Path(name).parts
    if len(parts) < 3:
        return None
    rel_parts = parts[1:]
    if rel_parts[0] not in _KEEP_DIRS:
        return None
    if len(rel_parts) != 2 or not rel_parts[1].endswith(".md"):
        return None
    return "/".join(rel_parts)


def _http_get(url: str, timeout: float = _HTTP_TIMEOUT) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/vnd.github+json, application/octet-stream, */*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def remote_sha(repo: str | None = None) -> str | None:
    repo = repo or config.FAQ_REPO
    if not repo:
        return None
    url = f"https://api.github.com/repos/{repo}/commits/master"
    try:
        payload = json.loads(_http_get(url, timeout=20).decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        log.warning("faq: could not read remote SHA for %s", repo, exc_info=True)
        return None
    sha = payload.get("sha")
    return str(sha) if sha else None


def _from_tarball(repo: str | None = None) -> dict[str, tuple[str, str]]:
    repo = repo or config.FAQ_REPO
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/master"
    blob = _http_get(url)
    pages: dict[str, tuple[str, str]] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            rel = _faq_rel(member.name)
            if rel is None:
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            text = extracted.read().decode("utf-8", errors="replace")
            pages[rel] = (_content_sha(text), text)
    return pages


def _from_raw_tree(repo: str | None = None) -> dict[str, tuple[str, str]]:
    repo = repo or config.FAQ_REPO
    url = f"https://api.github.com/repos/{repo}/git/trees/master?recursive=1"
    payload = json.loads(_http_get(url).decode("utf-8"))
    pages: dict[str, tuple[str, str]] = {}
    for entry in payload.get("tree") or []:
        if entry.get("type") != "blob":
            continue
        path = entry.get("path") or ""
        parts = path.split("/")
        if len(parts) != 2 or parts[0] not in _KEEP_DIRS or not parts[1].endswith(".md"):
            continue
        raw = (
            f"https://raw.githubusercontent.com/{repo}/master/{path}"
        )
        text = _http_get(raw).decode("utf-8", errors="replace")
        pages[path] = (_content_sha(text), text)
    return pages


def fetch_repo(repo: str | None = None) -> tuple[str, dict[str, tuple[str, str]]]:
    """Return (commit_or_content_sha, {path: (file_sha, markdown)})."""
    repo = repo or config.FAQ_REPO
    sha = remote_sha(repo)
    try:
        pages = _from_tarball(repo)
    except (urllib.error.URLError, TimeoutError, tarfile.TarError, OSError):
        log.warning("faq: tarball failed, falling back to raw files", exc_info=True)
        pages = _from_raw_tree(repo)
    if not sha:
        sha = hashlib.sha1(
            "".join(sorted(s for s, _ in pages.values())).encode()
        ).hexdigest()
    return sha, pages


def stored_sha(conn: sqlite3.Connection, repo: str | None = None) -> str | None:
    repo = repo or config.FAQ_REPO
    row = conn.execute(
        "SELECT sha FROM faq_state WHERE repo=?", (repo,)
    ).fetchone()
    return row[0] if row else None


def is_current(conn: sqlite3.Connection, sha: str | None, repo: str | None = None) -> bool:
    """True when `sha` matches the stored commit and chunks already exist."""
    if not sha:
        return False
    n = conn.execute("SELECT count(*) FROM faq_chunks").fetchone()[0]
    return bool(n) and stored_sha(conn, repo) == sha


def index_markdown(
    conn: sqlite3.Connection,
    files: dict[str, str],
    *,
    sha: str = "test",
    progress: bool = False,
) -> dict:
    """Index path → markdown without hitting GitHub. Used by tests and fetch."""
    pages = {path: (_content_sha(md), md) for path, md in files.items()}
    return apply_pages(conn, pages, sha=sha, progress=progress)


def apply_pages(
    conn: sqlite3.Connection,
    pages: dict[str, tuple[str, str]],
    *,
    sha: str,
    progress: bool = False,
) -> dict:
    """Replace changed paths, drop missing ones, embed new chunks."""
    existing = {
        r["path"]: r["sha"]
        for r in conn.execute("SELECT path, sha FROM faq_pages")
    }
    wanted = set(pages)
    for path in existing.keys() - wanted:
        conn.execute("DELETE FROM faq_chunks WHERE path=?", (path,))
        conn.execute("DELETE FROM faq_pages WHERE path=?", (path,))

    pending_ids: list[int] = []
    pending_texts: list[str] = []
    now = int(time.time())
    changed = 0
    for path, (file_sha, markdown) in pages.items():
        if existing.get(path) == file_sha:
            continue
        changed += 1
        conn.execute("DELETE FROM faq_chunks WHERE path=?", (path,))
        conn.execute("DELETE FROM faq_pages WHERE path=?", (path,))
        title = page_title(path)
        url = page_url(path)
        conn.execute(
            """INSERT INTO faq_pages (path, title, url, sha, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (path, title, url, file_sha, now),
        )
        for chunk in chunk_markdown(markdown, title=title):
            cur = conn.execute(
                "INSERT INTO faq_chunks (path, heading, text) VALUES (?, ?, ?)",
                (path, chunk.heading, chunk.text),
            )
            pending_ids.append(int(cur.lastrowid))
            pending_texts.append(chunk.text)

    if pending_texts:
        vecs = embed.encode_passages(pending_texts, progress=progress)
        conn.executemany(
            "INSERT OR REPLACE INTO faq_vecs (chunk_id, vec) VALUES (?, ?)",
            [(cid, embed.pack(v)) for cid, v in zip(pending_ids, vecs)],
        )

    repo = config.FAQ_REPO or "ru-ch/faq"
    conn.execute(
        """INSERT INTO faq_state (repo, sha, updated_at) VALUES (?, ?, ?)
           ON CONFLICT (repo) DO UPDATE SET sha=excluded.sha, updated_at=excluded.updated_at""",
        (repo, sha, now),
    )
    conn.commit()
    invalidate()
    n_chunks = conn.execute("SELECT count(*) FROM faq_chunks").fetchone()[0]
    n_pages = conn.execute("SELECT count(*) FROM faq_pages").fetchone()[0]
    log.info(
        "faq index: %s pages, %s chunks (%s path(s) changed)",
        n_pages, n_chunks, changed,
    )
    return {
        "pages": n_pages,
        "chunks": n_chunks,
        "changed": changed,
        "embedded": len(pending_ids),
    }


def update(
    conn: sqlite3.Connection,
    *,
    progress: bool = False,
    fetched: tuple[str, dict[str, tuple[str, str]]] | None = None,
) -> dict:
    """Fetch the remote wiki if it moved; no-op when disabled or unchanged.

    Network failure leaves the last good index in place. Pass `fetched` to skip
    the HTTP round-trip (the bot fetches off the SQLite lock).
    """
    if not config.FAQ_ENABLED or not config.FAQ_REPO:
        return {"pages": 0, "chunks": 0, "skipped": True}
    try:
        if fetched is None:
            sha = remote_sha()
            if is_current(conn, sha):
                n_chunks = conn.execute("SELECT count(*) FROM faq_chunks").fetchone()[0]
                n_pages = conn.execute("SELECT count(*) FROM faq_pages").fetchone()[0]
                return {"pages": n_pages, "chunks": n_chunks, "unchanged": True}
            sha, pages = fetch_repo()
        else:
            sha, pages = fetched
    except Exception:
        log.exception("faq fetch failed")
        n_chunks = conn.execute("SELECT count(*) FROM faq_chunks").fetchone()[0]
        n_pages = conn.execute("SELECT count(*) FROM faq_pages").fetchone()[0]
        return {"pages": n_pages, "chunks": n_chunks, "error": True}
    if not pages:
        log.warning("faq fetch returned no markdown files")
        return {"pages": 0, "chunks": 0, "error": True}
    previous = stored_sha(conn)
    have = conn.execute("SELECT count(*) FROM faq_chunks").fetchone()[0]
    if previous == sha and have:
        n_pages = conn.execute("SELECT count(*) FROM faq_pages").fetchone()[0]
        return {"pages": n_pages, "chunks": have, "unchanged": True}
    return apply_pages(conn, pages, sha=sha, progress=progress)


def _doc_frequency(conn: sqlite3.Connection, token: str) -> int:
    if token in _df_cache:
        return _df_cache[token]
    n = conn.execute(
        "SELECT count(*) FROM faq_fts WHERE faq_fts MATCH ?", (f'"{token}"',)
    ).fetchone()[0]
    _df_cache[token] = n
    return n


def _fts_query(conn: sqlite3.Connection, question: str) -> str:
    from . import retrieve

    tokens = re.findall(r"\w+", question, flags=re.UNICODE)
    tokens = [t.lower() for t in tokens if len(t) > 1]
    if not tokens:
        return ""
    total = conn.execute("SELECT count(*) FROM faq_chunks").fetchone()[0]
    if total:
        cutoff = total * config.STOPWORD_DF_RATIO
        informative = [
            t
            for t in tokens
            if retrieve.is_kept_term(t) or _doc_frequency(conn, t) <= cutoff
        ]
        tokens = informative or tokens
    return " OR ".join(f'"{t}"' for t in tokens)


def _keyword_ids(conn: sqlite3.Connection, question: str, limit: int) -> list[int]:
    query = _fts_query(conn, question)
    if not query:
        return []
    try:
        rows = conn.execute(
            """SELECT faq_fts.rowid AS cid
               FROM faq_fts
               WHERE faq_fts MATCH ?
               ORDER BY bm25(faq_fts)
               LIMIT ?""",
            (query, limit),
        )
    except sqlite3.OperationalError:
        return []
    return [int(r[0]) for r in rows]


def _vectors(conn: sqlite3.Connection) -> tuple[list[int], np.ndarray]:
    global _vec_ids, _vec_matrix
    if _vec_ids is not None and _vec_matrix is not None:
        return _vec_ids, _vec_matrix
    ids, blobs = [], []
    for row in conn.execute(
        "SELECT chunk_id, vec FROM faq_vecs ORDER BY chunk_id"
    ):
        ids.append(int(row[0]))
        blobs.append(embed.unpack(row[1]))
    matrix = (
        np.vstack(blobs)
        if blobs
        else np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    )
    _vec_ids, _vec_matrix = ids, matrix
    return ids, matrix


def _vector_ranking(
    conn: sqlite3.Connection,
    question: str,
    limit: int,
    query_vec: np.ndarray | None,
) -> tuple[list[int], dict[int, float]]:
    ids, matrix = _vectors(conn)
    if not ids:
        return [], {}
    if query_vec is None:
        query_vec = embed.encode_query(question)
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    if q.shape[0] != matrix.shape[1]:
        return [], {}
    scores = matrix @ q
    order = np.argsort(-scores)[:limit]
    ranking = [ids[i] for i in order]
    cosine = {ids[i]: float(scores[i]) for i in range(len(ids))}
    return ranking, cosine


def search(
    conn: sqlite3.Connection,
    question: str,
    *,
    query_vec: np.ndarray | None = None,
    top_k: int | None = None,
) -> list[Hit]:
    """Hybrid BM25 + cosine over FAQ chunks. No thread expansion, no recency."""
    if not config.FAQ_ENABLED:
        return []
    n = conn.execute("SELECT count(*) FROM faq_chunks").fetchone()[0]
    if not n:
        return []
    top_k = config.FAQ_TOP_K if top_k is None else top_k
    if top_k <= 0:
        return []
    pool = max(top_k * 4, top_k)

    kw = _keyword_ids(conn, question, pool)
    vec_rank, cosine = _vector_ranking(conn, question, pool, query_vec)

    fused: dict[int, float] = {}
    for ranking, weight in (
        (kw, config.WEIGHT_KEYWORD),
        (vec_rank, config.WEIGHT_VECTOR),
    ):
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0.0) + weight / (config.RRF_K + rank + 1)
    if not fused:
        return []

    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    ids = [cid for cid, _ in ranked]
    placeholders = ",".join("?" * len(ids))
    rows = {
        int(r["id"]): r
        for r in conn.execute(
            f"""SELECT c.id, c.path, c.heading, c.text, p.title, p.url
                FROM faq_chunks c JOIN faq_pages p ON p.path = c.path
                WHERE c.id IN ({placeholders})""",
            ids,
        )
    }
    hits: list[Hit] = []
    for cid, score in ranked:
        row = rows.get(cid)
        if row is None:
            continue
        hits.append(
            Hit(
                window_id=int(row["id"]),
                chat_id=0,
                first_msg=0,
                last_msg=0,
                ts_start=0,
                ts_end=0,
                speakers=row["heading"] or row["title"],
                text=row["text"],
                score=score,
                cosine=cosine.get(cid, 0.0),
                kind="faq",
                url=row["url"],
            )
        )

    has_vecs = bool(cosine)
    cosine_min = config.FAQ_COSINE_MIN
    if has_vecs and cosine_min > 0:
        hits = [h for h in hits if h.cosine >= cosine_min]
    return hits[:top_k]


def prepend(
    conn: sqlite3.Connection,
    question: str,
    chat_hits: list[Hit],
    query_vec: np.ndarray | None = None,
) -> list[Hit]:
    """FAQ excerpts first, then chat hits. Separate lists, not one RRF."""
    extra = search(conn, question, query_vec=query_vec)
    if not extra:
        return chat_hits
    return extra + chat_hits


def main() -> None:
    from . import db

    logconfig.setup()
    conn = db.connect()
    result = update(conn, progress=True)
    if result.get("skipped"):
        print("faq disabled (FAQ_ENABLED=off or FAQ_REPO empty)")
        return
    if result.get("error"):
        print("faq fetch failed — last index left in place")
        return
    if result.get("unchanged"):
        print(f"faq unchanged: {result['pages']} pages, {result['chunks']} chunks")
        return
    print(
        f"faq indexed {result['pages']} pages, {result['chunks']} chunks "
        f"({result['changed']} path(s) changed, {result['embedded']} embedded)"
    )


if __name__ == "__main__":
    main()
