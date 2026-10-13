"""Stem chat text the same way for the FTS index and for keyword queries.

SQLite's unicode61 tokenizer does not stem. Russian «виза» and «визы» would
miss each other, and each inflection of a particle would look rare enough to
survive the document-frequency stopword cutoff. Snowball stems are written
into the FTS index (via the stem_text SQL function) and applied to the query,
so both sides share one term.

Digits and mixed-script tokens are left as-is so passwords and identifiers
stay exact. STOPWORD_KEEP place names collapse to the configured stem, which
is what the inflected forms already match in is_kept_term.
"""

from __future__ import annotations

import re

import snowballstemmer

from . import config

_TOKEN = re.compile(r"\w+", flags=re.UNICODE)
_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")
_LATIN = re.compile(r"[A-Za-z]")
_KEEP_STEM_MIN = 4
_KEEP_INFLECT_MIN = 5

_ru = snowballstemmer.stemmer("russian")
_en = snowballstemmer.stemmer("english")


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


def is_kept_term(token: str, stems: tuple[str, ...] | list[str] | None = None) -> bool:
    """Protected from DF stopwording (place names, etc.)."""
    if stems is None:
        stems = config.STOPWORD_KEEP
    return any(matches_keep_stem(token, stem) for stem in stems)


def _keep_stem(token: str) -> str | None:
    for stem in config.STOPWORD_KEEP:
        if matches_keep_stem(token, stem):
            return stem.lower()
    return None


def stem_token(token: str) -> str:
    """One query or index token, lowercased, stemmed when that is safe."""
    token = token.lower()
    if any(ch.isdigit() for ch in token):
        return token
    kept = _keep_stem(token)
    if kept:
        return kept
    has_cyr = _CYRILLIC.search(token) is not None
    has_lat = _LATIN.search(token) is not None
    if has_cyr and has_lat:
        return token
    if has_cyr:
        return _ru.stemWord(token)
    if has_lat:
        return _en.stemWord(token)
    return token


def stem_text(text: str | None) -> str:
    """Space-joined stems of the words in `text`. Registered as a SQL function."""
    if not text:
        return ""
    stems = [stem_token(t) for t in _TOKEN.findall(text) if len(t) > 1]
    return " ".join(s for s in stems if s)
