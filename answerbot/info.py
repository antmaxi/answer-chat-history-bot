"""About-the-bot text for /info: last git commit (or file mtime), LLM, and source URL."""

from __future__ import annotations

import html
import logging
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from . import config, i18n

log = logging.getLogger("answerbot")


def fmt_dt_utc(dt: datetime) -> str:
    """Format a datetime in the configured display timezone (default UTC+2).

    Naive datetimes are treated as UTC instants. Aware datetimes are converted
    to the display zone. The label uses UTC±HH:MM for the display offset.
    """
    tz = config.display_timezone()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    local = dt.astimezone(tz)
    off = local.strftime("%z")
    return local.strftime("%Y-%m-%d %H:%M:%S") + f" UTC{off[:3]}:{off[3:5]}"


def _git_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    if Path(".git").exists():
        return Path(".")
    return None


def last_update() -> str:
    """Last git commit time, or this file's mtime if git is unavailable."""
    root = _git_root()
    if root is not None:
        try:
            ct = (
                subprocess.check_output(
                    ["git", "log", "-1", "--format=%ct"],
                    cwd=root,
                    stderr=subprocess.DEVNULL,
                )
                .decode("utf-8")
                .strip()
            )
            return fmt_dt_utc(datetime.fromtimestamp(int(ct), tz=UTC))
        except Exception:
            log.warning("Could not get last commit via git", exc_info=True)
    try:
        mtime = os.path.getmtime(__file__)
        return fmt_dt_utc(datetime.fromtimestamp(mtime, tz=UTC))
    except Exception:
        log.warning("Could not get file mtime", exc_info=True)
        return "unknown"


def fmt_duration_ms(ms: float) -> str:
    """Format a millisecond duration as seconds with one decimal place."""
    return f"{ms / 1000:.1f}s"


def _latency_phrase(summary: dict | None, lang: str) -> str:
    if not summary:
        return i18n.t(lang, "stats_latency_none")
    return i18n.t(
        lang,
        "stats_latency_range",
        median=fmt_duration_ms(summary["median_ms"]),
        std=fmt_duration_ms(summary["std_ms"]),
        min=fmt_duration_ms(summary["min_ms"]),
        max=fmt_duration_ms(summary["max_ms"]),
    )


def format_latency(s: dict, lang: str | None = None) -> str:
    """Ask-time lines (median ± std, min/max) for day / week / month."""
    lang = i18n.normalize_lang(lang)
    return i18n.t(
        lang,
        "stats_latency",
        day=_latency_phrase(s.get("latency_day"), lang),
        week=_latency_phrase(s.get("latency_week"), lang),
        month=_latency_phrase(s.get("latency_month"), lang),
    )


_PCT = re.compile(r"^(\d+(?:\.\d+)?)\s*%?$")
_PCT_RANGE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*%?\s*[-–—]\s*(\d+(?:\.\d+)?)\s*%?$"
)


def _parse_pct(token: str) -> float | None:
    m = _PCT.match(token.strip())
    if not m:
        return None
    return float(m.group(1))


def parse_stats_df_args(args: str | None) -> tuple[float, float] | None:
    """Parse `/stats a b` percents. None means plain stats; ValueError if invalid.

    Accepts `10 25`, `10-25`, optional `%` suffixes. Bounds are 0–100 inclusive;
    a > b is swapped. Same DF units as STOPWORD_DF_RATIO (percent of messages).
    """
    raw = (args or "").strip()
    if not raw:
        return None
    m = _PCT_RANGE.match(raw)
    if m:
        lo, hi = float(m.group(1)), float(m.group(2))
    else:
        parts = raw.replace(",", " ").split()
        if len(parts) != 2:
            raise ValueError("need two percentages")
        lo, hi = _parse_pct(parts[0]), _parse_pct(parts[1])
        if lo is None or hi is None:
            raise ValueError("need two percentages")
    if lo > hi:
        lo, hi = hi, lo
    if lo < 0 or hi > 100:
        raise ValueError("percentages must be 0–100")
    return lo, hi


def fmt_pct(value: float) -> str:
    """Drop trailing zeros: 10, 12.5, 0.25."""
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def format_term_df(
    messages: int,
    lo: float,
    hi: float,
    terms: list[tuple[str, int, float]],
    match_count: int,
    lang: str | None = None,
) -> str:
    lang = i18n.normalize_lang(lang)
    lo_s, hi_s = fmt_pct(lo), fmt_pct(hi)
    if not terms:
        return i18n.t(
            lang, "stats_df_empty", lo=lo_s, hi=hi_s, messages=messages
        )
    shown = len(terms)
    if match_count > shown:
        count = i18n.t(
            lang, "stats_df_shown", shown=shown, total=match_count
        )
    else:
        count = str(shown)
    lines = [
        i18n.t(
            lang,
            "stats_df",
            lo=lo_s,
            hi=hi_s,
            messages=messages,
            count=count,
        )
    ]
    for term, doc, pct in terms:
        lines.append(f"{fmt_pct(pct):>6}%  {doc:>5}  {term}")
    if match_count > shown:
        lines.append(i18n.t(lang, "stats_df_more", n=match_count - shown))
    return "\n".join(lines)


def telegram_chunks(text: str, limit: int = 3500) -> list[str]:
    """Split on newlines so each piece fits Telegram's 4096 cap."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= limit:
            chunks.append(rest)
            break
        cut = rest.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(rest[:cut])
        rest = rest[cut:].lstrip("\n")
    return chunks


def format_stats(s: dict, lang: str | None = None, *, questions: bool = False) -> str:
    lang = i18n.normalize_lang(lang)
    first, last = s.get("first_message"), s.get("last_message")
    span = ""
    if first and last:
        span = i18n.t(lang, "stats_span", first=first, last=last)
    text = (
        i18n.t(
            lang,
            "stats",
            messages=s.get("messages", 0),
            windows=s.get("windows", 0),
            embedded=s.get("embedded", 0),
            chats=s.get("chats", 0),
        )
        + span
    )
    if questions:
        text += i18n.t(
            lang,
            "stats_queries",
            day=s.get("questions_day", 0),
            day_admin=s.get("questions_day_admin", 0),
            day_other=s.get("questions_day_other", 0),
            week=s.get("questions_week", 0),
            week_admin=s.get("questions_week_admin", 0),
            week_other=s.get("questions_week_other", 0),
            month=s.get("questions_month", 0),
            month_admin=s.get("questions_month_admin", 0),
            month_other=s.get("questions_month_other", 0),
        )
        text += i18n.t(lang, "stats_last_user", when=_last_user_when(s, lang))
        text += format_latency(s, lang)
    return text


def _last_user_when(s: dict, lang: str) -> str:
    """Live non-admin use, else last completed non-admin ask, else never."""
    if s.get("user_in_use"):
        return i18n.t(lang, "stats_last_user_now")
    last = s.get("last_user_ask")
    if last:
        return last
    return i18n.t(lang, "stats_last_user_never")


_PROVIDER_LABELS = {
    "claude": "Claude",
    "gemini": "Gemini",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "cursor": "Cursor",
    "ollama": "Ollama",
}


def provider_label(provider: str | None = None) -> str:
    raw = (provider if provider is not None else config.LLM_PROVIDER).strip()
    return _PROVIDER_LABELS.get(raw.lower(), raw)


def format_info(updated: str, lang: str | None = None, stats: dict | None = None) -> str:
    lang = i18n.normalize_lang(lang)
    if updated == "unknown":
        updated = i18n.t(lang, "unknown")
    provider = config.LLM_PROVIDER.lower()
    retention = i18n.t(lang, "info_retention_cursor") if provider == "cursor" else ""
    text = i18n.t(
        lang,
        "info_msg",
        bot_name=i18n.t(lang, "bot_name"),
        last_commit=html.escape(updated),
        github_repo=html.escape(config.GITHUB_REPO),
        model=html.escape(config.ANSWER_MODEL),
        provider=html.escape(provider_label()),
        retention=retention,
    )
    if stats is not None:
        text += "\n\n" + format_stats(stats, lang)
    return text
