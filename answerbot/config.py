"""Configuration, read once from the environment / .env file."""

import os
from datetime import timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "data/answerbot.db"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_raw_log_path = os.getenv("LOG_PATH")
if _raw_log_path is None:
    LOG_PATH = DB_PATH.with_name("answerbot.log")
elif _raw_log_path.strip().lower() in ("", "0", "false", "no", "off", "none"):
    LOG_PATH = None
else:
    LOG_PATH = Path(_raw_log_path.strip())

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
# Optional Hub token for gated embed models and anonymous download rate limits.
HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip() or None

# Windowing: a new conversation window starts when the gap between consecutive
# messages exceeds GAP, or the current window grows past MAX_MSGS / MAX_CHARS.
WINDOW_GAP_SECONDS = int(os.getenv("WINDOW_GAP_SECONDS", str(30 * 60)))
WINDOW_MAX_MSGS = int(os.getenv("WINDOW_MAX_MSGS", "25"))
WINDOW_MAX_CHARS = int(os.getenv("WINDOW_MAX_CHARS", "1500"))
WINDOW_OVERLAP = int(os.getenv("WINDOW_OVERLAP", "2"))

# How far back a manual `index --update` re-windows, on top of the open tail, so
# recent edits get picked up. Recent messages are the ones most likely to be
# edited; older ones only drift if reconciled by a full reindex. 0 = tail only.
UPDATE_LOOKBACK_DAYS = int(os.getenv("UPDATE_LOOKBACK_DAYS", "14"))

# How speakers are labelled in windows and answers:
#   "name"    resolved public name, else stable "User N" (default).
#             Never the exporter's contact labels.
#   "id"      always "User N" — no names at all
#   "export"  resolved public name, else the export/contact label (opt-in)
SPEAKER_LABEL = os.getenv("SPEAKER_LABEL", "name").strip().lower()

# Retrieval
TOP_K = int(os.getenv("TOP_K", "10"))
# RRF_K flattens score differences; 60 is the paper's default and suits large
# corpora, but result lists here are short, so a smaller value keeps top ranks
# meaningfully separated.
RRF_K = int(os.getenv("RRF_K", "20"))
WEIGHT_VECTOR = float(os.getenv("WEIGHT_VECTOR", "1.0"))
WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", "0.7"))
# Query terms appearing in more than this fraction of messages are treated as
# stopwords and dropped from the keyword query.
STOPWORD_DF_RATIO = float(os.getenv("STOPWORD_DF_RATIO", "0.25"))
# Fusion still ranks TOP_K windows; the LLM only sees a prefix of that list.
# Always keep MIN_K, then stop at the first hit whose cosine is below COSINE_MIN,
# never more than MAX_K. Defaults match TOP_K so the model sees 10 excerpts.
# COSINE_MIN=0 disables the cutoff (always send MAX_K). Lower MIN_K/MAX_K to trim.
MIN_K = int(os.getenv("MIN_K", str(TOP_K)))
MAX_K = int(os.getenv("MAX_K", str(TOP_K)))
COSINE_MIN = float(os.getenv("COSINE_MIN", "0.7"))
# Fused scores are multiplied by 0.5^(age_days / half_life) using ts_end, so a
# year-old window scores half as much as an equivalent one from today when the
# half-life is 365. 0 disables. Time-range questions skip this — the filter
# already scoped the period.
RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "365"))
# Intra-op threads for the local embedding model. 1 keeps a small VPS usable.
EMBED_THREADS = int(os.getenv("EMBED_THREADS", "1"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CURSOR_API_KEY = os.getenv("CURSOR_API_KEY")
# OpenRouter uses these to attribute traffic; optional.
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "answer-chat-history-bot")
DEFAULT_ANSWER_MODELS = {
    "claude": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-20b",
    "openrouter": "openai/gpt-oss-20b:free",
    "cursor": "composer-2.5",
}
ANSWER_MODEL = os.getenv(
    "ANSWER_MODEL", DEFAULT_ANSWER_MODELS.get(LLM_PROVIDER.lower(), "claude-sonnet-5")
)
# Groq/OpenRouter completion budget. Reasoning models spend this on thinking
# plus the visible answer, so it needs to be larger than Claude's 1024.
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "8192"))
# Cap on prompt + completion tokens for one request. 0 = no extra cap
# (OpenRouter). Groq still applies its on_demand TPM default of 8000 when
# this is unset, because it reserves both against that limit.
ANSWER_MAX_REQUEST_TOKENS = int(os.getenv("ANSWER_MAX_REQUEST_TOKENS", "0"))
# Seconds a user must wait between answers in the same chat. 0 disables.
ANSWER_COOLDOWN_SECONDS = int(os.getenv("ANSWER_COOLDOWN_SECONDS", "8"))
# Sliding-hour caps on LLM answers (not local search). 0 disables. Admins skip.
ANSWER_MAX_PER_USER_PER_HOUR = int(os.getenv("ANSWER_MAX_PER_USER_PER_HOUR", "0"))
ANSWER_MAX_PER_HOUR = int(os.getenv("ANSWER_MAX_PER_HOUR", "0"))
# Persist each question + retrieved window ids locally. "0" / "false" turns it off.
QUERY_LOG = os.getenv("QUERY_LOG", "1").strip().lower() not in ("0", "false", "no", "off")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# How often the live bot re-windows recent history to pick up edits. 0 disables.
LIVE_LOOKBACK_HOURS = float(os.getenv("LIVE_LOOKBACK_HOURS", "6"))
MEMBERSHIP_CACHE_SECONDS = float(os.getenv("MEMBERSHIP_CACHE_SECONDS", "300"))
# Pause between getChatMember calls in /resolve (successes and misses).
RESOLVE_DELAY_SECONDS = float(os.getenv("RESOLVE_DELAY_SECONDS", "0.4"))
# If Telegram asks to wait longer than this, pause the job; /resolve resumes.
RESOLVE_MAX_FLOOD_WAIT = float(os.getenv("RESOLVE_MAX_FLOOD_WAIT", "120"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GITHUB_REPO = os.getenv(
    "GITHUB_REPO", "https://github.com/antmaxi/answer-chat-history-bot"
).strip() or "https://github.com/antmaxi/answer-chat-history-bot"


def parse_telegram_chat_id(raw: str | None) -> int | None:
    """Bot API chat id. A positive value is treated as a supergroup (`-100<id>`)."""
    if raw is None or not str(raw).strip():
        return None
    n = int(str(raw).strip())
    if n > 0:
        return int(f"-100{n}")
    return n


TELEGRAM_CHAT_ID = parse_telegram_chat_id(os.getenv("TELEGRAM_CHAT_ID"))
ADMIN_USER_IDS = {
    int(x) for x in os.getenv("ADMIN_USER_IDS", "").replace(",", " ").split() if x.strip()
}
# How many new messages may accumulate before the tail is re-windowed.
LIVE_REINDEX_EVERY = int(os.getenv("LIVE_REINDEX_EVERY", "20"))


def _display_utc_offset_hours() -> int:
    raw = os.getenv("DISPLAY_UTC_OFFSET_HOURS", "2").strip()
    try:
        hours = int(raw)
    except ValueError:
        return 2
    if hours < -12 or hours > 14:
        return 2
    return hours


# Wall-clock times in bot messages (e.g. /info, /stats) use this UTC offset.
DISPLAY_UTC_OFFSET_HOURS = _display_utc_offset_hours()


def display_timezone() -> timezone:
    return timezone(timedelta(hours=DISPLAY_UTC_OFFSET_HOURS))
