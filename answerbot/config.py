"""Configuration, read once from the environment / .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(os.getenv("DB_PATH", "answerbot.db"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

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
#   "name"  resolved real name, falling back to the export label (default)
#   "id"    stable anonymous "User N" (see aliases table) — no names, no real id
SPEAKER_LABEL = os.getenv("SPEAKER_LABEL", "name")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "8"))
# RRF_K flattens score differences; 60 is the paper's default and suits large
# corpora, but result lists here are short, so a smaller value keeps top ranks
# meaningfully separated.
RRF_K = int(os.getenv("RRF_K", "20"))
WEIGHT_VECTOR = float(os.getenv("WEIGHT_VECTOR", "1.0"))
WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", "0.7"))
# Query terms appearing in more than this fraction of messages are treated as
# stopwords and dropped from the keyword query.
STOPWORD_DF_RATIO = float(os.getenv("STOPWORD_DF_RATIO", "0.25"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
_DEFAULT_ANSWER_MODEL = {
    "claude": "claude-sonnet-5",
    "gemini": "gemini-2.5-flash",
}.get(LLM_PROVIDER.lower(), "claude-sonnet-5")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", _DEFAULT_ANSWER_MODEL)
# Seconds a user must wait between answers in the same chat. 0 disables.
ANSWER_COOLDOWN_SECONDS = int(os.getenv("ANSWER_COOLDOWN_SECONDS", "8"))
# Persist each question + retrieved window ids locally. "0" / "false" turns it off.
QUERY_LOG = os.getenv("QUERY_LOG", "1").strip().lower() not in ("0", "false", "no", "off")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# How often the live bot re-windows recent history to pick up edits. 0 disables.
LIVE_LOOKBACK_HOURS = float(os.getenv("LIVE_LOOKBACK_HOURS", "6"))
MEMBERSHIP_CACHE_SECONDS = float(os.getenv("MEMBERSHIP_CACHE_SECONDS", "300"))

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_USER_IDS = {
    int(x) for x in os.getenv("ADMIN_USER_IDS", "").replace(",", " ").split() if x.strip()
}
# How many new messages may accumulate before the tail is re-windowed.
LIVE_REINDEX_EVERY = int(os.getenv("LIVE_REINDEX_EVERY", "20"))
