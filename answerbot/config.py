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
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
ADMIN_USER_IDS = {
    int(x) for x in os.getenv("ADMIN_USER_IDS", "").replace(",", " ").split() if x.strip()
}
# How many new messages may accumulate before the tail is re-windowed.
LIVE_REINDEX_EVERY = int(os.getenv("LIVE_REINDEX_EVERY", "20"))
