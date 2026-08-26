"""Process-wide logging: timestamped stderr plus a rotating file next to the DB.

Idempotent. `LOG_PATH=off` keeps stderr only. Uncaught exceptions are logged
at CRITICAL so they hit the file and, once polling is up, the admin DM handler.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from . import config

_FMT = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
_NOISY = (
    "httpx",
    "httpcore",
    "aiohttp",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "filelock",
)
_STREAM_MARK = "_answerbot_stream"
_FILE_MARK = "_answerbot_file"
_POLL_FILTER_MARK = "_answerbot_poll"


def is_transient_poll_error(record: logging.LogRecord) -> bool:
    """aiogram already retries getUpdates network errors; they are not app failures."""
    if record.name != "aiogram.dispatcher":
        return False
    return "failed to fetch updates" in record.getMessage().lower()


class TransientPollFilter(logging.Filter):
    """Log Telegram long-poll disconnects at WARNING so they do not page admins."""

    def filter(self, record: logging.LogRecord) -> bool:
        if is_transient_poll_error(record) and record.levelno >= logging.ERROR:
            record.levelno = logging.WARNING
            record.levelname = "WARNING"
        return True


def _level() -> int:
    return getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO)


def _excepthook(exc_type, exc, tb) -> None:
    logging.getLogger("answerbot").critical("unhandled exception", exc_info=(exc_type, exc, tb))
    sys.__excepthook__(exc_type, exc, tb)


def asyncio_handler(loop, context: dict) -> None:
    """Log unhandled asyncio errors; used as the event-loop exception handler."""
    exc = context.get("exception")
    msg = context.get("message", "unhandled asyncio exception")
    log = logging.getLogger("answerbot")
    if exc is not None:
        log.error(msg, exc_info=(type(exc), exc, exc.__traceback__))
    else:
        log.error(msg)


def setup() -> None:
    """Attach stderr + rotating file handlers to the root logger once."""
    level = _level()
    root = logging.getLogger()
    root.setLevel(level)

    if not any(getattr(h, _STREAM_MARK, False) for h in root.handlers):
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(_FMT)
        stream.setLevel(level)
        setattr(stream, _STREAM_MARK, True)
        root.addHandler(stream)

    path = config.LOG_PATH
    if path is not None:
        key = str(path)
        if not any(getattr(h, _FILE_MARK, None) == key for h in root.handlers):
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
            fh.setFormatter(_FMT)
            fh.setLevel(level)
            setattr(fh, _FILE_MARK, key)
            root.addHandler(fh)
            logging.getLogger("answerbot").info("logging to %s", path)

    for name in _NOISY:
        if level >= logging.INFO:
            logging.getLogger(name).setLevel(logging.WARNING)

    dispatcher = logging.getLogger("aiogram.dispatcher")
    if not any(getattr(f, _POLL_FILTER_MARK, False) for f in dispatcher.filters):
        filt = TransientPollFilter()
        setattr(filt, _POLL_FILTER_MARK, True)
        dispatcher.addFilter(filt)

    if sys.excepthook is sys.__excepthook__:
        sys.excepthook = _excepthook
