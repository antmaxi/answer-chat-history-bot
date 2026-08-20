"""Forward ERROR+ log records to admin Telegram DMs.

Attached for the life of polling. Failures while delivering a DM are never
logged at ERROR, so a dead chat cannot recurse into more DMs. Bursts are
collapsed: the first error goes out immediately, later ones in the cooldown
window are counted and mentioned on the next send.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable

# Telegram's hard cap is 4096; leave headroom for the suppressed-count suffix.
MAX_LEN = 3500
MIN_INTERVAL = 20.0


def format_error(record: logging.LogRecord, max_len: int = MAX_LEN) -> str:
    """Logger, message, and traceback — truncated to a Telegram-safe length."""
    parts = [f"{record.levelname} {record.name}: {record.getMessage()}"]
    if record.exc_info and record.exc_info[0] is not None:
        parts.append(logging.Formatter().formatException(record.exc_info))
    elif record.exc_text:
        parts.append(record.exc_text)
    text = "\n\n".join(parts)
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


class AdminErrorHandler(logging.Handler):
    def __init__(self, min_interval: float = MIN_INTERVAL, max_len: int = MAX_LEN):
        super().__init__(level=logging.ERROR)
        self.min_interval = min_interval
        self.max_len = max_len
        self._loop: asyncio.AbstractEventLoop | None = None
        self._send: Callable[[str], Awaitable[None]] | None = None
        self._lock = threading.Lock()
        self._last_sent = 0.0
        self._suppressed = 0

    def attach(self, loop: asyncio.AbstractEventLoop, send: Callable[[str], Awaitable[None]]) -> None:
        self._loop = loop
        self._send = send
        root = logging.getLogger()
        if self not in root.handlers:
            root.addHandler(self)

    def detach(self) -> None:
        logging.getLogger().removeHandler(self)
        self._loop = None
        self._send = None

    def prepare(self, record: logging.LogRecord) -> str | None:
        """Text to send, or None if this record should be skipped / coalesced."""
        msg = record.getMessage()
        if "failed to notify admin" in msg:
            return None
        text = format_error(record, self.max_len)
        with self._lock:
            now = time.monotonic()
            if self._last_sent and now - self._last_sent < self.min_interval:
                self._suppressed += 1
                return None
            n = self._suppressed
            self._suppressed = 0
            self._last_sent = now
        if n:
            extra = f"\n\n…and {n} more error(s) suppressed"
            budget = 4096 - len(extra)
            if len(text) > budget:
                text = text[: budget - 1] + "…"
            text += extra
        return text

    def emit(self, record: logging.LogRecord) -> None:
        loop, send = self._loop, self._send
        if loop is None or send is None or loop.is_closed():
            return
        try:
            text = self.prepare(record)
            if not text:
                return
            asyncio.run_coroutine_threadsafe(self._deliver(text), loop)
        except Exception:
            pass

    async def _deliver(self, text: str) -> None:
        send = self._send
        if send is None:
            return
        try:
            await send(text)
        except Exception:
            pass
