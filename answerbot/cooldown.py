"""In-memory cooldown so one user cannot hammer the LLM."""

from __future__ import annotations

import time


class Cooldown:
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last: dict[tuple, float] = {}

    def remaining(self, key: tuple, *, now: float | None = None, exempt: bool = False) -> float:
        """Seconds left before `key` may fire again. 0 means allowed."""
        if exempt or self.seconds <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        last = self._last.get(key)
        if last is None:
            return 0.0
        wait = self.seconds - (now - last)
        return wait if wait > 0 else 0.0

    def touch(self, key: tuple, *, now: float | None = None) -> None:
        self._last[key] = time.monotonic() if now is None else now
