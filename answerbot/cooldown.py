"""In-memory cooldown and hourly quotas so one user cannot hammer the LLM."""

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


class Quota:
    """Sliding-window cap: `limit` hits per `window` seconds for each key. 0 = off."""

    def __init__(self, limit: int, window: float = 3600.0):
        self.limit = limit
        self.window = window
        self._hits: dict[tuple, list[float]] = {}

    def _prune(self, key: tuple, now: float) -> list[float]:
        cutoff = now - self.window
        hits = [t for t in self._hits.get(key, []) if t > cutoff]
        if hits:
            self._hits[key] = hits
        else:
            self._hits.pop(key, None)
        return hits

    def remaining(self, key: tuple, *, now: float | None = None, exempt: bool = False) -> float:
        """Seconds until a slot frees. 0 means allowed."""
        if exempt or self.limit <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        hits = self._prune(key, now)
        if len(hits) < self.limit:
            return 0.0
        return hits[0] + self.window - now

    def touch(self, key: tuple, *, now: float | None = None) -> None:
        if self.limit <= 0:
            return
        now = time.monotonic() if now is None else now
        hits = self._prune(key, now)
        hits.append(now)
        self._hits[key] = hits
