"""Parse casual time phrases in a question into a unix [start, end] window."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))


@dataclass(frozen=True)
class TimeRange:
    start: int
    end: int
    label: str

    def overlaps(self, ts_start: int, ts_end: int) -> bool:
        return ts_end >= self.start and ts_start <= self.end


def _aware(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _day_bounds(day: datetime) -> tuple[int, int]:
    start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1) - timedelta(seconds=1)
    return int(start.timestamp()), int(end.timestamp())


def _month_bounds(year: int, month: int) -> tuple[int, int]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    last = calendar.monthrange(year, month)[1]
    end = datetime(year, month, last, 23, 59, 59, tzinfo=timezone.utc)
    return int(start.timestamp()), int(end.timestamp())


def _resolve_month_year(month: int, year: int | None, now: datetime) -> int:
    if year is not None:
        return year
    # "in February" in August → this year's February; in January → last year's.
    return now.year if month <= now.month else now.year - 1


def parse_time_range(question: str, now: datetime | None = None) -> TimeRange | None:
    """Return a time window if the question names one, else None.

    Relative phrases ("last week") are wall-clock from `now`, not the chat's
    latest message — that's what people mean when they ask the live bot.
    """
    now = _aware(now or datetime.now(timezone.utc))
    q = question.lower()

    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    if m:
        day = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        start, end = _day_bounds(day)
        return TimeRange(start, end, m.group(1))

    m = re.search(rf"\bin ({_MONTH_ALT})(?:\s+(\d{{4}}))?\b", q)
    if m:
        month = _MONTHS[m.group(1)]
        year = _resolve_month_year(month, int(m.group(2)) if m.group(2) else None, now)
        start, end = _month_bounds(year, month)
        return TimeRange(start, end, f"{calendar.month_name[month]} {year}")

    m = re.search(rf"\bsince ({_MONTH_ALT})(?:\s+(\d{{4}}))?\b", q)
    if m:
        month = _MONTHS[m.group(1)]
        year = _resolve_month_year(month, int(m.group(2)) if m.group(2) else None, now)
        start, _ = _month_bounds(year, month)
        return TimeRange(start, int(now.timestamp()), f"since {calendar.month_name[month]} {year}")

    if re.search(r"\blast night\b", q):
        y = now - timedelta(days=1)
        start = int(y.replace(hour=17, minute=0, second=0, microsecond=0).timestamp())
        end = int(now.replace(hour=8, minute=0, second=0, microsecond=0).timestamp())
        if end < start:
            end = int(now.timestamp())
        return TimeRange(start, end, "last night")

    if re.search(r"\byesterday\b", q):
        start, end = _day_bounds(now - timedelta(days=1))
        return TimeRange(start, end, "yesterday")

    if re.search(r"\btoday\b", q):
        start, end = _day_bounds(now)
        return TimeRange(start, end, "today")

    if re.search(r"\bthis week\b", q):
        start_day = now - timedelta(days=now.weekday())
        start, _ = _day_bounds(start_day)
        return TimeRange(start, int(now.timestamp()), "this week")

    if re.search(r"\b(last|past) week\b", q):
        start = int((now - timedelta(days=7)).timestamp())
        return TimeRange(start, int(now.timestamp()), "last week")

    if re.search(r"\bthis month\b", q):
        start, _ = _month_bounds(now.year, now.month)
        return TimeRange(start, int(now.timestamp()), "this month")

    if re.search(r"\blast month\b", q):
        year, month = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
        start, end = _month_bounds(year, month)
        return TimeRange(start, end, "last month")

    if re.search(r"\bthis year\b", q):
        start = int(datetime(now.year, 1, 1, tzinfo=timezone.utc).timestamp())
        return TimeRange(start, int(now.timestamp()), "this year")

    if re.search(r"\blast year\b", q):
        start = int(datetime(now.year - 1, 1, 1, tzinfo=timezone.utc).timestamp())
        end = int(datetime(now.year, 1, 1, tzinfo=timezone.utc).timestamp()) - 1
        return TimeRange(start, end, "last year")

    return None
