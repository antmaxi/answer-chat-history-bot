""" /info: last-update timestamp and the about-the-bot message. """

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from answerbot import config
from answerbot.info import (
    fmt_dt_utc,
    fmt_duration_ms,
    format_info,
    format_latency,
    format_stats,
    last_update,
    provider_label,
)


GIT_COMMIT_EPOCH = 1775649600  # 2026-04-04 12:00:00 UTC


@pytest.fixture(autouse=True)
def _display_offset_utc2(monkeypatch):
    monkeypatch.setattr(config, "DISPLAY_UTC_OFFSET_HOURS", 2)


class TestFmtDtUtc:
    def test_positive_offset(self):
        dt = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        assert fmt_dt_utc(dt) == "2026-04-04 12:00:00 UTC+02:00"

    def test_zero_offset(self):
        dt = datetime(2026, 4, 4, 12, 0, 0, tzinfo=UTC)
        assert fmt_dt_utc(dt) == "2026-04-04 14:00:00 UTC+02:00"

    def test_negative_offset(self):
        dt = datetime(2026, 4, 4, 12, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
        assert fmt_dt_utc(dt) == "2026-04-04 19:00:00 UTC+02:00"

    def test_naive_as_utc(self):
        assert fmt_dt_utc(datetime(2026, 4, 4, 12, 0, 0)) == "2026-04-04 14:00:00 UTC+02:00"


class TestLastUpdate:
    def test_from_git(self, monkeypatch):
        monkeypatch.setattr("answerbot.info._git_root", lambda: Path("/repo"))
        monkeypatch.setattr(
            "answerbot.info.subprocess.check_output",
            lambda *a, **k: f"{GIT_COMMIT_EPOCH}\n".encode(),
        )
        expected = fmt_dt_utc(datetime.fromtimestamp(GIT_COMMIT_EPOCH, tz=UTC))
        assert last_update() == expected

    def test_git_failure_falls_back_to_mtime(self, monkeypatch):
        monkeypatch.setattr("answerbot.info._git_root", lambda: Path("/repo"))

        def boom(*a, **k):
            raise RuntimeError("git error")

        monkeypatch.setattr("answerbot.info.subprocess.check_output", boom)
        monkeypatch.setattr("answerbot.info.os.path.getmtime", lambda p: 1775728800)
        expected = fmt_dt_utc(datetime.fromtimestamp(1775728800, tz=UTC))
        assert last_update() == expected

    def test_no_git_dir_falls_back_to_mtime(self, monkeypatch):
        monkeypatch.setattr("answerbot.info._git_root", lambda: None)
        called = {"git": False}

        def git(*a, **k):
            called["git"] = True
            return b"0\n"

        monkeypatch.setattr("answerbot.info.subprocess.check_output", git)
        monkeypatch.setattr("answerbot.info.os.path.getmtime", lambda p: 1775728800)
        expected = fmt_dt_utc(datetime.fromtimestamp(1775728800, tz=UTC))
        assert last_update() == expected
        assert called["git"] is False

    def test_mtime_failure_is_unknown(self, monkeypatch):
        monkeypatch.setattr("answerbot.info._git_root", lambda: None)

        def boom(p):
            raise OSError("no file")

        monkeypatch.setattr("answerbot.info.os.path.getmtime", boom)
        assert last_update() == "unknown"


class TestFormatInfo:
    @pytest.fixture(autouse=True)
    def _llm(self, monkeypatch):
        monkeypatch.setattr(config, "LLM_PROVIDER", "claude")
        monkeypatch.setattr(config, "ANSWER_MODEL", "claude-sonnet-5")

    def test_includes_repo_contact_and_timestamp_en(self, monkeypatch):
        monkeypatch.setattr(config, "GITHUB_REPO", "https://test.repo")
        stamp = "2026-04-04 14:00:00 UTC+02:00"
        text = format_info(stamp, "en")
        assert "Chat History Bot" in text
        assert "Last update" in text
        assert stamp in text
        assert "https://test.repo" in text
        assert "@antmaxi" in text
        assert "UTC+02:00" in text
        assert "claude-sonnet-5" in text
        assert "Claude" in text
        assert "zero-retention" not in text

    def test_default_is_russian(self, monkeypatch):
        monkeypatch.setattr(config, "GITHUB_REPO", "https://test.repo")
        text = format_info("2026-04-04 14:00:00 UTC+02:00")
        assert "Последнее обновление" in text
        assert "Бот истории чата" in text
        assert "@antmaxi" in text

    def test_appends_index_stats(self, monkeypatch):
        monkeypatch.setattr(config, "GITHUB_REPO", "https://test.repo")
        s = {
            "messages": 10,
            "windows": 3,
            "embedded": 3,
            "chats": 1,
            "first_message": "2023-11-14 22:13 UTC",
            "last_message": "2023-11-14 23:13 UTC",
        }
        text = format_info("2026-04-04 14:00:00 UTC+02:00", "en", stats=s)
        assert "Last update" in text
        assert "messages: 10" in text
        assert "windows: 3" in text
        assert "embedded: 3" in text
        assert "chats: 1" in text
        assert "first: 2023-11-14 22:13 UTC" in text
        assert "last: 2023-11-14 23:13 UTC" in text

    def test_format_stats_omits_span_when_empty(self):
        text = format_stats(
            {"messages": 0, "windows": 0, "embedded": 0, "chats": 0}, "en"
        )
        assert "messages: 0" in text
        assert "first:" not in text
        assert "questions:" not in text

    def test_format_stats_includes_question_counts_when_requested(self):
        s = {
            "messages": 0,
            "windows": 0,
            "embedded": 0,
            "chats": 0,
            "questions_day": 2,
            "questions_day_admin": 1,
            "questions_day_other": 1,
            "questions_week": 5,
            "questions_week_admin": 1,
            "questions_week_other": 4,
            "questions_month": 9,
            "questions_month_admin": 2,
            "questions_month_other": 7,
            "latency_day": {
                "n": 2,
                "median_ms": 1500.0,
                "std_ms": 707.1067811865476,
                "min_ms": 1000.0,
                "max_ms": 2000.0,
            },
        }
        text = format_stats(s, "en", questions=True)
        assert "questions:" in text
        assert "last day: 2 (admin: 1, others: 1)" in text
        assert "last week: 5 (admin: 1, others: 4)" in text
        assert "last month: 9 (admin: 2, others: 7)" in text
        assert "ask time:" in text
        assert "last day: 1.5s ± 0.7s (min 1.0s / max 2.0s)" in text
        ru = format_stats(s, "ru", questions=True)
        assert "вопросов:" in ru
        assert "за сутки: 2 (админы: 1, остальные: 1)" in ru
        assert "за неделю: 5 (админы: 1, остальные: 4)" in ru
        assert "за месяц: 9 (админы: 2, остальные: 7)" in ru
        assert "время запроса:" in ru
        assert "нет данных" in ru

    def test_format_info_omits_question_counts(self, monkeypatch):
        monkeypatch.setattr(config, "GITHUB_REPO", "https://test.repo")
        s = {
            "messages": 10,
            "windows": 3,
            "embedded": 3,
            "chats": 1,
            "questions_day": 2,
            "questions_week": 5,
            "questions_month": 9,
        }
        text = format_info("2026-04-04 14:00:00 UTC+02:00", "en", stats=s)
        assert "messages: 10" in text
        assert "questions:" not in text
        assert "last day:" not in text
        assert "ask time:" not in text

    def test_cursor_includes_zero_retention(self, monkeypatch):
        monkeypatch.setattr(config, "GITHUB_REPO", "https://test.repo")
        monkeypatch.setattr(config, "LLM_PROVIDER", "cursor")
        monkeypatch.setattr(config, "ANSWER_MODEL", "composer-2.5")
        en = format_info("2026-04-04 14:00:00 UTC+02:00", "en")
        assert "composer-2.5" in en
        assert "Cursor" in en
        assert "zero-retention" in en
        assert "not stored after the request" in en
        assert "not used to train models" in en
        ru = format_info("2026-04-04 14:00:00 UTC+02:00", "ru")
        assert "composer-2.5" in ru
        assert "Cursor" in ru
        assert "нулевого хранения" in ru
        assert "не сохраняются после запроса" in ru
        assert "не используются для обучения моделей" in ru


class TestProviderLabel:
    def test_known_and_unknown(self):
        assert provider_label("cursor") == "Cursor"
        assert provider_label("openrouter") == "OpenRouter"
        assert provider_label("Custom") == "Custom"


class TestFormatLatency:
    def test_duration_is_seconds_with_one_decimal(self):
        assert fmt_duration_ms(0) == "0.0s"
        assert fmt_duration_ms(1500) == "1.5s"
        assert fmt_duration_ms(707.1067811865476) == "0.7s"

    def test_none_windows_are_na(self):
        text = format_latency({}, "en")
        assert "ask time:" in text
        assert text.count("n/a") == 3
        ru = format_latency({}, "ru")
        assert "время запроса:" in ru
        assert ru.count("нет данных") == 3

    def test_single_sample_has_zero_std(self):
        text = format_latency(
            {
                "latency_day": {
                    "n": 1,
                    "median_ms": 1200.0,
                    "std_ms": 0.0,
                    "min_ms": 1200.0,
                    "max_ms": 1200.0,
                }
            },
            "en",
        )
        assert "last day: 1.2s ± 0.0s (min 1.2s / max 1.2s)" in text
        assert "last week: n/a" in text
