""" /info: last-update timestamp and the about-the-bot message. """

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from answerbot import config
from answerbot.info import fmt_dt_utc, format_info, last_update


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

    def test_default_is_russian(self, monkeypatch):
        monkeypatch.setattr(config, "GITHUB_REPO", "https://test.repo")
        text = format_info("2026-04-04 14:00:00 UTC+02:00")
        assert "Последнее обновление" in text
        assert "Бот истории чата" in text
        assert "@antmaxi" in text
