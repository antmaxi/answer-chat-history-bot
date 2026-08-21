"""Golden-set retrieval, time-range filters, cooldowns, and the query log."""

from datetime import datetime, timezone

import numpy as np
import pytest

from answerbot import config, cooldown, db, eval as ev, index, retrieve
from answerbot.ingest.export import load_data
from answerbot.timerange import parse_time_range
from tests.make_fixture import build_export, record_fixture_names
from tests.test_pipeline import seed


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def fake_embed(monkeypatch):
    calls: list[list[str]] = []

    def fake(texts, batch_size=64, progress=False):
        calls.append(list(texts))
        return np.zeros((len(texts), config.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr("answerbot.embed.encode_passages", fake)
    monkeypatch.setattr(
        "answerbot.embed.encode_query",
        lambda q: np.zeros(config.EMBED_DIM, dtype=np.float32),
    )
    return calls


def test_golden_set_keyword_success(conn, fake_embed):
    load_data(conn, build_export(), source="fixture")
    record_fixture_names(conn)
    index.reindex(conn, progress=False)
    results = ev.evaluate(conn, keyword_only=True)
    missed = [r.case.question for r in results if not r.hit]
    assert missed == [], f"keyword success@k missed: {missed}"
    assert ev.recall(results) == 1.0


class TestTimeRangeParse:
    now = datetime(2026, 8, 13, 15, 0, tzinfo=timezone.utc)

    def test_no_phrase(self):
        assert parse_time_range("how much was the ski trip", self.now) is None

    def test_yesterday(self):
        tr = parse_time_range("what happened yesterday", self.now)
        assert tr is not None and tr.label == "yesterday"
        day = datetime(2026, 8, 12, tzinfo=timezone.utc)
        assert tr.start == int(day.timestamp())

    def test_last_week_is_rolling_seven_days(self):
        tr = parse_time_range("decisions last week", self.now)
        assert tr is not None
        assert tr.end - tr.start == 7 * 86400

    def test_in_february_this_year(self):
        tr = parse_time_range("the ski trip in February", self.now)
        assert tr is not None
        assert tr.label == "February 2026"
        feb = datetime(2026, 2, 1, tzinfo=timezone.utc)
        assert tr.start == int(feb.timestamp())

    def test_in_december_is_last_year_before_december(self):
        tr = parse_time_range("what did we say in December", self.now)
        assert tr is not None and tr.label == "December 2025"

    def test_iso_date(self):
        tr = parse_time_range("anything on 2026-02-14", self.now)
        assert tr is not None and tr.label == "2026-02-14"


class TestTimeRangeSearch:
    def test_filters_windows_outside_the_month(self, conn, fake_embed):
        jan = int(datetime(2026, 1, 15, tzinfo=timezone.utc).timestamp())
        feb = int(datetime(2026, 2, 15, tzinfo=timezone.utc).timestamp())
        seed(conn, [(1, jan, "january alpha topic"), (2, feb, "february beta topic")])
        index.reindex(conn, progress=False)
        now = datetime(2026, 8, 1, tzinfo=timezone.utc)

        feb_hits = retrieve.search(conn, "topic in February", chat_id=1, now=now)
        assert feb_hits
        assert all("beta" in h.text for h in feb_hits)
        assert all("alpha" not in h.text for h in feb_hits)

        jan_hits = retrieve.search(conn, "topic in January", chat_id=1, now=now)
        assert jan_hits
        assert all("alpha" in h.text for h in jan_hits)


class TestCooldown:
    def test_blocks_then_expires(self):
        c = cooldown.Cooldown(10)
        key = (1, 2)
        assert c.remaining(key, now=100.0) == 0.0
        c.touch(key, now=100.0)
        assert c.remaining(key, now=105.0) == 5.0
        assert c.remaining(key, now=110.0) == 0.0

    def test_exempt_and_disabled(self):
        c = cooldown.Cooldown(10)
        c.touch((1,), now=0.0)
        assert c.remaining((1,), now=1.0, exempt=True) == 0.0
        assert cooldown.Cooldown(0).remaining((1,), now=1.0) == 0.0


class TestQueryLog:
    def test_answer_writes_a_row(self, conn, fake_embed):
        from answerbot.answer import answer as run_answer

        class FakeLLM:
            model = "fake"
            def complete(self, system, user):
                return "found it [W1]"

        seed(conn, [(1, 1, "pangolin sighting")])
        index.reindex(conn, progress=False)
        run_answer(conn, "pangolin", chat_id=1, llm=FakeLLM())
        row = conn.execute("SELECT question, window_ids, model FROM query_log").fetchone()
        assert row["question"] == "pangolin"
        assert row["model"] == "fake"
        assert row["window_ids"].startswith("[")
