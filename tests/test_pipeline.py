"""Tests for the parts with real logic in them: parsing, windowing, FTS queries.

Deliberately no embedding calls — those need the model on disk and are slow.
Retrieval quality is checked by hand with `python -m answerbot.search`.
"""

import sqlite3

import numpy as np
import pytest

from answerbot import config, db, index
from answerbot.answer import Answer
from answerbot.index import build_windows
from answerbot.ingest import live
from answerbot.ingest.export import flatten_text, parse_sender_id, parse_ts
from answerbot.retrieve import Hit, fts_query


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def fake_embed(monkeypatch):
    """Skip the real model: record what gets embedded, return dummy vectors."""
    calls: list[list[str]] = []

    def fake(texts, batch_size=64, progress=False):
        calls.append(list(texts))
        return np.zeros((len(texts), config.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr("answerbot.embed.encode_passages", fake)
    return calls


def msg(msg_id: int, ts: int, text: str = "hi", reply_to=None, sender="Anna") -> sqlite3.Row:
    return {"msg_id": msg_id, "ts": ts, "text": text, "reply_to": reply_to, "sender": sender}


def seed(conn, specs, chat_id=1):
    """specs: list of (msg_id, ts, text). Inserts into messages (FTS via trigger)."""
    conn.executemany(
        "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (?, ?, ?, 'A', ?)",
        [(chat_id, mid, ts, text) for mid, ts, text in specs],
    )
    conn.commit()


def boundaries(conn, chat_id=1):
    return [
        (r["first_msg"], r["last_msg"])
        for r in conn.execute(
            "SELECT first_msg, last_msg FROM windows WHERE chat_id=? ORDER BY first_msg",
            (chat_id,),
        )
    ]


# A message stream that forces several windows: a big gap after msg 12, then a
# long run that trips the size split. Deterministic, so full and incremental
# indexing must agree on the boundaries.
STREAM = (
    [(i, i, f"m{i}") for i in range(1, 13)]
    + [(i, i + config.WINDOW_GAP_SECONDS + 1000, f"m{i}") for i in range(13, 45)]
)


class TestExportParsing:
    def test_flatten_plain_string(self):
        assert flatten_text("hello") == "hello"

    def test_flatten_entity_list(self):
        raw = ["see ", {"type": "link", "text": "example.com"}, " ok"]
        assert flatten_text(raw) == "see example.com ok"

    def test_flatten_empty(self):
        assert flatten_text([]) == ""
        assert flatten_text(None) == ""

    def test_sender_id_strips_prefix(self):
        assert parse_sender_id("user123456") == 123456
        assert parse_sender_id("channel99") == 99
        assert parse_sender_id(None) is None

    def test_ts_prefers_unixtime(self):
        assert parse_ts({"date_unixtime": "1609502400", "date": "2021-01-01T12:00:00"}) == 1609502400

    def test_ts_falls_back_to_iso(self):
        assert parse_ts({"date": "2021-01-01T12:00:00"}) is not None

    def test_ts_missing(self):
        assert parse_ts({}) is None


class TestWindowing:
    def test_long_gap_splits(self):
        # The gap is measured from the preceding message, not the window start.
        late = 10 + config.WINDOW_GAP_SECONDS + 1
        windows = build_windows([msg(1, 0), msg(2, 10), msg(3, late)])
        assert len(windows) == 2
        assert [m["msg_id"] for m in windows[0]] == [1, 2]

    def test_gap_split_has_no_overlap(self):
        """A topic change is a clean break — carrying context across it adds noise."""
        late = 10 + config.WINDOW_GAP_SECONDS + 1
        windows = build_windows([msg(1, 0), msg(2, 10), msg(3, late)])
        assert [m["msg_id"] for m in windows[1]] == [3]

    def test_reply_keeps_window_open_across_gap(self):
        gap = config.WINDOW_GAP_SECONDS * 5
        windows = build_windows([msg(1, 0), msg(2, gap, reply_to=1)])
        assert len(windows) == 1

    def test_size_split_overlaps(self):
        """Splitting mid-conversation carries the tail so answers aren't cut in half."""
        msgs = [msg(i, i * 10) for i in range(config.WINDOW_MAX_MSGS + 3)]
        windows = build_windows(msgs)
        assert len(windows) == 2
        carried = {m["msg_id"] for m in windows[0]} & {m["msg_id"] for m in windows[1]}
        assert len(carried) == config.WINDOW_OVERLAP

    def test_empty_input(self):
        assert build_windows([]) == []


class TestFtsQuery:
    def test_quotes_tokens_to_neutralize_syntax(self, conn):
        # Unquoted, these are FTS5 operators and would be a syntax error.
        q = fts_query(conn, "a AND b")
        assert '"and"' in q.lower()

    def test_drops_high_frequency_terms(self, conn):
        for i in range(20):
            conn.execute(
                "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1,?,?, 'A', ?)",
                (i, i, "the meeting" if i else "the pangolin"),
            )
        conn.commit()
        q = fts_query(conn, "the pangolin")
        assert '"pangolin"' in q
        assert '"the"' not in q  # in 100% of messages, so pure noise

    def test_all_stopwords_keeps_terms(self, conn):
        """Better to search noisily than to drop the keyword arm entirely."""
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1,1,1,'A','the the')"
        )
        conn.commit()
        assert fts_query(conn, "the") == '"the"'

    def test_no_usable_tokens(self, conn):
        assert fts_query(conn, "?! -") == ""


def hit(idx: int) -> Hit:
    return Hit(idx, 1, idx, idx, 0, 0, "Anna", f"body {idx}", 0.1)


class TestAnswerCitations:
    def test_cited_hits_are_only_referenced_windows(self):
        a = Answer("The cost was 200 [W2], see also [W3].", [hit(1), hit(2), hit(3)])
        assert [h.first_msg for h in a.cited_hits()] == [2, 3]

    def test_sources_block_ignores_uncited_and_out_of_range(self):
        a = Answer("answer [W1] and a hallucinated [W9]", [hit(1), hit(2)])
        block = a.sources_block()
        assert "[W1]" in block
        assert "[W2]" not in block  # retrieved but not cited
        assert "W9" not in block    # cited but never existed

    def test_sources_fall_back_to_top_hits_when_uncited(self):
        """An answer with no [W#] tags still gets links to what it drew on."""
        a = Answer("Postgres, for the JSON support.", [hit(1), hit(2), hit(3), hit(4)])
        pairs = a.source_links(limit=3)
        assert [i for i, _ in pairs] == [1, 2, 3]
        assert "t.me" in a.sources_block()

    def test_no_hits_means_no_sources(self):
        a = Answer("I couldn't find that in the chat history.", [])
        assert a.source_links() == []
        assert a.sources_block() == ""

    def test_every_source_carries_a_link(self):
        a = Answer("cost was 200 [W2]", [hit(1), hit(2), hit(3)])
        for _, h in a.source_links():
            assert h.link().startswith("https://t.me/c/")


class TestLiveIngest:
    def test_pending_counts_from_watermark(self, conn):
        for i in range(1, 6):
            live.add_message(conn, chat_id=1, msg_id=i, sender="A", sender_id=1, ts=i, text="hi")
        assert live.pending_count(conn, 1) == 5

        conn.execute("INSERT INTO state (chat_id, last_indexed_msg_id) VALUES (1, 3)")
        conn.commit()
        assert live.pending_count(conn, 1) == 2  # only msgs 4 and 5 are new

    def test_maybe_reindex_waits_for_threshold(self, conn, monkeypatch):
        monkeypatch.setattr(config, "LIVE_REINDEX_EVERY", 3)
        called = []
        monkeypatch.setattr(live, "reindex_tail", lambda c, cid: called.append(cid))

        for i in range(1, 3):
            live.add_message(conn, chat_id=1, msg_id=i, sender="A", sender_id=1, ts=i, text="hi")
        assert live.maybe_reindex(conn, 1) is False
        assert called == []

        live.add_message(conn, chat_id=1, msg_id=3, sender="A", sender_id=1, ts=3, text="hi")
        assert live.maybe_reindex(conn, 1) is True
        assert called == [1]

    def test_add_message_is_idempotent(self, conn):
        live.add_message(conn, chat_id=1, msg_id=1, sender="A", sender_id=1, ts=1, text="first")
        live.add_message(conn, chat_id=1, msg_id=1, sender="A", sender_id=1, ts=1, text="edited")
        rows = conn.execute("SELECT text FROM messages WHERE chat_id=1 AND msg_id=1").fetchall()
        assert len(rows) == 1
        assert rows[0]["text"] == "edited"


class TestIncrementalIndex:
    def test_update_matches_full_reindex(self, fake_embed):
        """Indexing in two passes must yield the same windows as one full pass.

        This is the property that makes the incremental path safe to rely on.
        """
        half = len(STREAM) // 2

        inc = db.connect(":memory:")
        seed(inc, STREAM[:half])
        index.reindex(inc, progress=False)
        seed(inc, STREAM[half:])
        index.update(inc)

        full = db.connect(":memory:")
        seed(full, STREAM)
        index.reindex(full, progress=False)

        assert boundaries(inc) == boundaries(full)
        assert boundaries(inc)  # sanity: there actually are multiple windows
        assert len(boundaries(inc)) > 1

    def test_update_only_embeds_the_tail(self, conn, fake_embed):
        seed(conn, STREAM)
        index.reindex(conn, progress=False)
        total_windows = len(boundaries(conn))
        fake_embed.clear()

        # three new messages arrive contiguously after the last one
        last_ts = STREAM[-1][1]
        seed(conn, [(100 + i, last_ts + 1 + i, f"new{i}") for i in range(3)])
        index.update(conn)

        # exactly one embed call, covering only the reopened tail — far fewer
        # texts than the whole corpus of windows.
        assert len(fake_embed) == 1
        assert len(fake_embed[0]) < total_windows

    def test_update_is_noop_when_nothing_new(self, conn, fake_embed):
        seed(conn, STREAM)
        index.reindex(conn, progress=False)
        fake_embed.clear()

        result = index.update(conn)
        assert result["windows"] == 0
        assert result["chats"] == 0
        assert fake_embed == []  # nothing re-embedded

    def test_update_advances_watermark_and_is_searchable(self, conn, fake_embed):
        seed(conn, STREAM)
        index.reindex(conn, progress=False)

        seed(conn, [(200, STREAM[-1][1] + 5, "pangolin sighting")])
        index.update(conn)

        wm = conn.execute("SELECT last_indexed_msg_id FROM state WHERE chat_id=1").fetchone()[0]
        assert wm == 200
        # the new message made it into a window
        got = conn.execute(
            "SELECT count(*) FROM windows WHERE chat_id=1 AND last_msg>=200"
        ).fetchone()[0]
        assert got == 1

    def test_update_indexes_a_never_indexed_chat(self, conn, fake_embed):
        """update on a fresh chat falls back to a full index."""
        seed(conn, STREAM)
        result = index.update(conn)
        assert result["windows"] == len(boundaries(conn))
        assert len(fake_embed) == 1  # embedded everything, once

    def _window_text_for(self, conn, msg_id):
        return conn.execute(
            "SELECT text FROM windows WHERE chat_id=1 AND first_msg<=? AND last_msg>=?",
            (msg_id, msg_id),
        ).fetchone()["text"]

    def test_pure_tail_update_misses_an_old_edit(self, conn, fake_embed):
        """Baseline: without lookback, an edit to a non-tail message is not refreshed."""
        seed(conn, STREAM)
        index.reindex(conn, progress=False)

        conn.execute("UPDATE messages SET text='EDITED' WHERE chat_id=1 AND msg_id=20")
        conn.commit()  # STREAM has no msg beyond 44, so nothing new to trigger a tail rebuild
        index.update(conn)  # lookback_days=0

        assert "EDITED" not in self._window_text_for(conn, 20)  # window still stale

    def test_lookback_refreshes_a_recent_edit(self, conn, fake_embed):
        seed(conn, STREAM)
        index.reindex(conn, progress=False)
        fake_embed.clear()

        conn.execute("UPDATE messages SET text='EDITED' WHERE chat_id=1 AND msg_id=20")
        conn.commit()
        # STREAM spans well under a day, so any positive lookback covers msg 20.
        result = index.update(conn, lookback_days=1)

        assert "EDITED" in self._window_text_for(conn, 20)  # window rebuilt
        assert result["windows"] > 0
        assert len(fake_embed) == 1  # the refreshed windows were re-embedded

    def test_lookback_leaves_older_history_untouched(self, conn, fake_embed):
        """Lookback rebuilds only recent windows, not the whole corpus."""
        # One message per day for 40 days: 40 daily windows (a day > the gap).
        day_stream = [(i, i * 86400, f"day{i}") for i in range(1, 41)]
        seed(conn, day_stream)
        index.reindex(conn, progress=False)
        total = len(boundaries(conn))
        assert total == 40
        fake_embed.clear()

        index.update(conn, lookback_days=5)  # only the last ~5 days
        assert fake_embed, "expected some re-embedding"
        assert len(fake_embed[0]) <= 6  # ~5 recent windows, far fewer than 40
        assert len(fake_embed[0]) < total


def test_fts_index_stays_in_sync(conn):
    """The triggers, not the app, are responsible for keeping FTS current."""
    conn.execute(
        "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1,1,1,'A','pangolin')"
    )
    conn.commit()
    found = conn.execute(
        "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'pangolin'"
    ).fetchone()[0]
    assert found == 1

    conn.execute("UPDATE messages SET text='armadillo' WHERE msg_id=1")
    conn.commit()
    stale = conn.execute(
        "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH 'pangolin'"
    ).fetchone()[0]
    assert stale == 0
