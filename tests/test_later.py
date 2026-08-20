"""Follow-ups, speaker filters, membership cache, schema helpers, LLM errors."""

import json
import logging
import sys
import urllib.error
import urllib.request

import numpy as np
import pytest

from answerbot import config, db, followup, index, membership, people, retrieve
from tests.make_fixture import build_export
from answerbot.ingest.export import load_data
from answerbot.ingest import live


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def fake_embed(monkeypatch):
    def fake(texts, batch_size=64, progress=False):
        return np.zeros((len(texts), config.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr("answerbot.embed.encode_passages", fake)
    monkeypatch.setattr(
        "answerbot.embed.encode_query",
        lambda q: np.zeros(config.EMBED_DIM, dtype=np.float32),
    )


class TestFollowup:
    def test_standalone_is_unchanged(self):
        q = "how much was the ski trip"
        assert followup.looks_like_followup(q) is False
        assert followup.rewrite(q, "wifi password") == q

    def test_short_pronoun_question_is_stitched(self):
        assert followup.looks_like_followup("how much was it")
        assert followup.rewrite("how much was it", "the ski trip") == (
            "the ski trip — follow-up: how much was it"
        )

    def test_what_about_is_a_followup(self):
        assert followup.looks_like_followup("what about the other one?")

    def test_force_rewrites_even_when_standalone(self):
        q = "when is the ski trip"
        assert not followup.looks_like_followup(q)
        assert followup.rewrite(q, "wifi password", force=True).startswith("wifi password")

    def test_no_prior_returns_the_question(self):
        assert followup.rewrite("how much was it", None) == "how much was it"
        assert followup.rewrite("how much was it", "  ") == "how much was it"


class TestSpeakerParse:
    names = ["Anna Maria", "Anna", "Nino"]

    def test_what_did_x_say(self):
        assert people.parse_speaker("what did Nino say about standup", self.names) == "Nino"

    def test_longest_name_wins(self):
        assert people.parse_speaker("what did Anna Maria say", self.names) == "Anna Maria"

    def test_no_cue_is_not_a_speaker_query(self):
        assert people.parse_speaker("how is Anna doing", self.names) is None

    def test_short_names_are_ignored(self):
        assert people.parse_speaker("what did Ed say", ["Ed", "Nino"]) is None


class TestSpeakerSearch:
    def test_filters_to_named_speaker(self, conn, fake_embed):
        gap = config.WINDOW_GAP_SECONDS + 10
        conn.executemany(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1, ?, ?, ?, ?)",
            [
                (1, 0, "Anna", "the pangolin is in the attic"),
                (2, 5, "Anna", "I saw it yesterday"),
                (3, gap, "Nino", "the armadillo lives by the river"),
                (4, gap + 5, "Nino", "do not feed it"),
            ],
        )
        conn.commit()
        index.reindex(conn, progress=False)

        nino = retrieve.search(conn, "what did Nino say about animals", chat_id=1)
        assert nino
        blob = "\n".join(h.text for h in nino).lower()
        assert "armadillo" in blob
        assert "pangolin" not in blob
        assert all("nino" in h.speakers.lower() for h in nino)

        anna = retrieve.search(conn, "what did Anna say about animals", chat_id=1)
        blob = "\n".join(h.text for h in anna).lower()
        assert "pangolin" in blob
        assert "armadillo" not in blob

    def test_fixture_nino_standup(self, conn, fake_embed):
        load_data(conn, build_export(), source="fixture")
        index.reindex(conn, progress=False)
        hits = retrieve.search(conn, "what did Nino say about standup")
        blob = "\n".join(h.text for h in hits).lower()
        assert "10:30" in blob
        assert all("nino" in h.speakers.lower() for h in hits)

        postgres = retrieve.search(conn, "what did Nino say about postgres")
        blob = "\n".join(h.text for h in postgres).lower()
        assert "postgres" not in blob


class TestMembershipCache:
    def test_ttl_and_invalidate(self):
        c = membership.MembershipCache(10)
        assert c.get(1, 2, now=0) is None
        c.remember(1, 2, True, now=0)
        assert c.get(1, 2, now=5) is True
        assert c.get(1, 2, now=10) is None
        c.remember(1, 2, False, now=20)
        c.remember(1, 3, True, now=20)
        c.invalidate(user_id=1, chat_id=2)
        assert c.get(1, 2, now=21) is None
        assert c.get(1, 3, now=21) is True


class TestSchemaMigrate:
    def test_ensure_column_adds_once(self, conn):
        assert db.ensure_column(conn, "messages", "extra_col", "TEXT") is True
        cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
        assert "extra_col" in cols
        assert db.ensure_column(conn, "messages", "extra_col", "TEXT") is False

    def test_ensure_column_rejects_injection(self, conn):
        with pytest.raises(ValueError):
            db.ensure_column(conn, "messages;drop", "x", "INT")
        with pytest.raises(ValueError):
            db.ensure_column(conn, "messages", "x;drop", "INT")

    def test_dm_prefs_roundtrip(self, conn):
        assert db.get_dm_chat(conn, 7) is None
        db.set_dm_chat(conn, 7, -100)
        assert db.get_dm_chat(conn, 7) == -100
        db.set_dm_chat(conn, 7, None)
        assert db.get_dm_chat(conn, 7) is None

    def test_connect_creates_later_tables(self, conn):
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "dm_prefs" in tables
        assert "query_log" in tables
        assert "aliases" in tables


class TestChatIdAlign:
    def test_remap_moves_messages_and_windows(self, conn):
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (149, 1, 1, 'A', 'hi')"
        )
        conn.execute(
            "INSERT INTO windows (chat_id, first_msg, last_msg, ts_start, ts_end, speakers, text) "
            "VALUES (149, 1, 1, 1, 1, 'A', 'A: hi')"
        )
        conn.execute("INSERT INTO state (chat_id, last_indexed_msg_id) VALUES (149, 1)")
        db.set_dm_chat(conn, 7, 149)
        assert db.remap_chat_id(conn, 149, -100149) == 1
        assert {r[0] for r in conn.execute("SELECT DISTINCT chat_id FROM messages")} == {-100149}
        assert conn.execute("SELECT chat_id FROM windows").fetchone()[0] == -100149
        assert conn.execute("SELECT chat_id FROM state").fetchone()[0] == -100149
        assert db.get_dm_chat(conn, 7) == -100149

    def test_remap_merges_when_both_ids_exist(self, conn):
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (149, 1, 1, 'A', 'export')"
        )
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (-100149, 1, 1, 'A', 'live')"
        )
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (149, 2, 2, 'A', 'old only')"
        )
        assert db.remap_chat_id(conn, 149, -100149) == 2
        rows = {
            r[0]: r[1]
            for r in conn.execute("SELECT msg_id, text FROM messages WHERE chat_id=-100149")
        }
        assert rows[1] == "live"
        assert rows[2] == "old only"
        assert conn.execute("SELECT count(*) FROM messages WHERE chat_id=149").fetchone()[0] == 0

    def test_load_data_stores_bot_api_id(self, conn):
        result = load_data(conn, build_export(), source="fixture")
        assert result["chat_id"] == -1001234567890
        chats = {r[0] for r in conn.execute("SELECT DISTINCT chat_id FROM messages")}
        assert chats == {-1001234567890}

    def test_live_adopts_desktop_export_id(self, conn):
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) "
            "VALUES (1234567890, 1, 1, 'A', 'export')"
        )
        live.add_message(conn, -1001234567890, 2, "A", 1, 2, "fresh")
        chats = {r[0] for r in conn.execute("SELECT DISTINCT chat_id FROM messages")}
        assert chats == {-1001234567890}
        n = conn.execute("SELECT count(*) FROM messages").fetchone()[0]
        assert n == 2


class TestGemini:
    def _llm(self, text):
        from answerbot.llm import GeminiLLM

        class FakeResp:
            def __init__(self, value):
                self.text = value

        class FakeClient:
            def __init__(self, value):
                self._text = value
                self.models = self
                self.kwargs = None

            def generate_content(self, **kw):
                self.kwargs = kw
                return FakeResp(self._text)

        llm = GeminiLLM.__new__(GeminiLLM)
        llm.model = "gemini-2.5-flash"
        llm.client = FakeClient(text)
        return llm

    def test_complete_returns_text(self):
        llm = self._llm("  hello  ")
        assert llm.complete("system", "user") == "hello"
        assert llm.client.kwargs["model"] == "gemini-2.5-flash"
        assert llm.client.kwargs["contents"] == "user"
        assert llm.client.kwargs["config"]["system_instruction"] == "system"

    def test_empty_response_raises(self):
        with pytest.raises(RuntimeError, match="no text"):
            self._llm("  ").complete("s", "u")

    def test_get_llm_dispatches(self, monkeypatch):
        from answerbot import llm as llm_mod

        monkeypatch.setattr(config, "LLM_PROVIDER", "gemini")
        monkeypatch.setattr(llm_mod, "GeminiLLM", lambda: "gemini-llm")
        assert llm_mod.get_llm() == "gemini-llm"


class TestOllamaErrors:
    def test_empty_response_raises(self, monkeypatch):
        from answerbot.llm import OllamaLLM

        class FakeResp:
            def read(self):
                return json.dumps({"response": ""}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: FakeResp())
        with pytest.raises(RuntimeError, match="no text"):
            OllamaLLM(model="x", host="http://localhost:9").complete("s", "u")

    def test_url_error_is_wrapped(self, monkeypatch):
        from answerbot.llm import OllamaLLM

        def boom(*a, **k):
            raise urllib.error.URLError("refused")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        with pytest.raises(RuntimeError, match="Ollama"):
            OllamaLLM(model="x", host="http://localhost:9").complete("s", "u")


def _error_record(msg="failed to answer", exc=True):
    exc_info = None
    if exc:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            exc_info = sys.exc_info()
    return logging.LogRecord(
        "answerbot", logging.ERROR, __file__, 1, msg, (), exc_info
    )


class TestAdminErrorHandler:
    def test_format_includes_message_and_traceback(self):
        from answerbot.adminlog import format_error

        text = format_error(_error_record())
        assert "ERROR answerbot: failed to answer" in text
        assert "RuntimeError: boom" in text

    def test_format_truncates(self):
        from answerbot.adminlog import format_error

        text = format_error(_error_record("x" * 200), max_len=40)
        assert len(text) == 40
        assert text.endswith("…")

    def test_prepare_coalesces_bursts(self):
        from answerbot.adminlog import AdminErrorHandler

        h = AdminErrorHandler(min_interval=60)
        first = h.prepare(_error_record("one"))
        assert first is not None and "one" in first
        assert h.prepare(_error_record("two")) is None
        assert h.prepare(_error_record("three")) is None
        h._last_sent = 0
        later = h.prepare(_error_record("four"))
        assert later is not None
        assert "four" in later
        assert "2 more error(s) suppressed" in later

    def test_notify_failure_is_not_forwarded(self):
        from answerbot.adminlog import AdminErrorHandler

        h = AdminErrorHandler(min_interval=0)
        rec = _error_record("failed to notify admin 1 (Bot is up)", exc=False)
        assert h.prepare(rec) is None
