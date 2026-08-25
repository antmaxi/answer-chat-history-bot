"""Follow-ups, speaker filters, membership cache, schema helpers, LLM errors."""

import json
import logging
import sys
import urllib.error
import urllib.request

import numpy as np
import pytest

from answerbot import config, db, followup, index, membership, people, retrieve
from tests.make_fixture import build_export, record_fixture_names
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
            "INSERT INTO messages (chat_id, msg_id, ts, sender, sender_id, text) VALUES (1, ?, ?, ?, ?, ?)",
            [
                (1, 0, "Anna", 10, "the pangolin is in the attic"),
                (2, 5, "Anna", 10, "I saw it yesterday"),
                (3, gap, "Nino", 20, "the armadillo lives by the river"),
                (4, gap + 5, "Nino", 20, "do not feed it"),
            ],
        )
        people.record(conn, 10, "Anna", None, "live")
        people.record(conn, 20, "Nino", None, "live")
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
        record_fixture_names(conn)
        index.reindex(conn, progress=False)
        hits = retrieve.search(conn, "what did Nino say about standup")
        blob = "\n".join(h.text for h in hits).lower()
        assert "10:30" in blob
        assert all("nino" in h.speakers.lower() for h in hits)

        postgres = retrieve.search(conn, "what did Nino say about postgres")
        blob = "\n".join(h.text for h in postgres).lower()
        assert "postgres" not in blob


class TestTelegramChatId:
    def test_empty_is_none(self):
        assert config.parse_telegram_chat_id(None) is None
        assert config.parse_telegram_chat_id("") is None
        assert config.parse_telegram_chat_id("  ") is None

    def test_bot_api_supergroup_is_kept(self):
        assert config.parse_telegram_chat_id("-1001495905530") == -1001495905530

    def test_positive_id_becomes_supergroup(self):
        assert config.parse_telegram_chat_id("1495905530") == -1001495905530
        assert config.parse_telegram_chat_id(" 1495905530 ") == -1001495905530


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

    def test_is_chat_member_statuses(self):
        from types import SimpleNamespace

        assert membership.is_chat_member(SimpleNamespace(status="member")) is True
        assert membership.is_chat_member(SimpleNamespace(status="administrator")) is True
        assert membership.is_chat_member(SimpleNamespace(status="creator")) is True
        assert membership.is_chat_member(SimpleNamespace(status="restricted", is_member=True)) is True
        assert membership.is_chat_member(SimpleNamespace(status="restricted", is_member=False)) is False
        assert membership.is_chat_member(SimpleNamespace(status="left")) is False
        assert membership.is_chat_member(SimpleNamespace(status="kicked")) is False


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

    def test_user_lang_defaults_to_russian(self, conn):
        assert db.get_user_lang(conn, 7) == "ru"

    def test_user_lang_roundtrip(self, conn):
        assert db.set_user_lang(conn, 7, "en") == "en"
        assert db.get_user_lang(conn, 7) == "en"
        assert db.set_user_lang(conn, 7, "ru") == "ru"
        assert db.get_user_lang(conn, 7) == "ru"

    def test_user_lang_unknown_falls_back(self, conn):
        conn.execute("INSERT INTO user_prefs (user_id, lang) VALUES (7, 'de')")
        conn.commit()
        assert db.get_user_lang(conn, 7) == "ru"
        assert db.set_user_lang(conn, 7, "de") == "ru"
        assert db.get_user_lang(conn, 7) == "ru"

    def test_connect_creates_later_tables(self, conn):
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "dm_prefs" in tables
        assert "query_log" in tables
        assert "aliases" in tables
        assert "user_prefs" in tables

    def test_stats_includes_message_span(self, conn, monkeypatch):
        monkeypatch.setattr(config, "DISPLAY_UTC_OFFSET_HOURS", 2)
        empty = db.stats(conn)
        assert empty["messages"] == 0
        assert empty["first_message"] is None
        assert empty["last_message"] is None
        assert empty["questions_day"] == 0
        assert empty["questions_week"] == 0
        assert empty["questions_month"] == 0
        assert empty["latency_day"] is None
        assert empty["latency_week"] is None
        assert empty["latency_month"] is None
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1, 1, 1700000000, 'A', 'old')"
        )
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1, 2, 1700003600, 'A', 'new')"
        )
        s = db.stats(conn)
        assert s["messages"] == 2
        assert s["first_message"] == "2023-11-15 00:13:20 UTC+02:00"
        assert s["last_message"] == "2023-11-15 01:13:20 UTC+02:00"

    def test_stats_span_uses_display_timezone(self, conn, monkeypatch):
        monkeypatch.setattr(config, "DISPLAY_UTC_OFFSET_HOURS", 0)
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1, 1, 1700000000, 'A', 'old')"
        )
        s = db.stats(conn)
        assert s["first_message"] == "2023-11-14 22:13:20 UTC+00:00"
        assert s["last_message"] == "2023-11-14 22:13:20 UTC+00:00"

    def test_stats_counts_questions_in_rolling_windows(self, conn):
        now = 1_800_000_000
        rows = [
            (now - 3600, "today"),
            (now - 2 * 86400, "this week"),
            (now - 10 * 86400, "this month"),
            (now - 40 * 86400, "older"),
        ]
        conn.executemany(
            "INSERT INTO query_log (ts, question, window_ids, cited_ids) VALUES (?, ?, '[]', '[]')",
            rows,
        )
        conn.commit()
        s = db.stats(conn, now=now)
        assert s["questions_day"] == 1
        assert s["questions_week"] == 2
        assert s["questions_month"] == 3
        assert s["questions_day_admin"] == 0
        assert s["questions_day_other"] == 1
        assert s["questions_week_other"] == 2
        assert s["questions_month_other"] == 3

    def test_stats_splits_questions_by_admin(self, conn, monkeypatch):
        monkeypatch.setattr(config, "ADMIN_USER_IDS", {10, 20})
        now = 1_800_000_000
        rows = [
            (now - 3600, "admin today", 10),
            (now - 3600, "other today", 99),
            (now - 2 * 86400, "admin this week", 20),
            (now - 10 * 86400, "other this month", 99),
            (now - 10 * 86400, "no user", None),
        ]
        conn.executemany(
            "INSERT INTO query_log (ts, question, window_ids, cited_ids, user_id) "
            "VALUES (?, ?, '[]', '[]', ?)",
            rows,
        )
        conn.commit()
        s = db.stats(conn, now=now)
        assert s["questions_day"] == 2
        assert s["questions_day_admin"] == 1
        assert s["questions_day_other"] == 1
        assert s["questions_week"] == 3
        assert s["questions_week_admin"] == 2
        assert s["questions_week_other"] == 1
        assert s["questions_month"] == 5
        assert s["questions_month_admin"] == 2
        assert s["questions_month_other"] == 3

    def test_stats_summarizes_ask_latency(self, conn):
        now = 1_800_000_000
        rows = [
            (now - 3600, 1000),
            (now - 3600, 2000),
            (now - 2 * 86400, 3000),
            (now - 10 * 86400, 4000),
            (now - 40 * 86400, 99999),
            (now - 100, None),
        ]
        conn.executemany(
            "INSERT INTO query_log (ts, question, window_ids, cited_ids, latency_ms) "
            "VALUES (?, 'q', '[]', '[]', ?)",
            rows,
        )
        conn.commit()
        s = db.stats(conn, now=now)
        day = s["latency_day"]
        assert day["n"] == 2
        assert day["median_ms"] == 1500
        assert day["min_ms"] == 1000
        assert day["max_ms"] == 2000
        assert abs(day["std_ms"] - 707.1067811865476) < 1e-6
        week = s["latency_week"]
        assert week["n"] == 3
        assert week["median_ms"] == 2000
        assert week["std_ms"] == 1000
        assert week["min_ms"] == 1000
        assert week["max_ms"] == 3000
        month = s["latency_month"]
        assert month["n"] == 4
        assert month["median_ms"] == 2500
        assert month["min_ms"] == 1000
        assert month["max_ms"] == 4000


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


class TestCursor:
    def _install_sdk(self, monkeypatch, prompt):
        from answerbot import llm as llm_mod

        class AgentOptions:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class LocalAgentOptions:
            def __init__(self, *, cwd):
                self.cwd = cwd

        class CursorAgentError(Exception):
            def __init__(self, message, is_retryable=False):
                super().__init__(message)
                self.message = message
                self.is_retryable = is_retryable

        class Agent:
            @staticmethod
            def prompt(message, options):
                return prompt(message, options)

        monkeypatch.setattr(
            llm_mod,
            "_cursor_sdk",
            lambda: (Agent, AgentOptions, CursorAgentError, LocalAgentOptions),
        )
        return CursorAgentError

    def test_default_answer_model(self):
        assert config.DEFAULT_ANSWER_MODELS["cursor"] == "composer-2.5"

    def test_constructor_uses_config(self, monkeypatch):
        from answerbot.llm import CursorLLM

        monkeypatch.setattr(config, "CURSOR_API_KEY", "crsr_env")
        monkeypatch.setattr(config, "ANSWER_MODEL", "composer-2.5")
        llm = CursorLLM()
        assert llm.api_key == "crsr_env"
        assert llm.model == "composer-2.5"

    def test_complete_concatenates_and_disables_tools(self, monkeypatch):
        from pathlib import Path

        from answerbot.llm import CursorLLM

        captured = {}

        class FakeResult:
            status = "finished"
            result = "  hi  "
            id = "run-1"

        def prompt(message, options):
            captured["message"] = message
            captured["options"] = options
            assert Path(options.local.cwd).is_dir()
            return FakeResult()

        self._install_sdk(monkeypatch, prompt)
        llm = CursorLLM(api_key="crsr_x", model="composer-2.5")
        assert llm.complete("  sys  ", "  usr  ") == "hi"
        assert captured["message"] == "sys\n\nusr"
        assert captured["options"].api_key == "crsr_x"
        assert captured["options"].model == "composer-2.5"
        assert captured["options"].tools == []

    def test_explicit_cwd(self, monkeypatch, tmp_path):
        from answerbot.llm import CursorLLM

        captured = {}

        class FakeResult:
            status = "finished"
            result = "ok"
            id = "run-1"

        def prompt(message, options):
            captured["cwd"] = options.local.cwd
            return FakeResult()

        self._install_sdk(monkeypatch, prompt)
        llm = CursorLLM(api_key="k", model="composer-2.5", cwd=str(tmp_path))
        assert llm.complete("s", "u") == "ok"
        assert captured["cwd"] == str(tmp_path)

    def test_missing_key_raises(self):
        from answerbot.llm import CursorLLM

        with pytest.raises(RuntimeError, match="API key is not set"):
            CursorLLM(api_key="", model="composer-2.5").complete("s", "u")

    def test_missing_sdk_raises(self, monkeypatch):
        import sys

        from answerbot.llm import CursorLLM, _cursor_sdk

        monkeypatch.setitem(sys.modules, "cursor_sdk", None)
        with pytest.raises(RuntimeError, match="cursor-sdk"):
            _cursor_sdk()
        with pytest.raises(RuntimeError, match="cursor-sdk"):
            CursorLLM(api_key="k", model="composer-2.5").complete("s", "u")

    def test_run_error_raises(self, monkeypatch):
        from answerbot.llm import CursorLLM

        class FakeResult:
            status = "error"
            result = ""
            id = "run-9"

        self._install_sdk(monkeypatch, lambda message, options: FakeResult())
        with pytest.raises(RuntimeError, match="Cursor run error \\(run-9\\)"):
            CursorLLM(api_key="k", model="composer-2.5", cwd=".").complete("s", "u")

    def test_empty_response_raises(self, monkeypatch):
        from answerbot.llm import CursorLLM

        class FakeResult:
            status = "finished"
            result = "  "
            id = "run-1"

        self._install_sdk(monkeypatch, lambda message, options: FakeResult())
        with pytest.raises(RuntimeError, match="no text"):
            CursorLLM(api_key="k", model="composer-2.5", cwd=".").complete("s", "u")

    def test_startup_error_is_wrapped(self, monkeypatch):
        from answerbot.llm import CursorLLM

        box = {}

        def boom(message, options):
            raise box["err"]("invalid key")

        box["err"] = self._install_sdk(monkeypatch, boom)
        with pytest.raises(RuntimeError, match="Cursor: invalid key"):
            CursorLLM(api_key="k", model="composer-2.5", cwd=".").complete("s", "u")

    def test_get_llm_dispatches(self, monkeypatch):
        from answerbot import llm as llm_mod

        monkeypatch.setattr(config, "LLM_PROVIDER", "cursor")
        monkeypatch.setattr(llm_mod, "CursorLLM", lambda: "cursor-llm")
        assert llm_mod.get_llm() == "cursor-llm"


class TestOpenAICompat:
    def _complete(self, llm, body, monkeypatch, raw=None):
        from answerbot import llm as llm_mod

        class FakeResp:
            def read(self):
                if raw is not None:
                    return raw
                return json.dumps(body).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        captured = {}

        def fake_open(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = {k: v for k, v in req.header_items()}
            captured["payload"] = json.loads(req.data.decode())
            captured["timeout"] = timeout
            return FakeResp()

        monkeypatch.setattr(llm_mod, "_http_open", fake_open)
        text = llm.complete("sys", "usr")
        return text, captured

    def test_default_answer_models(self):
        assert config.DEFAULT_ANSWER_MODELS["groq"] == "openai/gpt-oss-20b"
        assert config.DEFAULT_ANSWER_MODELS["openrouter"] == "openai/gpt-oss-20b:free"

    def test_constructors_use_config(self, monkeypatch):
        from answerbot.llm import GroqLLM, OpenRouterLLM

        monkeypatch.setattr(config, "GROQ_API_KEY", "gsk-env")
        monkeypatch.setattr(config, "OPENROUTER_API_KEY", "or-env")
        monkeypatch.setattr(config, "ANSWER_MODEL", "openai/gpt-oss-20b")
        monkeypatch.setattr(config, "ANSWER_MAX_REQUEST_TOKENS", 0)
        monkeypatch.setattr(config, "OPENROUTER_HTTP_REFERER", "")
        monkeypatch.setattr(config, "OPENROUTER_APP_TITLE", "answer-chat-history-bot")

        groq = GroqLLM()
        assert groq.api_key == "gsk-env"
        assert groq.model == "openai/gpt-oss-20b"
        assert groq.base_url == "https://api.groq.com/openai/v1"
        assert groq.max_request_tokens == 8000

        router = OpenRouterLLM()
        assert router.api_key == "or-env"
        assert router.base_url == "https://openrouter.ai/api/v1"
        assert router.extra_headers == {"X-Title": "answer-chat-history-bot"}
        assert router.max_request_tokens == 0

    def test_groq_complete(self, monkeypatch):
        from answerbot.llm import GroqLLM

        llm = GroqLLM(api_key="gsk", model="openai/gpt-oss-20b")
        text, cap = self._complete(
            llm,
            {"choices": [{"message": {"content": "  hi  "}}]},
            monkeypatch,
        )
        assert text == "hi"
        assert cap["url"] == "https://api.groq.com/openai/v1/chat/completions"
        headers = {k.lower(): v for k, v in cap["headers"].items()}
        assert headers["authorization"] == "Bearer gsk"
        assert headers["user-agent"] == "answer-chat-history-bot/0.1.0"
        assert cap["payload"]["messages"][0] == {"role": "system", "content": "sys"}
        from answerbot.llm import _completion_tokens

        expected = _completion_tokens(
            "sys", "usr", config.ANSWER_MAX_TOKENS, llm.max_request_tokens
        )
        assert cap["payload"]["max_tokens"] == expected
        assert cap["payload"]["max_completion_tokens"] == expected
        assert cap["timeout"] == config.LLM_TIMEOUT

    def test_groq_caps_completion_to_tpm_budget(self, monkeypatch):
        from answerbot.llm import GroqLLM, _CHAT_OVERHEAD_TOKENS, _estimate_tokens

        monkeypatch.setattr(config, "ANSWER_MAX_TOKENS", 8192)
        monkeypatch.setattr(config, "ANSWER_MAX_REQUEST_TOKENS", 0)
        llm = GroqLLM(api_key="gsk", model="openai/gpt-oss-20b")
        _, cap = self._complete(
            llm,
            {"choices": [{"message": {"content": "ok"}}]},
            monkeypatch,
        )
        prompt = _estimate_tokens("sys") + _estimate_tokens("usr") + _CHAT_OVERHEAD_TOKENS
        assert cap["payload"]["max_tokens"] == 2048
        assert cap["payload"]["max_tokens"] + prompt < 8000
        assert cap["payload"]["max_tokens"] < 8192

    def test_groq_prompt_over_budget_raises(self, monkeypatch):
        from answerbot.llm import GroqLLM

        monkeypatch.setattr(config, "ANSWER_MAX_REQUEST_TOKENS", 0)
        llm = GroqLLM(api_key="k", model="x")
        with pytest.raises(RuntimeError, match="too large for the request budget"):
            llm.complete("s", "x" * 100_000)

    def test_openrouter_headers_and_parts(self, monkeypatch):
        from answerbot.llm import OpenRouterLLM

        monkeypatch.setattr(config, "OPENROUTER_HTTP_REFERER", "https://example.test")
        monkeypatch.setattr(config, "OPENROUTER_APP_TITLE", "answer-chat-history-bot")
        llm = OpenRouterLLM(api_key="sk-or", model="openai/gpt-oss-20b:free")
        text, cap = self._complete(
            llm,
            {"choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]},
            monkeypatch,
        )
        assert text == "ok"
        assert cap["url"] == "https://openrouter.ai/api/v1/chat/completions"
        headers = {k.lower(): v for k, v in cap["headers"].items()}
        assert headers["http-referer"] == "https://example.test"
        assert headers["x-title"] == "answer-chat-history-bot"
        assert headers["authorization"] == "Bearer sk-or"
        assert headers["user-agent"] == "answer-chat-history-bot/0.1.0"
        assert cap["payload"]["max_tokens"] == config.ANSWER_MAX_TOKENS

    def test_empty_response_raises(self, monkeypatch):
        from answerbot.llm import GroqLLM

        with pytest.raises(RuntimeError, match="no text"):
            self._complete(
                GroqLLM(api_key="k", model="x"),
                {"choices": [{"message": {"content": "  "}}]},
                monkeypatch,
            )

    def test_length_finish_reason_raises(self, monkeypatch):
        from answerbot.llm import GroqLLM

        with pytest.raises(RuntimeError, match="token limit"):
            self._complete(
                GroqLLM(api_key="k", model="x"),
                {
                    "choices": [
                        {
                            "message": {"content": ""},
                            "finish_reason": "length",
                        }
                    ]
                },
                monkeypatch,
            )

    def test_json_error_body_is_wrapped(self, monkeypatch):
        from answerbot.llm import GroqLLM

        with pytest.raises(RuntimeError, match="returned an error: nope"):
            self._complete(
                GroqLLM(api_key="k", model="x"),
                {"error": {"message": "nope"}},
                monkeypatch,
            )

    def test_invalid_json_is_wrapped(self, monkeypatch):
        from answerbot.llm import GroqLLM

        with pytest.raises(RuntimeError, match="invalid JSON"):
            self._complete(
                GroqLLM(api_key="k", model="x"),
                {},
                monkeypatch,
                raw=b"not-json",
            )

    def test_http_error_uses_json_message(self, monkeypatch):
        from answerbot import llm as llm_mod
        from answerbot.llm import GroqLLM

        def boom(*a, **k):
            from email.message import Message
            from io import BytesIO

            raise urllib.error.HTTPError(
                "https://api.groq.com/openai/v1/chat/completions",
                429,
                "Too Many Requests",
                hdrs=Message(),
                fp=BytesIO(b'{"error":{"message":"rate"}}'),
            )

        monkeypatch.setattr(llm_mod, "_http_open", boom)
        with pytest.raises(RuntimeError, match="Groq HTTP 429: rate"):
            GroqLLM(api_key="k", model="x").complete("s", "u")

    def test_groq_retries_429_after_suggested_wait(self, monkeypatch):
        from email.message import Message
        from io import BytesIO

        from answerbot import llm as llm_mod
        from answerbot.llm import GroqLLM

        calls = {"n": 0}
        slept = []

        class FakeResp:
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_open(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.HTTPError(
                    "https://api.groq.com/openai/v1/chat/completions",
                    429,
                    "Too Many Requests",
                    hdrs=Message(),
                    fp=BytesIO(
                        b'{"error":{"message":"Rate limit reached. Please try again in 7.85s."}}'
                    ),
                )
            return FakeResp()

        monkeypatch.setattr(llm_mod, "_http_open", fake_open)
        monkeypatch.setattr(llm_mod.time, "sleep", slept.append)
        assert GroqLLM(api_key="k", model="x").complete("s", "u") == "ok"
        assert calls["n"] == 2
        assert len(slept) == 1
        assert slept[0] == pytest.approx(8.1, abs=0.05)

    def test_groq_does_not_retry_long_429_wait(self, monkeypatch):
        from email.message import Message
        from io import BytesIO

        from answerbot import llm as llm_mod
        from answerbot.llm import GroqLLM

        def boom(*a, **k):
            raise urllib.error.HTTPError(
                "https://api.groq.com/openai/v1/chat/completions",
                429,
                "Too Many Requests",
                hdrs=Message(),
                fp=BytesIO(
                    b'{"error":{"message":"Rate limit reached. Please try again in 45s."}}'
                ),
            )

        monkeypatch.setattr(llm_mod, "_http_open", boom)
        monkeypatch.setattr(llm_mod.time, "sleep", lambda s: pytest.fail("slept"))
        with pytest.raises(RuntimeError, match="Groq HTTP 429:"):
            GroqLLM(api_key="k", model="x").complete("s", "u")

    def test_http_error_does_not_dump_raw_body(self, monkeypatch):
        from answerbot import llm as llm_mod
        from answerbot.llm import GroqLLM

        def boom(*a, **k):
            from email.message import Message
            from io import BytesIO

            raise urllib.error.HTTPError(
                "https://api.groq.com/openai/v1/chat/completions",
                500,
                "Internal Server Error",
                hdrs=Message(),
                fp=BytesIO(b"<html>trace</html>"),
            )

        monkeypatch.setattr(llm_mod, "_http_open", boom)
        with pytest.raises(RuntimeError, match="Groq HTTP 500: Internal Server Error") as ei:
            GroqLLM(api_key="k", model="x").complete("s", "u")
        assert "trace" not in str(ei.value)
        assert "<html>" not in str(ei.value)

    def test_url_error_is_wrapped(self, monkeypatch):
        from answerbot import llm as llm_mod
        from answerbot.llm import GroqLLM

        def boom(*a, **k):
            raise urllib.error.URLError("refused")

        monkeypatch.setattr(llm_mod, "_http_open", boom)
        with pytest.raises(RuntimeError, match="Groq at https://api.groq.com/openai/v1 failed"):
            GroqLLM(api_key="k", model="x").complete("s", "u")

    def test_timeout_is_wrapped(self, monkeypatch):
        from answerbot import llm as llm_mod
        from answerbot.llm import GroqLLM

        def boom(*a, **k):
            raise TimeoutError()

        monkeypatch.setattr(llm_mod, "_http_open", boom)
        with pytest.raises(RuntimeError, match="timed out after"):
            GroqLLM(api_key="k", model="x").complete("s", "u")

    def test_missing_key_raises(self):
        from answerbot.llm import GroqLLM, OpenRouterLLM

        with pytest.raises(RuntimeError, match="API key is not set"):
            GroqLLM(api_key="", model="x").complete("s", "u")
        with pytest.raises(RuntimeError, match="API key is not set"):
            OpenRouterLLM(api_key="", model="x").complete("s", "u")

    def test_redirects_are_not_followed(self):
        from answerbot.llm import _NoRedirectHandler

        handler = _NoRedirectHandler()
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions")
        assert (
            handler.redirect_request(
                req, None, 302, "Found", {}, "https://evil.example/steal"
            )
            is None
        )

    def test_get_llm_dispatches(self, monkeypatch):
        from answerbot import llm as llm_mod

        monkeypatch.setattr(config, "LLM_PROVIDER", "groq")
        monkeypatch.setattr(llm_mod, "GroqLLM", lambda: "groq-llm")
        assert llm_mod.get_llm() == "groq-llm"
        monkeypatch.setattr(config, "LLM_PROVIDER", "openrouter")
        monkeypatch.setattr(llm_mod, "OpenRouterLLM", lambda: "or-llm")
        assert llm_mod.get_llm() == "or-llm"


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


def _drop_log_handlers():
    from answerbot import logconfig

    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, logconfig._STREAM_MARK, False) or getattr(h, logconfig._FILE_MARK, None):
            h.close()
            root.removeHandler(h)


class TestLogconfig:
    def test_writes_timestamped_file(self, tmp_path, monkeypatch):
        from answerbot import logconfig

        path = tmp_path / "answerbot.log"
        monkeypatch.setattr(config, "LOG_PATH", path)
        monkeypatch.setattr(config, "LOG_LEVEL", "INFO")
        _drop_log_handlers()
        try:
            logconfig.setup()
            logging.getLogger("answerbot").error("disk full")
            text = path.read_text(encoding="utf-8")
        finally:
            _drop_log_handlers()
        assert "ERROR answerbot: disk full" in text
        assert text.split(" ", 1)[0].count("-") == 2  # YYYY-MM-DD

    def test_setup_is_idempotent(self, tmp_path, monkeypatch):
        from answerbot import logconfig

        path = tmp_path / "answerbot.log"
        monkeypatch.setattr(config, "LOG_PATH", path)
        _drop_log_handlers()
        try:
            logconfig.setup()
            logconfig.setup()
            files = [
                h for h in logging.getLogger().handlers
                if getattr(h, logconfig._FILE_MARK, None) == str(path)
            ]
        finally:
            _drop_log_handlers()
        assert len(files) == 1

    def test_off_skips_the_file(self, tmp_path, monkeypatch):
        from answerbot import logconfig

        monkeypatch.setattr(config, "LOG_PATH", None)
        _drop_log_handlers()
        try:
            logconfig.setup()
            n_files = sum(1 for h in logging.getLogger().handlers if getattr(h, logconfig._FILE_MARK, None))
        finally:
            _drop_log_handlers()
        assert n_files == 0

    def test_asyncio_handler_logs_exception(self, caplog):
        from answerbot import logconfig

        caplog.set_level(logging.ERROR, logger="answerbot")

        try:
            raise RuntimeError("loop boom")
        except RuntimeError as exc:
            logconfig.asyncio_handler(None, {"message": "Task exception", "exception": exc})
        assert "Task exception" in caplog.text
        assert "loop boom" in caplog.text


class TestEmbedToken:
    def teardown_method(self):
        from answerbot import embed

        embed._model = None

    def _stub_sentence_transformer(self, monkeypatch, captured):
        import types

        torch = types.ModuleType("torch")
        torch.set_num_threads = lambda n: None
        torch.set_num_interop_threads = lambda n: None

        st = types.ModuleType("sentence_transformers")

        class SentenceTransformer:
            def __init__(self, name, token=None):
                captured["name"] = name
                captured["token"] = token

        st.SentenceTransformer = SentenceTransformer
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "sentence_transformers", st)

    def test_passes_hf_token(self, monkeypatch):
        from answerbot import embed

        captured = {}
        self._stub_sentence_transformer(monkeypatch, captured)
        monkeypatch.setattr(config, "HF_TOKEN", "hf_secret")
        monkeypatch.setattr(config, "EMBED_MODEL", "org/gated-model")
        embed._model = None

        model = embed._get_model()
        assert captured == {"name": "org/gated-model", "token": "hf_secret"}
        assert model is embed._model

    def test_loads_without_token(self, monkeypatch):
        from answerbot import embed

        captured = {}
        self._stub_sentence_transformer(monkeypatch, captured)
        monkeypatch.setattr(config, "HF_TOKEN", None)
        monkeypatch.setattr(config, "EMBED_MODEL", "intfloat/multilingual-e5-small")
        embed._model = None

        embed._get_model()
        assert captured == {
            "name": "intfloat/multilingual-e5-small",
            "token": None,
        }
