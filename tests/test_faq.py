"""FAQ wiki ingest, chunking, retrieval, and answer citations."""

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from answerbot import answer, config, db, faq, index, retrieve
from answerbot.retrieve import Hit


BYT = """# Выбрасывание мусора

## Общая информация
Мусор сортируется. PET принимается в магазинах.

### Zürich
Календарь выброса мусора: stadt-zuerich.ch/entsorgungskalender

# Страховки
Страховка квартиры обойдется примерно в 400 CHF.
"""

VISA = """# Визы

Для въезда нужна виза D или разрешение.
"""


@pytest.fixture
def conn():
    retrieve.invalidate_cache()
    c = db.connect(":memory:")
    yield c
    c.close()
    retrieve.invalidate_cache()


@pytest.fixture
def fake_embed(monkeypatch):
    calls: list[list[str]] = []

    def fake(texts, batch_size=64, progress=False, **_k):
        calls.append(list(texts))
        return np.zeros((len(texts), config.EMBED_DIM), dtype=np.float32)

    monkeypatch.setattr("answerbot.embed.encode_passages", fake)
    monkeypatch.setattr(
        "answerbot.embed.encode_query",
        lambda q: np.zeros(config.EMBED_DIM, dtype=np.float32),
    )
    return calls


@pytest.fixture
def faq_on(monkeypatch):
    monkeypatch.setattr(config, "FAQ_ENABLED", True)
    monkeypatch.setattr(config, "FAQ_COSINE_MIN", 0.0)
    monkeypatch.setattr(config, "FAQ_TOP_K", 3)
    monkeypatch.setattr(config, "FAQ_SITE", "https://ru-ch.github.io/faq")


def _index(conn, files=None):
    files = files or {"docs/Быт.md": BYT, "inbox/Визы.md": VISA}
    return faq.index_markdown(conn, files)


class TestChunkMarkdown:
    def test_heading_trail_includes_parents(self):
        chunks = faq.chunk_markdown(BYT, title="Быт")
        headings = [c.heading for c in chunks]
        assert any("Zürich" in h for h in headings)
        zurich = next(c for c in chunks if "Zürich" in c.heading)
        assert "Выбрасывание мусора" in zurich.heading
        assert "Выбрасывание мусора" in zurich.text
        assert "stadt-zuerich" in zurich.text

    def test_splits_on_h1(self):
        chunks = faq.chunk_markdown(BYT, title="Быт")
        assert any("Страховки" in c.heading and "400 CHF" in c.text for c in chunks)

    def test_caps_long_section(self):
        body = "word " * 400
        md = f"# Title\n\n{body}"
        chunks = faq.chunk_markdown(md, title="Doc", max_chars=80)
        assert len(chunks) > 1
        assert all(len(c.text) <= 80 for c in chunks)


class TestPageUrl:
    def test_docs_md_to_html(self):
        assert faq.page_url("docs/Быт.md") == "https://ru-ch.github.io/faq/docs/Быт.html"

    def test_inbox(self):
        assert (
            faq.page_url("inbox/Визы.md", site="https://ru-ch.github.io/faq/")
            == "https://ru-ch.github.io/faq/inbox/Визы.html"
        )


class TestIndexAndSearch:
    def test_fts_finds_the_section(self, conn, fake_embed, faq_on):
        _index(conn)
        hits = faq.search(conn, "entsorgungskalender")
        assert hits
        assert hits[0].kind == "faq"
        assert "Zürich" in hits[0].speakers or "entsorgungskalender" in hits[0].text.lower()
        assert hits[0].link() == "https://ru-ch.github.io/faq/docs/Быт.html"

    def test_disabled_returns_nothing(self, conn, fake_embed, monkeypatch):
        monkeypatch.setattr(config, "FAQ_ENABLED", False)
        _index(conn)
        assert faq.search(conn, "мусор") == []

    def test_unchanged_sha_does_not_reembed(self, conn, fake_embed, faq_on):
        first = _index(conn)
        assert first["embedded"] > 0
        fake_embed.clear()
        second = faq.index_markdown(
            conn, {"docs/Быт.md": BYT, "inbox/Визы.md": VISA}
        )
        assert second["changed"] == 0
        assert second["embedded"] == 0
        assert fake_embed == []

    def test_update_skips_when_disabled(self, conn, monkeypatch):
        monkeypatch.setattr(config, "FAQ_ENABLED", False)
        result = faq.update(conn)
        assert result.get("skipped")


class TestRetrieveMerge:
    def test_faq_hits_come_first(self, conn, fake_embed, faq_on):
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) "
            "VALUES (1, 1, 1, 'A', 'wifi password is hunter2')"
        )
        conn.commit()
        index.reindex(conn, progress=False)
        _index(conn)
        hits = retrieve.search(conn, "мусор Zürich", chat_id=1)
        assert hits
        assert hits[0].kind == "faq"

    def test_time_range_skips_faq(self, conn, fake_embed, faq_on):
        ts = 1_700_000_000
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) "
            "VALUES (1, 1, ?, 'A', 'мусор выбрасывают по четвергам')",
            (ts,),
        )
        conn.commit()
        index.reindex(conn, progress=False)
        _index(conn)
        hits = retrieve.search(
            conn,
            "мусор yesterday",
            chat_id=1,
            now=datetime.fromtimestamp(ts, tz=timezone.utc),
        )
        assert all(h.kind != "faq" for h in hits)

    def test_empty_allow_list_skips_faq_too(self, conn, fake_embed, faq_on):
        _index(conn)
        assert retrieve.search(conn, "мусор", chat_id=[]) == []


class TestAnswerCitations:
    def test_context_header_and_html_link(self, faq_on):
        hit = Hit(
            1,
            0,
            0,
            0,
            0,
            0,
            "Быт · Zürich",
            "календарь",
            1.0,
            kind="faq",
            url="https://ru-ch.github.io/faq/docs/Быт.html",
        )
        ctx = answer.build_context([hit])
        assert "[W1] FAQ, Быт · Zürich:" in ctx
        body = answer.format_answer_body(answer.Answer("see [W1]", [hit]))
        assert "ru-ch.github.io/faq/docs" in body
        assert answer.jump_i18n_key(hit) == "go_to_faq"

    def test_complete_answer_cites_faq(self, faq_on):
        class FakeLLM:
            def complete(self, system, user):
                assert "FAQ" in user
                return "Сортируйте мусор [W1]"

        hit = Hit(
            7,
            0,
            0,
            0,
            0,
            0,
            "Быт · Zürich",
            "PET в магазинах",
            1.0,
            kind="faq",
            url="https://ru-ch.github.io/faq/docs/Быт.html",
        )
        result = answer.complete_answer("куда сдавать PET", [hit], FakeLLM())
        assert result.cited_hits()[0].kind == "faq"
        assert result.primary_link().startswith("https://ru-ch.github.io/")

    def test_query_log_uses_negative_chunk_ids(
        self, conn, fake_embed, faq_on, monkeypatch
    ):
        monkeypatch.setattr(config, "QUERY_LOG", True)

        class FakeLLM:
            def complete(self, system, user):
                return "found it [W1]"

        _index(conn)
        result = answer.answer(
            conn, "мусор Zürich", chat_id=1, llm=FakeLLM(), flush=False
        )
        assert result.hits
        assert result.hits[0].kind == "faq"
        row = conn.execute(
            "SELECT window_ids, cited_ids FROM query_log"
        ).fetchone()
        window_ids = json.loads(row[0])
        cited_ids = json.loads(row[1])
        assert window_ids[0] < 0
        assert cited_ids[0] < 0

    def test_prompt_mentions_faq(self):
        assert "FAQ" in answer.SYSTEM
        assert "more than one chat" in answer.SYSTEM

    def test_sources_html_skips_chat_label_for_faq(self, faq_on):
        hit = Hit(
            1,
            0,
            0,
            0,
            0,
            0,
            "Быт · Zürich",
            "body",
            1.0,
            kind="faq",
            url="https://ru-ch.github.io/faq/docs/Быт.html",
        )
        html = answer.format_sources_html(
            answer.Answer("see [W1]", [hit]),
            chat_titles={0: "Main"},
            include_chat=True,
        )
        assert "Main · " not in html
        assert "FAQ" in html


class TestToyVectors:
    def test_cosine_ranks_the_matching_section(self, conn, faq_on, monkeypatch):
        dim = config.EMBED_DIM

        def pack_text(text: str) -> np.ndarray:
            v = np.zeros(dim, np.float32)
            t = text.lower()
            if "visa" in t or "виз" in t:
                v[0] = 1
            elif "страхов" in t or "chf" in t:
                v[1] = 1
            else:
                v[2] = 1
            n = float(np.linalg.norm(v))
            if n:
                v /= n
            return v

        monkeypatch.setattr(
            "answerbot.embed.encode_passages",
            lambda texts, batch_size=64, progress=False, **k: np.stack(
                [pack_text(t) for t in texts]
            ),
        )
        monkeypatch.setattr("answerbot.embed.encode_query", lambda q: pack_text(q))
        _index(conn)
        hits = faq.search(conn, "виза для въезда")
        assert hits
        assert "Визы" in hits[0].speakers or "виз" in hits[0].text.lower()
