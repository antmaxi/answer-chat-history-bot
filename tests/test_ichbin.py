"""#ichbin profiles and similar-people ranking. Embeddings are faked."""

from types import SimpleNamespace

import numpy as np
import pytest

from answerbot import config, db, ichbin


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", None)
    monkeypatch.setattr(config, "ICHBIN_COSINE_MIN", 0.5)
    monkeypatch.setattr(config, "ICHBIN_TOP_K", 5)
    monkeypatch.setattr(config, "ICHBIN_MAX_CHARS", 1500)
    c = db.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def fake_embed(monkeypatch):
    """Unit vectors from keywords so cosine is predictable without the model."""

    def fake(texts, batch_size=64, progress=False, **kwargs):
        rows = []
        for text in texts:
            vector = np.zeros(config.EMBED_DIM, np.float32)
            low = text.lower()
            if "climb" in low and "cook" in low:
                vector[0] = 0.6
                vector[1] = 0.8
            elif "climb" in low:
                vector[0] = 1.0
            elif "cook" in low:
                vector[1] = 1.0
            else:
                vector[2] = 1.0
            vector /= np.linalg.norm(vector)
            rows.append(vector)
        if not rows:
            return np.zeros((0, config.EMBED_DIM), np.float32)
        return np.stack(rows)

    monkeypatch.setattr("answerbot.embed.encode_passages", fake)
    return fake


def add(conn, sender_id, msg_id, text, *, ts=None, chat_id=1, sender="Ann"):
    conn.execute(
        """INSERT INTO messages (chat_id, msg_id, sender_id, sender, ts, text)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (chat_id, msg_id, sender_id, sender, msg_id if ts is None else ts, text),
    )
    conn.commit()


def person(conn, sender_id, display, username=None, updated_at=1):
    conn.execute(
        """INSERT INTO people (sender_id, display_name, username, source, updated_at)
           VALUES (?, ?, ?, 'manual', ?)""",
        (sender_id, display, username, updated_at),
    )
    conn.commit()


class TestTag:
    def test_token_boundaries(self):
        assert ichbin.has_tag("#ichbin")
        assert ichbin.has_tag("#IchBin")
        assert ichbin.has_tag("#ICHBIN")
        assert ichbin.has_tag("see #ichbin, please")
        assert ichbin.has_tag("привет #ichbin я")
        assert not ichbin.has_tag("#ichbinhier")
        assert not ichbin.has_tag("my#ichbin")
        assert not ichbin.has_tag("ichbin")
        assert not ichbin.has_tag("")


class TestCapAndLink:
    def test_cap_keeps_the_latest_message_when_it_fits(self):
        older = "OLD #ichbin ignore-me"
        newer = "NEW #ichbin latest-tail!!"
        joined = older + "\n" + newer
        assert ichbin.cap_tail(joined, 30) == newer

    def test_cap_keeps_a_raw_tail_of_one_long_message(self):
        text = "X" * 10 + " #ichbin " + "Y" * 40
        assert ichbin.cap_tail(text, 15) == text[-15:]

    def test_message_link_strips_supergroup_prefix(self):
        assert ichbin.message_link(-1001234567890, 42) == "https://t.me/c/1234567890/42"
        assert ichbin.message_link(7, 3) == "https://t.me/c/7/3"


class TestForwardAndUsername:
    def test_forward_origin(self):
        assert ichbin.forward_sender(None) == ("none", None)
        assert ichbin.forward_sender(SimpleNamespace(type="hidden_user")) == ("hidden", None)
        assert ichbin.forward_sender(
            SimpleNamespace(type="user", sender_user=SimpleNamespace(id=9))
        ) == ("user", 9)
        assert ichbin.forward_sender(SimpleNamespace(type="user", sender_user=None)) == (
            "other",
            None,
        )
        assert ichbin.forward_sender(SimpleNamespace(type="channel")) == ("other", None)

        class MessageOriginUser:
            sender_user = SimpleNamespace(id=4)

        assert ichbin.forward_sender(MessageOriginUser()) == ("user", 4)

    def test_parse_username(self):
        assert ichbin.parse_username("@AnnaZ") == "annaz"
        assert ichbin.parse_username("annaz extra") == "annaz"
        assert ichbin.parse_username("no") is None
        assert ichbin.parse_username("@ab") is None
        assert ichbin.parse_username("1badname") is None

    def test_resolve_is_case_insensitive(self, conn):
        person(conn, 5, "Anna", "AnnaZ")
        assert ichbin.resolve_username(conn, "@annaz") == ("ok", 5, "annaz")
        assert ichbin.resolve_username(conn, "missinguser") == ("unknown", None, "missinguser")
        assert ichbin.resolve_username(conn, "no") == ("bad", None, None)

    def test_duplicate_username_prefers_a_profile(self, conn, fake_embed):
        person(conn, 10, "Newer", "shared", updated_at=50)
        person(conn, 11, "Has intro", "Shared", updated_at=1)
        add(conn, 11, 1, "I climb #ichbin", sender="Has intro")
        ichbin.sync(conn, all_chats=True)
        assert ichbin.sender_id_for_username(conn, "@shared") == 11


class TestSync:
    def test_joins_messages_and_links_the_newest(self, conn, fake_embed):
        add(conn, 1, 1, "alpha #ichbin early", ts=10, chat_id=-10055, sender="Ann")
        add(conn, 1, 2, "beta #ichbin late", ts=20, chat_id=-10055, sender="Ann")
        add(conn, 2, 3, "other #ichbin", ts=15, sender="Bea")
        add(conn, None, 4, "anon #ichbin", sender="Ghost")
        add(conn, 3, 5, "nope #ichbinhier", sender="Cee")
        ichbin.sync(conn, all_chats=True)
        row = conn.execute(
            "SELECT text, msg_count, chat_id, msg_id FROM ichbin_profiles WHERE sender_id=1"
        ).fetchone()
        assert row["text"] == "alpha #ichbin early\nbeta #ichbin late"
        assert row["msg_count"] == 2
        assert row["chat_id"] == -10055
        assert row["msg_id"] == 2
        assert ichbin.has_profile(conn, 2)
        assert not ichbin.has_profile(conn, 3)
        ids = {row[0] for row in conn.execute("SELECT sender_id FROM ichbin_profiles")}
        assert ids == {1, 2}

    def test_cap_stores_the_latest_tail(self, conn, fake_embed, monkeypatch):
        monkeypatch.setattr(config, "ICHBIN_MAX_CHARS", 30)
        older = "OLD #ichbin ignore-me"
        newer = "NEW #ichbin latest-tail!!"
        add(conn, 1, 1, older, ts=1)
        add(conn, 1, 2, newer, ts=2)
        ichbin.sync(conn, all_chats=True)
        text = conn.execute(
            "SELECT text FROM ichbin_profiles WHERE sender_id=1"
        ).fetchone()["text"]
        assert text == newer

    def test_unchanged_text_is_not_reembedded(self, conn, monkeypatch):
        calls = []

        def fake(texts, batch_size=64, progress=False, **kwargs):
            calls.append(list(texts))
            return np.ones((len(texts), config.EMBED_DIM), np.float32)

        monkeypatch.setattr("answerbot.embed.encode_passages", fake)
        add(conn, 1, 5, "I climb #ichbin", ts=5)
        ichbin.sync(conn, all_chats=True)
        ichbin.sync(conn, all_chats=True)
        assert len(calls) == 1
        conn.execute("UPDATE ichbin_profiles SET msg_id=1 WHERE sender_id=1")
        conn.commit()
        ichbin.sync(conn, all_chats=True)
        assert len(calls) == 1
        assert conn.execute(
            "SELECT msg_id FROM ichbin_profiles WHERE sender_id=1"
        ).fetchone()["msg_id"] == 5

    def test_profile_is_removed_when_the_tag_is_edited_away(self, conn, fake_embed):
        add(conn, 1, 1, "I climb #ichbin")
        ichbin.sync(conn, all_chats=True)
        assert ichbin.has_profile(conn, 1)
        conn.execute("UPDATE messages SET text=? WHERE msg_id=1", ("hello",))
        conn.commit()
        ichbin.sync(conn, all_chats=True)
        assert not ichbin.has_profile(conn, 1)
        assert conn.execute("SELECT count(*) FROM ichbin_vecs").fetchone()[0] == 0

    def test_main_chat_leaves_other_chats_alone(self, conn, fake_embed, monkeypatch):
        monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", 1)
        add(conn, 1, 1, "I climb #ichbin", chat_id=1)
        add(conn, 2, 2, "I cook #ichbin", chat_id=2)
        ichbin.sync(conn)
        assert ichbin.has_profile(conn, 1)
        assert not ichbin.has_profile(conn, 2)
        ichbin.sync(conn, all_chats=True)
        assert ichbin.has_profile(conn, 2)
        conn.execute("UPDATE messages SET text=? WHERE sender_id=1", ("no tag",))
        conn.commit()
        ichbin.sync(conn)
        assert not ichbin.has_profile(conn, 1)
        assert ichbin.has_profile(conn, 2)


class TestRecommend:
    def _seed(self, conn):
        person(conn, 1, "Alice", "alice")
        person(conn, 2, "Bruno", "bruno")
        person(conn, 3, "Cara", "cara")
        person(conn, 4, "Dana", "dana")
        add(conn, 1, 1, "I climb #ichbin mountains", chat_id=-1009, sender="Alice")
        add(conn, 2, 2, "I climb #ichbin too " + ("rock " * 40), sender="Bruno")
        add(conn, 3, 3, "I cook #ichbin dinner", sender="Cara")
        add(conn, 4, 4, "I climb and cook #ichbin", sender="Dana")
        ichbin.sync(conn, all_chats=True)

    def test_cosine_order_and_floor(self, conn, fake_embed, monkeypatch):
        self._seed(conn)
        rec = ichbin.recommend(conn, 1)
        assert rec is not None
        assert [n.card.sender_id for n in rec.neighbors] == [2, 4]
        assert rec.neighbors[0].cosine == pytest.approx(1.0)
        assert rec.neighbors[1].cosine == pytest.approx(0.6, abs=1e-5)
        monkeypatch.setattr(config, "ICHBIN_COSINE_MIN", 0.7)
        rec = ichbin.recommend(conn, 1)
        assert [n.card.sender_id for n in rec.neighbors] == [2]
        monkeypatch.setattr(config, "ICHBIN_COSINE_MIN", 0.5)
        monkeypatch.setattr(config, "ICHBIN_TOP_K", 1)
        rec = ichbin.recommend(conn, 1)
        assert [n.card.sender_id for n in rec.neighbors] == [2]

    def test_reply_leads_with_the_full_ichbin_text(self, conn, fake_embed):
        self._seed(conn)
        rec = ichbin.recommend(conn, 1)
        html = ichbin.format_similar("en", rec)
        body = rec.target.text
        assert body == "I climb #ichbin mountains"
        assert html.index(body) < html.index("Similar")
        assert html.index(body) < html.index("Bruno")
        assert "https://t.me/c/9/1" in html
        assert "@alice" in html
        assert rec.neighbors[0].card.text not in html
        assert "…" in html
        assert ichbin.recommend(conn, 99) is None

    def test_reply_escapes_html(self, conn, fake_embed):
        person(conn, 1, "A<b>", "ann")
        add(conn, 1, 1, "1 < 2 #ichbin")
        ichbin.sync(conn, all_chats=True)
        html = ichbin.format_similar("en", ichbin.recommend(conn, 1))
        assert "A&lt;b&gt;" in html
        assert "1 &lt; 2 #ichbin" in html
        assert "<b>" not in html

    def test_display_name_never_uses_the_raw_id(self, conn):
        add(conn, 8, 1, "plain", sender="Export Label")
        assert ichbin.display_name(conn, 8) == "Export Label"
        conn.execute("INSERT INTO aliases (sender_id, ordinal) VALUES (9, 3)")
        assert ichbin.display_name(conn, 9) == "User 3"
        assert ichbin.display_name(conn, 10) == "User"
        person(conn, 8, "Public", "pub")
        assert ichbin.display_name(conn, 8) == "Public"
