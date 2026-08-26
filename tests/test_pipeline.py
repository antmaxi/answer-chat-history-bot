"""Tests for the parts with real logic in them: parsing, windowing, FTS queries.

Deliberately no embedding calls — those need the model on disk and are slow.
Retrieval quality is checked by hand with `python -m answerbot.search`.
"""

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest

from answerbot import config, db, index, people, retrieve
from answerbot.timerange import TimeRange
from answerbot.answer import (
    Answer,
    SYSTEM,
    answer as run_answer,
    build_context,
    chat_label,
    complete_answer,
    format_answer_body,
    markdown_to_html,
)
from answerbot.index import build_windows, render
from answerbot.ingest import live
from answerbot.ingest.export import flatten_text, parse_sender_id, parse_ts
from answerbot.retrieve import Hit, fts_query, term_df_band


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
    monkeypatch.setattr(
        "answerbot.embed.encode_query",
        lambda q: np.zeros(config.EMBED_DIM, dtype=np.float32),
    )
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

    def test_supergroup_export_id_becomes_bot_api(self):
        from answerbot.ingest.export import bot_api_chat_id, bot_api_candidates, desktop_ids_for

        assert bot_api_chat_id({"id": 1495905530, "type": "private_supergroup"}) == -1001495905530
        assert bot_api_chat_id({"id": -1001495905530, "type": "private_supergroup"}) == -1001495905530
        assert bot_api_chat_id({"id": 555, "type": "private_group"}) == -555
        assert bot_api_chat_id({"id": 42, "type": "personal_chat"}) == 42
        assert desktop_ids_for(-1001495905530) == [1495905530]
        assert bot_api_candidates(1495905530) == [1495905530, -1001495905530, -1495905530]


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


class TestTermDfBand:
    def test_empty_corpus(self, conn):
        n, terms, match_count = term_df_band(conn, 0, 100)
        assert n == 0
        assert terms == []
        assert match_count == 0

    def test_band_by_percent(self, conn):
        # 20 messages: "the" in all (100%), "ok" in 6 (30%), "yeah" in 4 (20%),
        # "pangolin" in 1 (5%). Single-letter "a" must not appear.
        for i in range(20):
            parts = ["the", "a"]
            if i < 6:
                parts.append("ok")
            if i < 4:
                parts.append("yeah")
            if i == 0:
                parts.append("pangolin")
            conn.execute(
                "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1,?,?, 'A', ?)",
                (i, i, " ".join(parts)),
            )
        conn.commit()
        n, terms, match_count = term_df_band(conn, 15, 25)
        assert n == 20
        names = [t[0] for t in terms]
        assert names == ["yeah"]
        assert terms[0][1] == 4
        assert terms[0][2] == 20.0
        assert match_count == 1
        assert "a" not in names

        n, terms, _ = term_df_band(conn, 25, 100)
        names = [t[0] for t in terms]
        assert names == ["the", "ok"]
        assert terms[0][2] == 100.0
        assert terms[1][2] == 30.0

        n, terms, _ = term_df_band(conn, 0, 10)
        names = [t[0] for t in terms]
        assert names == ["pangolin"]

    def test_swaps_inverted_bounds(self, conn):
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1,1,1,'A','rareword')"
        )
        conn.commit()
        _, forward, _ = term_df_band(conn, 0, 100)
        _, backward, _ = term_df_band(conn, 100, 0)
        assert forward == backward

    def test_limit_caps_list_not_count(self, conn):
        for i in range(10):
            conn.execute(
                "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1,?,?, 'A', ?)",
                (i, i, f"token{i} shared"),
            )
        conn.commit()
        _, terms, match_count = term_df_band(conn, 0, 100, limit=3)
        assert len(terms) == 3
        assert match_count > 3
        assert terms[0][0] == "shared"


def hit(idx: int) -> Hit:
    return Hit(idx, 1, idx, idx, 0, 0, "Anna", f"body {idx}", 0.1)


class TestAnswerCitations:
    def test_cited_hits_are_only_referenced_windows(self):
        a = Answer("The cost was 200 [W2], see also [W3].", [hit(1), hit(2), hit(3)])
        assert [h.first_msg for h in a.cited_hits()] == [2, 3]

    def test_sources_block_lists_every_result_and_marks_cited(self):
        a = Answer("answer [W1] and a hallucinated [W9]", [hit(1), hit(2)])
        block = a.sources_block()
        assert "[W1] ✓" in block          # cited windows get a check
        assert "[W2]" in block            # every retrieved window is linked...
        assert "[W2] ✓" not in block      # ...but not marked cited
        assert "W9" not in block          # out-of-range citation is ignored
        assert block.count("https://t.me/c/") == 2  # a link per result

    def test_all_sources_flags_cited(self):
        a = Answer("see [W2]", [hit(1), hit(2), hit(3)])
        assert [(i, c) for i, _, c in a.all_sources()] == [(1, False), (2, True), (3, False)]

    def test_no_hits_means_no_sources(self):
        a = Answer("I couldn't find that in the chat history.", [])
        assert a.source_links() == []
        assert a.sources_block() == ""

    def test_every_source_carries_a_link(self):
        a = Answer("cost was 200 [W2]", [hit(1), hit(2), hit(3)])
        for _, h in a.source_links():
            assert h.link().startswith("https://t.me/c/")

    def test_primary_link_is_first_message_of_top_source(self):
        # cited: primary is the top cited window's first message
        a = Answer("see [W2] and [W3]", [hit(1), hit(2), hit(3)])
        assert a.primary_link() == hit(2).link()  # W2 is the first cited
        # uncited: falls back to the top retrieved window
        b = Answer("no citations here", [hit(5), hit(6)])
        assert b.primary_link() == hit(5).link()
        # nothing retrieved: no link
        assert Answer("I couldn't find that.", []).primary_link() is None

    def test_sources_join_windows_that_share_a_message_link(self):
        """Overlapping windows can start on the same Telegram message."""
        a = Answer(
            "see [W1] and [W2]",
            [hit(1), Hit(2, 1, 1, 3, 0, 0, "Anna", "body 2", 0.1), hit(3)],
        )
        groups = a.grouped_sources()
        assert [(idxs, h.first_msg, cited) for idxs, h, cited in groups] == [
            ([1, 2], 1, True),
            ([3], 3, False),
        ]
        block = a.sources_block()
        assert block.count("https://t.me/c/") == 2
        assert "[W1] [W2] ✓" in block
        assert "[W3]" in block
        assert "[W3] ✓" not in block


class TestAnswerMarkdown:
    def test_prompt_mentions_multiple_chats(self):
        assert "more than one chat" in SYSTEM

    def test_prompt_asks_for_markdown(self):
        assert "Markdown" in SYSTEM
        assert "**bold**" in SYSTEM

    def test_bold_and_citations_render_as_telegram_html(self):
        text = (
            "По переписке **без пермита** чаще всего советуют не «полноценные» "
            "швейцарские банки, а **Revolut и Wise** — в т.ч. по фото визы, "
            "пока ждёте пермит [W3]. У **Revolut** есть ограничение: не откроют, "
            "если срок визы **from–until меньше 90 дней**, даже с регистрацией [W3]."
        )
        body = format_answer_body(Answer(text, [hit(1), hit(2), hit(3)]))
        assert "<b>без пермита</b>" in body
        assert "<b>Revolut и Wise</b>" in body
        assert "<b>from–until меньше 90 дней</b>" in body
        assert f'<a href="{hit(3).link()}">[W3]</a>' in body
        assert "**" not in body

    def test_parenthetical_citation_url_is_replaced_with_source_link(self):
        text = "advice [W3] (https://evil.example/x) and [W2](https://evil.example/y)"
        body = format_answer_body(Answer(text, [hit(1), hit(2), hit(3)]))
        assert "evil.example" not in body
        assert f'<a href="{hit(3).link()}">[W3]</a>' in body
        assert f'<a href="{hit(2).link()}">[W2]</a>' in body

    def test_markdown_link_and_code_and_italic(self):
        html = markdown_to_html("see [Revolut](https://revolut.com) and `IBAN` vs *maybe*")
        assert '<a href="https://revolut.com">Revolut</a>' in html
        assert "<code>IBAN</code>" in html
        assert "<i>maybe</i>" in html

    def test_raw_html_is_escaped(self):
        html = markdown_to_html("use <b>tags</b> & **this**")
        assert "&lt;b&gt;tags&lt;/b&gt;" in html
        assert "<b>this</b>" in html

    def test_star_lists_are_not_italic(self):
        html = markdown_to_html("* Revolut\n* Wise")
        assert "<i>" not in html
        assert "* Revolut" in html

    def test_outer_markdown_fence_is_unwrapped(self):
        html = markdown_to_html("```markdown\n**ok** [W1]\n```")
        assert "<b>ok</b>" in html
        assert "<pre>" not in html
        assert "[W1]" in html


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


class TestPeopleNames:
    def test_record_and_resolve(self, conn):
        people.record(conn, 42, "Real Name", "realuser", "live")
        conn.commit()
        names = people.name_map(conn)
        assert people.resolve(names, 42, "Private Label") == "Real Name"

    def test_keeps_emoji_and_styled_unicode(self, conn):
        family = "👨‍👩‍👧"
        bold = "𝗔𝗻𝗻𝗮"  # mathematical sans-serif bold, not HTML
        people.record(conn, 1, f"  {family} Anna  ", None, "live")
        people.record(conn, 2, bold, None, "api")
        people.record(conn, 3, "<b>not html</b>", None, "api")
        conn.commit()
        names = people.name_map(conn)
        assert names[1] == f"{family} Anna"
        assert names[2] == bold
        assert names[3] == "<b>not html</b>"

    def test_name_from_user_joins_emoji_names(self):
        from types import SimpleNamespace

        user = SimpleNamespace(first_name="🔥 Nino", last_name="𝗕𝗼𝘁", full_name="ignored")
        assert people.name_from_user(user) == "🔥 Nino 𝗕𝗼𝘁"
        assert people.name_from_user(SimpleNamespace(first_name="  🎉  ", last_name="")) == "🎉"
        assert people.name_from_user(None) is None

    def test_pending_ids_skips_resolved_and_misses(self, conn):
        conn.executemany(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, sender_id, text) "
            "VALUES (1, ?, 1, 'x', ?, 'hi')",
            [(1, 10), (2, 20), (3, 30), (4, 40)],
        )
        people.record(conn, 10, "Anna", None, "live")
        people.mark_miss(conn, 20, "left")
        conn.commit()
        assert people.pending_ids(conn, 1) == [30, 40]
        assert people.pending_ids(conn, 1, retry_misses=True) == [20, 30, 40]
        stats = people.lookup_stats(conn, 1)
        assert stats == {"total": 4, "resolved": 1, "missed": 1, "pending": 2}

    def test_pending_lookups_try_each_chat_before_a_global_miss(self, conn):
        conn.executemany(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, sender_id, text) "
            "VALUES (?, ?, 1, 'x', ?, 'hi')",
            [(1, 1, 10), (2, 1, 10), (2, 2, 20), (3, 1, 30)],
        )
        people.record(conn, 10, "Anna", None, "live")
        conn.commit()
        lookups = people.pending_lookups(conn, [1, 2, 3])
        assert lookups == [(20, [2]), (30, [3])]
        assert people.pending_ids(conn, [1, 2]) == [20]
        stats = people.lookup_stats(conn, [1, 2])
        assert stats["total"] == 2
        assert stats["resolved"] == 1
        assert stats["pending"] == 1

    def test_record_clears_miss(self, conn):
        people.mark_miss(conn, 7, "left")
        people.record(conn, 7, "🔥 Victor", None, "live")
        conn.commit()
        assert people.pending_ids(conn, 1) == []
        assert conn.execute("SELECT count(*) FROM resolve_misses").fetchone()[0] == 0

    def test_miss_reason_from_error(self):
        assert people.miss_reason_from_error("USER_NOT_PARTICIPANT") == "left"
        assert people.miss_reason_from_error("member list is inaccessible") == "hidden"
        assert people.miss_reason_from_error("Forbidden: bot was blocked") == "forbidden"
        assert people.miss_reason_from_error("timeout") == "error"

    def test_resolve_falls_back_to_given_string(self, conn):
        assert people.resolve({}, 42, "Private Label") == "Private Label"
        assert people.resolve({}, None, "Private Label") == "Private Label"

    def test_name_mode_falls_back_to_alias_not_contact(self):
        assert people.speaker_label({}, 7, "Private Label", mode="name", aliases={7: 3}) == "User 3"
        assert people.speaker_label({}, None, "Private Label", mode="name") == "User unknown"

    def test_export_mode_uses_contact_as_fallback(self):
        assert people.speaker_label({7: "Victor"}, 7, "Private Label", mode="export") == "Victor"
        assert people.speaker_label({}, 7, "Private Label", mode="export") == "Private Label"

    def test_more_trusted_source_wins_and_is_kept(self, conn):
        people.record(conn, 1, "Live Name", None, "live")
        people.record(conn, 1, "Manual Name", None, "manual")
        assert people.name_map(conn)[1] == "Manual Name"
        # a later live sighting must NOT clobber the hand-set name
        assert people.record(conn, 1, "Live Again", None, "live") is False
        assert people.name_map(conn)[1] == "Manual Name"

    def test_render_uses_resolved_names(self):
        msgs = [
            {"ts": 0, "sender": "Vutyan нейроэкономика Витя", "sender_id": 7, "text": "hi"},
            {"ts": 1, "sender": "Unknown Label", "sender_id": 8, "text": "yo"},
        ]
        out = render(msgs, names={7: "Victor"}, mode="name", aliases={7: 1, 8: 2})
        assert "Victor: hi" in out
        assert "Vutyan" not in out          # private label replaced
        assert "User 2: yo" in out          # unresolved sender is User N, not the contact label
        assert "Unknown Label" not in out

    def test_id_mode_uses_stable_alias_not_real_id(self):
        # even a resolved name is suppressed, and the label is the ordinal, not the id
        assert people.speaker_label({7: "Victor"}, 7, "Label", mode="id", aliases={7: 1}) == "User 1"
        assert "7" not in people.speaker_label({}, 7, "Label", mode="id", aliases={7: 3})
        assert people.speaker_label({}, None, "Label", mode="id") == "User unknown"

    def test_name_mode_still_resolves(self):
        assert people.speaker_label({7: "Victor"}, 7, "Label", mode="name") == "Victor"

    def test_known_speakers_omit_contact_labels(self, conn, monkeypatch):
        monkeypatch.setattr(config, "SPEAKER_LABEL", "name")
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, sender_id, text) "
            "VALUES (1, 1, 1, 'Private Label', 99, 'hi')"
        )
        people.record(conn, 99, "Victor", None, "live")
        conn.commit()
        assert people.known_speakers(conn) == ["Victor"]
        monkeypatch.setattr(config, "SPEAKER_LABEL", "export")
        assert "Private Label" in people.known_speakers(conn)

    def test_ensure_aliases_is_stable(self, conn):
        people.ensure_aliases(conn, [500, 100, 300])
        first = people.alias_map(conn)
        assert set(first.values()) == {1, 2, 3}       # sequential ordinals
        assert set(first) == {500, 100, 300}
        # a later batch keeps existing ordinals and only appends new ones
        people.ensure_aliases(conn, [100, 999])
        second = people.alias_map(conn)
        assert second[100] == first[100]              # unchanged
        assert second[999] == 4                        # next free ordinal

    def test_parse_who_arg(self):
        assert people.parse_who_arg(" 12345 ") == ("id", 12345)
        assert people.parse_who_arg("User 3") == ("alias", 3)
        assert people.parse_who_arg("user#12") == ("alias", 12)
        assert people.parse_who_arg("#7") == ("alias", 7)
        assert people.parse_who_arg("0") is None
        assert people.parse_who_arg("User 0") is None
        assert people.parse_who_arg("@anna") is None
        assert people.parse_who_arg("") is None

    def test_whois_and_format(self, conn):
        people.record(conn, 99, "🔥 Anna", "anna_bot", "api")
        people.ensure_aliases(conn, [99])
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, sender_id, text) "
            "VALUES (1, 1, 1, 'Private Label', 99, 'hi')"
        )
        conn.commit()
        info = people.whois(conn, 99)
        assert info["display_name"] == "🔥 Anna"
        assert info["username"] == "anna_bot"
        assert info["alias"] == 1
        assert info["messages"] == 1
        assert info["export_name"] == "Private Label"
        text = people.format_who("en", info)
        assert "99" in text
        assert "🔥 Anna" in text
        assert "@anna_bot" in text
        assert "User 1" in text
        assert "Private Label" not in text
        assert "&lt;" in people.format_who("en", {**info, "display_name": "<b>x</b>"})
        missing = people.whois(conn, 123)
        assert not people.who_has_local_info(missing)
        assert people.sender_id_for_alias(conn, 1) == 99
        assert people.sender_id_for_alias(conn, 99) is None

    def test_whois_export_only(self, conn):
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, sender_id, text) "
            "VALUES (1, 1, 1, 'Private Label', 7, 'hi')"
        )
        conn.commit()
        info = people.whois(conn, 7)
        assert info["display_name"] is None
        assert info["export_name"] == "Private Label"
        assert "Private Label" in people.format_who("en", info)

    def test_render_id_mode_shows_no_names_or_real_ids(self):
        msgs = [
            {"ts": 0, "sender": "Vutyan нейроэкономика Витя", "sender_id": 7, "text": "hi"},
            {"ts": 1, "sender": "Some Label", "sender_id": 8, "text": "yo"},
        ]
        out = render(msgs, names={7: "Victor"}, mode="id", aliases={7: 1, 8: 2})
        assert "User 1: hi" in out and "User 2: yo" in out
        assert "Victor" not in out and "Vutyan" not in out and "Some Label" not in out
        assert "User 7" not in out and "User 8" not in out  # real ids never shown

    def test_reindex_id_mode(self, conn, monkeypatch, fake_embed):
        monkeypatch.setattr(config, "SPEAKER_LABEL", "id")
        seed(conn, [(1, 0, "hello")])
        conn.execute("UPDATE messages SET sender='Private Label', sender_id=99")
        conn.commit()
        people.record(conn, 99, "Actual Person", None, "manual")
        conn.commit()

        index.reindex(conn, progress=False)

        row = conn.execute("SELECT text, speakers FROM windows WHERE chat_id=1").fetchone()
        assert row["speakers"] == "User 1"             # ordinal, not the id 99
        assert "99" not in row["text"]                  # real id never leaks
        assert "Actual Person" not in row["text"] and "Private Label" not in row["text"]

    def test_reindex_name_mode_skips_contact_labels(self, conn, monkeypatch, fake_embed):
        monkeypatch.setattr(config, "SPEAKER_LABEL", "name")
        seed(conn, [(1, 0, "hello")])
        conn.execute("UPDATE messages SET sender='Private Label', sender_id=99")
        conn.commit()

        index.reindex(conn, progress=False)

        row = conn.execute("SELECT text, speakers FROM windows WHERE chat_id=1").fetchone()
        assert row["speakers"] == "User 1"
        assert "Private Label" not in row["text"]
        assert "99" not in row["text"]

    def test_reindex_export_mode_keeps_contact_labels(self, conn, monkeypatch, fake_embed):
        monkeypatch.setattr(config, "SPEAKER_LABEL", "export")
        seed(conn, [(1, 0, "hello")])
        conn.execute("UPDATE messages SET sender='Private Label', sender_id=99")
        conn.commit()

        index.reindex(conn, progress=False)

        row = conn.execute("SELECT text, speakers FROM windows WHERE chat_id=1").fetchone()
        assert row["speakers"] == "Private Label"
        assert "Private Label" in row["text"]

    def test_load_mapping_counts_written(self, conn):
        n = people.load_mapping(conn, [
            {"sender_id": 1, "name": "Anna"},
            {"sender_id": 2, "name": "Boris"},
            {"sender_id": 3, "name": ""},      # empty name is skipped
        ])
        assert n == 2


def test_reindex_applies_resolved_names(conn):
    """End to end: a resolved name replaces the export label in window text."""
    seed(conn, [(1, 0, "hello"), (2, 5, "world")])
    conn.execute("UPDATE messages SET sender='Private Label', sender_id=99")
    conn.commit()
    people.record(conn, 99, "Actual Person", None, "manual")
    conn.commit()

    # index without embeddings by faking the encoder
    import numpy as np
    from answerbot import embed
    orig = embed.encode_passages
    embed.encode_passages = lambda texts, **k: np.zeros((len(texts), config.EMBED_DIM), np.float32)
    try:
        index.reindex(conn, progress=False)
    finally:
        embed.encode_passages = orig

    row = conn.execute("SELECT text, speakers FROM windows WHERE chat_id=1").fetchone()
    assert "Actual Person" in row["text"]
    assert "Private Label" not in row["text"]
    assert row["speakers"] == "Actual Person"


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


class FakeLLM:
    def complete(self, system, user):
        return "found it [W1]"


class TestChatScope:
    def test_normalize_chat_ids(self):
        assert retrieve.normalize_chat_ids(None) is None
        assert retrieve.normalize_chat_ids(5) == [5]
        assert retrieve.normalize_chat_ids([1, 2]) == [1, 2]
        assert retrieve.normalize_chat_ids(()) == []

    def test_empty_allow_list_returns_nothing(self, conn, fake_embed):
        seed(conn, STREAM[:10])
        index.reindex(conn, progress=False)
        assert retrieve.search(conn, "m1", chat_id=[]) == []

    def test_does_not_return_other_chats(self, conn, fake_embed):
        seed(conn, [(1, 1, "pangolin in alpha")], chat_id=1)
        seed(conn, [(1, 1, "pangolin in beta")], chat_id=2)
        index.reindex(conn, progress=False)
        hits = retrieve.search(conn, "pangolin", chat_id=1)
        assert hits
        assert all(h.chat_id == 1 for h in hits)
        assert all("beta" not in h.text for h in hits)

    def test_allow_list_includes_only_listed_chats(self, conn, fake_embed):
        seed(conn, [(1, 1, "alpha unique")], chat_id=1)
        seed(conn, [(1, 1, "beta unique")], chat_id=2)
        seed(conn, [(1, 1, "gamma unique")], chat_id=3)
        index.reindex(conn, progress=False)
        hits = retrieve.search(conn, "unique", chat_id=[1, 2])
        chats = {h.chat_id for h in hits}
        assert chats == {1, 2}

    def test_query_vec_skips_encode_query(self, conn, fake_embed, monkeypatch):
        seed(conn, [(1, 1, "pangolin")], chat_id=1)
        index.reindex(conn, progress=False)

        def boom(q):
            raise AssertionError("encode_query should not run")

        monkeypatch.setattr("answerbot.embed.encode_query", boom)
        vec = np.zeros(config.EMBED_DIM, dtype=np.float32)
        hits = retrieve.search(conn, "pangolin", chat_id=1, query_vec=vec)
        assert hits
        assert all("pangolin" in h.text for h in hits)

    def test_answer_with_empty_allow_list_does_not_search(self, conn, fake_embed):
        seed(conn, [(1, 1, "secret from other chat")], chat_id=1)
        index.reindex(conn, progress=False)
        result = run_answer(conn, "secret", chat_id=[], llm=FakeLLM())
        assert result.hits == []
        assert "couldn't find" in result.text.lower()

    def test_query_log_keeps_the_allow_list(self, conn, fake_embed, monkeypatch):
        monkeypatch.setattr(config, "QUERY_LOG", True)
        seed(conn, [(1, 1, "alpha unique")], chat_id=1)
        seed(conn, [(1, 1, "beta unique")], chat_id=2)
        index.reindex(conn, progress=False)
        run_answer(conn, "unique", chat_id=[1, 2], llm=FakeLLM())
        row = conn.execute("SELECT chat_ids FROM query_log").fetchone()
        assert json.loads(row[0]) == [1, 2]

    def test_build_context_labels_source_chat_when_mixed(self):
        hits = [
            Hit(1, 11, 1, 1, 0, 0, "Anna", "alpha body", 0.1),
            Hit(2, 22, 1, 1, 0, 0, "Nino", "beta body", 0.1),
        ]
        titles = {11: "Main", 22: "Offtopic"}
        assert chat_label(hits[0], titles) == "Main"
        ctx = build_context(hits, titles)
        assert "[W1] Main," in ctx
        assert "[W2] Offtopic," in ctx
        single = build_context(hits[:1], titles)
        assert "Main" not in single
        assert "[W1] " in single

    def test_complete_answer_prompt_includes_chat_titles(self):
        class CaptureLLM:
            def __init__(self):
                self.user = None

            def complete(self, system, user):
                self.user = user
                return "found it [W1]"

        llm = CaptureLLM()
        hits = [
            Hit(1, 11, 1, 1, 0, 0, "Anna", "alpha body", 0.1),
            Hit(2, 22, 1, 1, 0, 0, "Nino", "beta body", 0.1),
        ]
        complete_answer("what", hits, llm, chat_titles={11: "Main", 22: "Offtopic"})
        assert "Main" in llm.user
        assert "Offtopic" in llm.user


class TestAnswerFlushesTail:
    def test_unwindowed_message_is_invisible_until_flush(self, conn, fake_embed):
        seed(conn, STREAM)
        index.reindex(conn, progress=False)
        last_ts = STREAM[-1][1]
        seed(conn, [(200, last_ts + 5, "pangolin sighting")])

        assert conn.execute(
            "SELECT count(*) FROM windows WHERE chat_id=1 AND last_msg>=200"
        ).fetchone()[0] == 0
        # The new text is not in any window yet, so it cannot appear in hits.
        # (Dummy zero vectors still rank existing windows; we care about the text.)
        hits = retrieve.search(conn, "pangolin sighting", chat_id=1)
        assert all("pangolin" not in h.text.lower() for h in hits)

        result = run_answer(conn, "pangolin sighting", chat_id=1, llm=FakeLLM())
        assert any("pangolin" in h.text.lower() for h in result.hits)


class TestLiveEdits:
    def _window_text_for(self, conn, msg_id):
        return conn.execute(
            "SELECT text FROM windows WHERE chat_id=1 AND first_msg<=? AND last_msg>=?",
            (msg_id, msg_id),
        ).fetchone()["text"]

    def test_refresh_if_in_tail_rebuilds_last_window(self, conn, fake_embed):
        seed(conn, STREAM)
        index.reindex(conn, progress=False)
        last_id = STREAM[-1][0]
        conn.execute(
            "UPDATE messages SET text='EDITED_TAIL' WHERE chat_id=1 AND msg_id=?", (last_id,)
        )
        conn.commit()
        assert live.refresh_if_in_tail(conn, 1, last_id) is True
        assert "EDITED_TAIL" in self._window_text_for(conn, last_id)

    def test_refresh_skips_old_messages(self, conn, fake_embed):
        seed(conn, STREAM)
        index.reindex(conn, progress=False)
        conn.execute("UPDATE messages SET text='EDITED_OLD' WHERE chat_id=1 AND msg_id=2")
        conn.commit()
        assert live.refresh_if_in_tail(conn, 1, 2) is False
        assert "EDITED_OLD" not in self._window_text_for(conn, 2)

    def test_force_update_refreshes_tail_without_new_messages(self, conn, fake_embed):
        seed(conn, STREAM)
        index.reindex(conn, progress=False)
        last_id = STREAM[-1][0]
        conn.execute(
            "UPDATE messages SET text='FORCED' WHERE chat_id=1 AND msg_id=?", (last_id,)
        )
        conn.commit()
        fake_embed.clear()
        result = index.update(conn)  # no new messages, lookback 0
        assert result["windows"] == 0
        assert "FORCED" not in self._window_text_for(conn, last_id)

        result = index.update(conn, force=True)
        assert result["windows"] > 0
        assert "FORCED" in self._window_text_for(conn, last_id)



class TestCapHits:
    def _hits(self, cosines):
        return [
            Hit(i, 1, i, i, 0, 0, "A", f"t{i}", 1.0 - i * 0.01, cosine)
            for i, cosine in enumerate(cosines, 1)
        ]

    def test_stops_after_min_k_once_cosine_drops(self):
        kept = retrieve.cap_hits(
            self._hits([0.9, 0.85, 0.8, 0.4, 0.9]), min_k=3, max_k=5, cosine_min=0.7
        )
        assert [h.window_id for h in kept] == [1, 2, 3]

    def test_keeps_high_cosine_hits_until_max_k(self):
        kept = retrieve.cap_hits(
            self._hits([0.9, 0.88, 0.86, 0.84, 0.82, 0.80]),
            min_k=3,
            max_k=5,
            cosine_min=0.7,
        )
        assert [h.window_id for h in kept] == [1, 2, 3, 4, 5]

    def test_zero_threshold_is_a_plain_max_k_cap(self):
        kept = retrieve.cap_hits(
            self._hits([0.1, 0.1, 0.1, 0.1]), min_k=2, max_k=3, cosine_min=0
        )
        assert len(kept) == 3

    def test_search_caps_when_top_k_is_omitted(self, conn, fake_embed, monkeypatch):
        monkeypatch.setattr(config, "MIN_K", 2)
        monkeypatch.setattr(config, "MAX_K", 5)
        monkeypatch.setattr(config, "COSINE_MIN", 0.7)
        seed(conn, STREAM[:20])
        index.reindex(conn, progress=False)
        # Dummy zero vectors => cosine 0, so the cutoff trips right after MIN_K.
        hits = retrieve.search(conn, "m1", chat_id=1)
        assert len(hits) == 2

    def test_explicit_top_k_skips_the_cosine_cap(self, conn, fake_embed, monkeypatch):
        monkeypatch.setattr(config, "MIN_K", 1)
        monkeypatch.setattr(config, "MAX_K", 2)
        monkeypatch.setattr(config, "COSINE_MIN", 0.7)
        seed(conn, STREAM)
        index.reindex(conn, progress=False)
        hits = retrieve.search(conn, "m1", chat_id=1, top_k=5)
        # STREAM yields 3 windows. Cap would stop at MIN_K=1 (cosine is 0);
        # skipping it returns every fused window, even past MAX_K=2.
        assert len(hits) == 3
        assert len(retrieve.search(conn, "m1", chat_id=1)) == 1


class TestRecency:
    def test_disabled_half_life_is_a_no_op(self):
        assert retrieve.recency_weight(0, 10**9, 0) == 1.0
        assert retrieve.recency_weight(0, 10**9, -5) == 1.0

    def test_one_half_life_halves_the_weight(self):
        now = 1_700_000_000
        w = retrieve.recency_weight(now - 10 * 86400, now, 10)
        assert abs(w - 0.5) < 1e-9

    def test_future_timestamps_are_not_boosted(self):
        assert retrieve.recency_weight(200, 100, 30) == 1.0

    def test_newer_duplicate_outranks_the_older_one(self, conn, fake_embed, monkeypatch):
        monkeypatch.setattr(config, "RECENCY_HALF_LIFE_DAYS", 365)
        now = datetime(2026, 8, 1, tzinfo=timezone.utc)
        old = int(datetime(2024, 8, 1, tzinfo=timezone.utc).timestamp())
        new = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())
        seed(
            conn,
            [
                (1, old, "the office wifi password is hunter2"),
                (2, new, "the office wifi password is hunter2"),
            ],
        )
        index.reindex(conn, progress=False)
        hits = retrieve.search(conn, "wifi password", chat_id=1, now=now)
        assert len(hits) >= 2
        assert hits[0].ts_end == new
        assert hits[1].ts_end == old

    def test_time_range_skips_recency_decay(self, conn, fake_embed, monkeypatch):
        """An explicit period already scoped the hits; don't decay inside it."""
        monkeypatch.setattr(config, "RECENCY_HALF_LIFE_DAYS", 365)

        def boom(*_a, **_k):
            raise AssertionError("recency should not run when a time range is set")

        monkeypatch.setattr(retrieve, "recency_weight", boom)
        now = datetime(2026, 8, 1, tzinfo=timezone.utc)
        old = int(datetime(2024, 8, 1, tzinfo=timezone.utc).timestamp())
        new = int(datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp())
        seed(
            conn,
            [
                (1, old, "the office wifi password is hunter2"),
                (2, new, "the office wifi password is hunter2"),
            ],
        )
        index.reindex(conn, progress=False)
        span = TimeRange(old - 10, new + 10, "both")
        hits = retrieve.search(
            conn, "wifi password", chat_id=1, now=now, time_range=span
        )
        assert {h.ts_end for h in hits} >= {old, new}


class TestConnect:
    def test_checkpoint_on_memory_is_a_no_op(self, conn):
        db.checkpoint(conn)

    def test_file_open_roundtrip(self, tmp_path):
        path = tmp_path / "chat.db"
        conn = db.connect(path)
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1, 1, 1, 'A', 'hi')"
        )
        conn.commit()
        db.checkpoint(conn)
        conn.close()
        conn = db.connect(path)
        assert conn.execute("SELECT text FROM messages").fetchone()[0] == "hi"
        conn.close()

    def test_garbage_wal_is_dropped(self, tmp_path):
        path = tmp_path / "chat.db"
        conn = db.connect(path)
        conn.execute(
            "INSERT INTO messages (chat_id, msg_id, ts, sender, text) VALUES (1, 1, 1, 'A', 'hi')"
        )
        conn.commit()
        db.checkpoint(conn)
        conn.close()
        (tmp_path / "chat.db-wal").write_bytes(b"not a wal" * 200)
        (tmp_path / "chat.db-shm").write_bytes(b"\x00" * 32)
        conn = db.connect(path)
        assert conn.execute("SELECT text FROM messages").fetchone()[0] == "hi"
        conn.close()

    def test_junk_file_names_the_path(self, tmp_path):
        path = tmp_path / "chat.db"
        path.write_bytes(b"not a sqlite database")
        with pytest.raises(sqlite3.DatabaseError, match="chat.db"):
            db.connect(path)


class TestEncodeProgress:
    def test_on_progress_fires_per_batch(self, monkeypatch):
        class FakeModel:
            def encode(self, texts, **_kw):
                return np.zeros((len(texts), config.EMBED_DIM), np.float32)

        monkeypatch.setattr("answerbot.embed._get_model", lambda: FakeModel())
        monkeypatch.setattr("answerbot.embed._needs_e5_prefix", lambda: False)
        from answerbot import embed

        seen: list[tuple[int, int]] = []
        texts = ["a"] * 130
        vecs = embed.encode_passages(
            texts,
            batch_size=64,
            on_progress=lambda done, n: seen.append((done, n)),
        )
        assert vecs.shape == (130, config.EMBED_DIM)
        assert seen == [(64, 130), (128, 130), (130, 130)]
