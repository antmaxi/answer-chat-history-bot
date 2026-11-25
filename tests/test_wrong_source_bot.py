"""Exercise feedback from the rendered Telegram button through the dispatcher."""

import asyncio
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot
from aiogram.types import Update

from answerbot import answer, config, db, i18n, logconfig


@pytest.fixture
def bot_module(monkeypatch):
    conn = db.connect(":memory:", check_same_thread=False)
    with monkeypatch.context() as setup:
        setup.setattr(db, "connect", lambda **kwargs: conn)
        setup.setattr(logconfig, "setup", lambda: None)
        from answerbot import bot

    monkeypatch.setattr(bot, "conn", conn)
    async def run_inline(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_inline)
    monkeypatch.setattr(bot, "_lang_for", AsyncMock(return_value="en"))
    monkeypatch.setattr(bot, "_history", defaultdict(deque))
    monkeypatch.setattr(bot, "_note_ask_latency", lambda seconds: None)
    monkeypatch.setattr(bot, "_answers", bot.cooldown.Cooldown(0))
    monkeypatch.setattr(bot, "_quota_block", lambda *args: None)
    monkeypatch.setattr(bot, "_embed", AsyncMock(return_value=None))
    monkeypatch.setattr(bot, "_spin_thinking", AsyncMock(return_value=None))
    monkeypatch.setattr(config, "ADMIN_USER_IDS", [])
    monkeypatch.setattr(config, "TELEGRAM_CHAT_ID", -100123)
    yield bot
    conn.close()


@pytest.mark.parametrize("logging_enabled", [True, False])
def test_wrong_source_button_and_callback(bot_module, monkeypatch, logging_enabled):
    bot = bot_module
    monkeypatch.setattr(config, "QUERY_LOG", logging_enabled)
    hit = SimpleNamespace(log_id=lambda: 42)
    result = answer.Answer("See the schedule [W1].", [hit])
    monkeypatch.setattr(bot.retrieve, "search", lambda *args, **kwargs: [hit])
    monkeypatch.setattr(bot.answer, "complete_answer", lambda *args, **kwargs: result)
    monkeypatch.setattr(bot, "format_answer", lambda *args, **kwargs: result.text)
    thinking = SimpleNamespace(edit_text=AsyncMock())
    message = SimpleNamespace(
        chat=SimpleNamespace(id=-100123),
        from_user=SimpleNamespace(id=12),
        reply_to_message=None,
        reply=AsyncMock(return_value=thinking),
    )
    session = AsyncMock()
    telegram = Bot("123:token", session=session)

    async def click(button, user_id):
        update = Update.model_validate({
            "update_id": user_id,
            "callback_query": {
                "id": str(user_id),
                "from": {"id": user_id, "is_bot": False, "first_name": "User"},
                "chat_instance": "group",
                "message": {
                    "message_id": 100,
                    "date": 1700000000,
                    "chat": {"id": -100123, "type": "supergroup"},
                },
                "data": button.callback_data,
            },
        })
        await bot.dp.feed_update(telegram, update)
        return session.await_args.args[1].text

    async def exercise():
        await bot.respond(message, "Where is the schedule?", -100123)
        thinking.edit_text.assert_awaited_once()
        markup = thinking.edit_text.await_args.kwargs["reply_markup"]
        if not logging_enabled:
            assert markup is None
            assert bot.conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0] == 0
            return
        button = markup.inline_keyboard[0][0]
        assert button.text == "Wrong source"
        assert await click(button, 13) == i18n.t("en", "report_unavailable")
        assert not db.retrieval_misses(bot.conn)
        assert await click(button, 12) == i18n.t("en", "report_saved")
        row, = db.retrieval_misses(bot.conn)
        assert row["question"] == "Where is the schedule?"
        assert row["cited_ids"] == "[42]"

    asyncio.run(exercise())
