"""The Telegram bot: aiogram wrapper around retrieve + answer, plus live ingest.

Pinned to one supergroup (`TELEGRAM_CHAT_ID`). Replies when @mentioned or when
someone replies to one of its own messages. Requires privacy mode OFF in
BotFather, or it receives no group messages to index at all.

DM behaviour: any plain message is a question, if the sender is a member of
that group. Search always runs against `TELEGRAM_CHAT_ID`.
"""

import asyncio
import logging
import re
import threading
import time

from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, F, html
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramForbiddenError
from aiogram.filters import Command
from aiogram.types import Message

from . import adminlog, answer, config, cooldown, db, embed, followup, index, logconfig, membership, people, retrieve
from .ingest import live
from .ingest.export import desktop_ids_for

logconfig.setup()
log = logging.getLogger("answerbot")

dp = Dispatcher()
# One shared connection, used from worker threads (asyncio.to_thread), so it's
# opened without the same-thread guard and every access is serialized by a lock.
conn = db.connect(check_same_thread=False)
_db_lock = threading.Lock()
# Indexing (window rebuild + encode) is serialized separately so ingest and
# search can proceed while SentenceTransformer is running.
_index_lock = asyncio.Lock()
_answers = cooldown.Cooldown(config.ANSWER_COOLDOWN_SECONDS)
_members = membership.MembershipCache(config.MEMBERSHIP_CACHE_SECONDS)
_history: dict[tuple[int, int], deque[str]] = defaultdict(lambda: deque(maxlen=4))
_admin_errors = adminlog.AdminErrorHandler()


async def _db(fn, *args):
    """Run a blocking DB call off the event loop, one at a time."""
    def locked():
        with _db_lock:
            return fn(*args)

    return await asyncio.to_thread(locked)


async def _apply_jobs(jobs: list[index.EmbedJob]) -> dict:
    """Encode off the DB lock, then write vectors under it."""
    windows = 0
    for job in jobs:
        vecs = None
        if job.pending_texts:
            vecs = await asyncio.to_thread(embed.encode_passages, job.pending_texts, 64, False)
        windows += await _db(index.apply_job, conn, job, vecs)
    return {"chats": len(jobs), "windows": windows}


async def index_chats(
    chat_id: int | list[int] | None,
    *,
    lookback: int = 0,
    force: bool = False,
    full: bool = False,
) -> dict:
    """Rebuild windows (and encode) without holding the DB lock during embed."""
    async with _index_lock:
        if full:
            jobs = await _db(index.plan_reindex, conn, chat_id)
        else:
            jobs = await _db(index.plan_update, conn, chat_id, lookback, force)
        return await _apply_jobs(jobs)


async def _background_index(chat_id: int, *, force: bool = False) -> None:
    try:
        await index_chats(chat_id, force=force)
    except Exception:
        log.exception("background reindex failed chat_id=%s", chat_id)


async def _periodic_lookback() -> None:
    hours = config.LIVE_LOOKBACK_HOURS
    if hours <= 0:
        return
    while True:
        await asyncio.sleep(hours * 3600)
        log.info("periodic lookback: last %s days", config.UPDATE_LOOKBACK_DAYS)
        try:
            await index_chats(config.TELEGRAM_CHAT_ID, lookback=config.UPDATE_LOOKBACK_DAYS)
        except Exception:
            log.exception("periodic lookback failed")


def _configured_chat() -> int:
    chat_id = config.TELEGRAM_CHAT_ID
    if chat_id is None:
        raise RuntimeError("TELEGRAM_CHAT_ID is not set")
    return chat_id


def _span_lines(s: dict) -> str:
    first, last = s.get("first_message"), s.get("last_message")
    if not first or not last:
        return ""
    return f"\nfirst: {first}\nlast: {last}"


def _is_configured_chat(chat_id: int) -> bool:
    return chat_id == _configured_chat()


async def _user_in_configured_chat(bot: Bot, user_id: int) -> bool:
    """Whether this user is currently a member of TELEGRAM_CHAT_ID."""
    chat_id = _configured_chat()
    cached = _members.get(user_id, chat_id)
    if cached is not None:
        return cached
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        ok = membership.is_chat_member(member)
    except Exception:
        ok = False
    _members.remember(user_id, chat_id, ok)
    return ok


_DECLINE = "You're not a member of the group this bot serves."


async def _ensure_member(message: Message, bot: Bot) -> bool:
    """Allow the configured group, or a DM from a current member. Otherwise decline."""
    user = message.from_user
    if user is None:
        return False
    if message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        return _is_configured_chat(message.chat.id)
    if await _user_in_configured_chat(bot, user.id):
        return True
    await message.reply(_DECLINE)
    return False


def format_answer(result: answer.Answer) -> str:
    # Quote first (brackets survive escaping), then turn each [W#] into a link to
    # the relevant messages, so the citations in the answer are clickable.
    body = html.quote(result.text)

    def linkify(m: "re.Match") -> str:
        i = int(m.group(1))
        if 1 <= i <= len(result.hits):
            return f'<a href="{result.hits[i - 1].link()}">[W{i}]</a>'
        return m.group(0)

    body = answer.CITATION.sub(linkify, body)

    # A direct jump to the first message the answer is grounded in.
    link = result.primary_link()
    if link:
        body += f'\n\n➡️ <a href="{link}">Go to the first message</a>'

    sources = result.all_sources()
    if sources:
        lines = "\n".join(
            f'<a href="{h.link()}">[W{i}]</a>{" ✓" if was_cited else ""} '
            f'{html.quote(h.when())} · {html.quote(h.speakers)}'
            for i, h, was_cited in sources
        )
        body += "\n\n<b>Sources</b>\n" + lines
    return body


def _record_person(sender_id, name, username, source="live") -> None:
    people.record(conn, sender_id, name, username, source)
    conn.commit()


async def _align_export_chat_ids() -> None:
    """Rewrite a Desktop-export chat id onto TELEGRAM_CHAT_ID if it is still stored."""
    target = _configured_chat()
    moved = 0
    for old in desktop_ids_for(target):
        n = await _db(db.remap_chat_id, conn, old, target)
        if n:
            log.info("aligning desktop chat_id %s -> Bot API %s (%s messages)", old, target, n)
            moved += n
    if moved:
        retrieve.invalidate_cache()
        _members.invalidate()


async def respond(message: Message, question: str, chat_id: int | list[int]) -> None:
    if not question.strip():
        await message.reply("Ask me a question about this chat's history.")
        return
    user_id = message.from_user.id if message.from_user else 0
    wait = _answers.remaining(
        (user_id, message.chat.id),
        exempt=user_id in config.ADMIN_USER_IDS,
    )
    if wait > 0:
        await message.reply(f"Wait {int(wait) + 1}s before asking again.")
        return
    _answers.touch((user_id, message.chat.id))

    key = (message.chat.id, user_id)
    prior = _history[key][-1] if _history[key] else None
    force = False
    reply = message.reply_to_message
    if reply and reply.from_user:
        me = await message.bot.me()
        force = reply.from_user.id == me.id
    search_q = followup.rewrite(question, prior, force=force)

    thinking = await message.reply("…")
    t0 = time.monotonic()
    try:
        # Encode any unwindowed tail off the DB lock, then search under it,
        # then call the LLM without holding either lock.
        await index_chats(chat_id)
        hits = await _db(retrieve.search, conn, search_q, chat_id)

        def complete():
            return answer.complete_answer(question, hits)

        result = await asyncio.to_thread(complete)
        await _db(answer._record, conn, question, chat_id, result, t0, None)
        _history[key].append(question)
        await thinking.edit_text(format_answer(result), parse_mode="HTML", disable_web_page_preview=True)
    except Exception:
        log.exception("failed to answer")
        await thinking.edit_text("Something went wrong answering that.")


# --- Commands -------------------------------------------------------------

@dp.message(Command("start", "help"))
async def cmd_help(message: Message, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    await message.reply(
        "I answer questions from the group's history.\n"
        "In the group, @mention me or reply to my messages. In DM, just ask.\n"
        "Commands: /ask <question>, /stats"
        "\nAdmins: /reindex (recent), /reindex full, /resolve (fix member names)"
    )


@dp.message(Command("ask"))
async def cmd_ask(message: Message, command, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    await respond(message, command.args or "", _configured_chat())


@dp.message(Command("stats"))
async def cmd_stats(message: Message, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    s = await _db(db.stats, conn)
    await message.reply(
        f"messages: {s['messages']}\nwindows: {s['windows']}\n"
        f"embedded: {s['embedded']}\nchats: {s['chats']}"
        f"{_span_lines(s)}"
    )


@dp.message(Command("reindex"))
async def cmd_reindex(message: Message, command, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    if message.from_user.id not in config.ADMIN_USER_IDS:
        await message.reply("Admins only.")
        return
    chat_id = _configured_chat()
    full = (command.args or "").strip().lower() == "full"
    if full:
        await message.reply("Full reindex…")
        result = await index_chats(chat_id, full=True)
    else:
        await message.reply("Updating recent history…")
        result = await index_chats(chat_id, lookback=config.UPDATE_LOOKBACK_DAYS)
    await message.reply(f"Done: {result['windows']} windows across {result['chats']} chat(s).")


@dp.message(Command("resolve"))
async def cmd_resolve(message: Message, bot: Bot) -> None:
    """Look up members' real names via the Bot API, replacing export labels.

    Only people still in the configured group can be looked up; anyone who left
    keeps their export label.
    """
    if not await _ensure_member(message, bot):
        return
    if message.from_user.id not in config.ADMIN_USER_IDS:
        await message.reply("Admins only.")
        return
    chat_id = _configured_chat()

    ids = await _db(
        lambda: [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT sender_id FROM messages WHERE chat_id=? AND sender_id IS NOT NULL",
                (chat_id,),
            )
        ]
    )
    await message.reply(f"Resolving {len(ids)} people via the API — this can take a while…")

    done = 0
    for uid in ids:
        try:
            member = await bot.get_chat_member(chat_id, uid)
            await _db(_record_person, uid, member.user.full_name, member.user.username, "api")
            done += 1
        except Exception:
            continue  # user left the group, or the lookup was rejected
        await asyncio.sleep(0.1)  # be gentle with rate limits

    await message.reply(
        f"Resolved {done}/{len(ids)} names. Run /reindex to rewrite history with them."
    )


# --- Group messages -------------------------------------------------------

async def _mentions_bot(message: Message, bot: Bot) -> bool:
    me = await bot.me()
    if message.reply_to_message and message.reply_to_message.from_user.id == me.id:
        return True
    text = message.text or message.caption or ""
    return f"@{me.username}".lower() in text.lower()


async def _ingest_group_message(message: Message, *, edited: bool = False) -> None:
    text = message.text or message.caption
    if not text:
        return

    # Always ingest, so history keeps growing whether or not we're addressed.
    await _db(
        live.add_message,
        conn,
        message.chat.id,
        message.message_id,
        message.from_user.full_name if message.from_user else "Unknown",
        message.from_user.id if message.from_user else None,
        int(message.date.timestamp()),
        text,
        message.reply_to_message.message_id if message.reply_to_message else None,
    )
    # The sender's own full name is their real public name — record it so it
    # overrides any private export label on the next reindex.
    if message.from_user:
        u = message.from_user
        await _db(_record_person, u.id, u.full_name, u.username)
    if edited:
        asyncio.create_task(_background_index(message.chat.id, force=True))
    else:
        pending = await _db(live.pending_count, conn, message.chat.id)
        if pending >= config.LIVE_REINDEX_EVERY:
            asyncio.create_task(_background_index(message.chat.id))


@dp.message(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def on_group_message(message: Message, bot: Bot) -> None:
    if not _is_configured_chat(message.chat.id):
        return
    text = message.text or message.caption
    if not text:
        return

    await _ingest_group_message(message)

    if await _mentions_bot(message, bot):
        me = await bot.me()
        question = text.replace(f"@{me.username}", "").strip()
        await respond(message, question, _configured_chat())


@dp.edited_message(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def on_group_edit(message: Message) -> None:
    if not _is_configured_chat(message.chat.id):
        return
    await _ingest_group_message(message, edited=True)


# --- Direct messages ------------------------------------------------------

@dp.message(F.chat.type == ChatType.PRIVATE)
async def on_private_message(message: Message, bot: Bot) -> None:
    if not message.text:
        return
    if not await _ensure_member(message, bot):
        return
    await respond(message, message.text, _configured_chat())


async def _notify_admins(bot: Bot, text: str) -> None:
    """DM each admin. Telegram only delivers if they have already /start'd the bot."""
    for uid in sorted(config.ADMIN_USER_IDS):
        try:
            await bot.send_message(uid, text)
        except TelegramForbiddenError:
            log.warning("admin %s has not started a chat with the bot; cannot send %r", uid, text)
        except Exception:
            log.warning("failed to notify admin %s (%s)", uid, text, exc_info=True)


async def _on_startup(bot: Bot) -> None:
    s = db.stats(conn)
    log.info("bot is up; database %s: %s", config.DB_PATH, s)
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(logconfig.asyncio_handler)
    await _align_export_chat_ids()
    try:
        chat = await bot.get_chat(_configured_chat())
        title = chat.title or str(_configured_chat())
    except Exception:
        title = str(_configured_chat())
        log.warning("cannot reach TELEGRAM_CHAT_ID=%s", _configured_chat())
    log.info("configured chat %s (%s)", _configured_chat(), title)
    _admin_errors.attach(loop, lambda text: _notify_admins(bot, text))
    await _notify_admins(
        bot,
        f"Bot is up\n{config.DB_PATH}: {s['messages']} messages, {s['windows']} windows"
        f"{_span_lines(s)}"
        f"\nchat: {title} (`{_configured_chat()}`)",
    )


async def _on_shutdown(bot: Bot) -> None:
    log.info("bot is down")
    await _notify_admins(bot, "Bot is down")
    _admin_errors.detach()


dp.startup.register(_on_startup)
dp.shutdown.register(_on_shutdown)


async def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("set TELEGRAM_BOT_TOKEN (see .env.example)")
    if config.TELEGRAM_CHAT_ID is None:
        raise SystemExit("set TELEGRAM_CHAT_ID (the supergroup's Bot API id, see .env.example)")
    bot = Bot(config.TELEGRAM_BOT_TOKEN)
    log.info("starting polling")
    asyncio.create_task(_periodic_lookback())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
