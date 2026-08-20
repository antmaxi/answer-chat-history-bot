"""The Telegram bot: aiogram wrapper around retrieve + answer, plus live ingest.

Group behaviour: replies only when @mentioned or when someone replies to one of
its own messages, so it stays quiet otherwise. Requires privacy mode OFF in
BotFather, or it receives no group messages to index at all.

DM behaviour: any plain message is a question. Access is gated on membership of
indexed chats, and search is restricted to those chats — never the whole DB.
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
from .ingest.export import bot_api_candidates

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
_titles: dict[int, str] = {}
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
            await index_chats(None, lookback=config.UPDATE_LOOKBACK_DAYS)
        except Exception:
            log.exception("periodic lookback failed")


async def _chat_title(bot: Bot, chat_id: int) -> str:
    if chat_id in _titles:
        return _titles[chat_id]
    try:
        chat = await bot.get_chat(chat_id)
        _titles[chat_id] = chat.title or str(chat_id)
    except Exception:
        _titles[chat_id] = str(chat_id)
    return _titles[chat_id]


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


async def indexed_chats() -> set[int]:
    return await _db(
        lambda: {r[0] for r in conn.execute("SELECT DISTINCT chat_id FROM messages")}
    )


async def _align_export_chat_ids(bot: Bot) -> None:
    """Rewrite Desktop-export chat ids to the Bot API form Telegram will accept.

    getChat / getChatMember need `-100<id>` for a supergroup. An export stored
    under the bare positive id looks like a chat the bot has never joined.
    """
    for cid in sorted(await indexed_chats()):
        resolved = None
        for candidate in bot_api_candidates(cid):
            try:
                await bot.get_chat(candidate)
            except Exception:
                continue
            resolved = candidate
            break
        if resolved is None or resolved == cid:
            continue
        log.info("aligning desktop chat_id %s -> Bot API %s", cid, resolved)
        await _db(db.remap_chat_id, conn, cid, resolved)
        retrieve.invalidate_cache()
        _members.invalidate()


async def indexed_chats_for_user(bot: Bot, user_id: int) -> list[int]:
    """Indexed chats this user is currently a member of.

    This is the DM allow-list: search must not run over any other chat_id.
    Membership is cached so a busy DM does not call getChatMember on every ask.
    """
    allowed: list[int] = []
    for chat_id in sorted(await indexed_chats()):
        cached = _members.get(user_id, chat_id)
        if cached is True:
            allowed.append(chat_id)
            continue
        if cached is False:
            continue
        try:
            member = await bot.get_chat_member(chat_id, user_id)
            ok = member.status not in ("left", "kicked")
        except Exception:
            ok = False
        _members.remember(user_id, chat_id, ok)
        if ok:
            allowed.append(chat_id)
    return allowed


async def dm_scope(bot: Bot, user_id: int) -> list[int]:
    """Allow-list for a DM, honoring /chat if it still points at a membership."""
    allowed = await indexed_chats_for_user(bot, user_id)
    pref = await _db(db.get_dm_chat, conn, user_id)
    if pref is not None and pref in allowed:
        return [pref]
    if pref is not None:
        await _db(db.set_dm_chat, conn, user_id, None)
    return allowed


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
async def cmd_help(message: Message) -> None:
    await message.reply(
        "I answer questions from this chat's history.\n"
        "In a group, @mention me or reply to my messages. In DM, just ask.\n"
        "Commands: /ask <question>, /stats, /chats, /chat"
        "\nAdmins: /reindex (recent), /reindex full, /resolve (fix member names)"
    )


@dp.message(Command("ask"))
async def cmd_ask(message: Message, command, bot: Bot) -> None:
    if message.chat.type == ChatType.PRIVATE:
        allowed = await dm_scope(bot, message.from_user.id)
        if not allowed:
            await message.reply("You need to be a member of a chat I've indexed to ask me things.")
            return
        await respond(message, command.args or "", allowed)
        return
    await respond(message, command.args or "", message.chat.id)


@dp.message(Command("chats"))
async def cmd_chats(message: Message, bot: Bot) -> None:
    if message.chat.type != ChatType.PRIVATE:
        await message.reply("Use /chats in a DM to pick which group I search.")
        return
    allowed = await indexed_chats_for_user(bot, message.from_user.id)
    if not allowed:
        await message.reply("You need to be a member of a chat I've indexed.")
        return
    pref = await _db(db.get_dm_chat, conn, message.from_user.id)
    lines = []
    for i, cid in enumerate(allowed, 1):
        mark = " ←" if cid == pref else ""
        lines.append(f"{i}. {await _chat_title(bot, cid)} (`{cid}`){mark}")
    hint = "All your chats (use /chat N to focus)." if pref not in allowed else "Use /chat all to search every chat."
    await message.reply("Indexed chats you can ask about:\n" + "\n".join(lines) + "\n\n" + hint)


@dp.message(Command("chat"))
async def cmd_chat(message: Message, command, bot: Bot) -> None:
    if message.chat.type != ChatType.PRIVATE:
        await message.reply("Use /chat in a DM.")
        return
    allowed = await indexed_chats_for_user(bot, message.from_user.id)
    if not allowed:
        await message.reply("You need to be a member of a chat I've indexed.")
        return
    arg = (command.args or "").strip()
    if not arg:
        await message.reply("Pick a number from /chats, a chat id, or /chat all.")
        return
    if arg.lower() in ("all", "any", "*"):
        await _db(db.set_dm_chat, conn, message.from_user.id, None)
        await message.reply("I'll search all your indexed chats.")
        return
    chosen = None
    if arg.lstrip("-").isdigit():
        n = int(arg)
        if 1 <= n <= len(allowed):
            chosen = allowed[n - 1]
        elif n in allowed:
            chosen = n
    if chosen is None:
        await message.reply("Pick a number from /chats, a chat id, or /chat all.")
        return
    await _db(db.set_dm_chat, conn, message.from_user.id, chosen)
    await message.reply(f"Searching {await _chat_title(bot, chosen)}.")


@dp.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    s = await _db(db.stats, conn)
    await message.reply(
        f"messages: {s['messages']}\nwindows: {s['windows']}\n"
        f"embedded: {s['embedded']}\nchats: {s['chats']}"
    )


@dp.message(Command("reindex"))
async def cmd_reindex(message: Message, command) -> None:
    if message.from_user.id not in config.ADMIN_USER_IDS:
        await message.reply("Admins only.")
        return
    full = (command.args or "").strip().lower() == "full"
    if full:
        await message.reply("Full reindex…")
        result = await index_chats(None, full=True)
    else:
        await message.reply("Updating recent history…")
        result = await index_chats(None, lookback=config.UPDATE_LOOKBACK_DAYS)
    await message.reply(f"Done: {result['windows']} windows across {result['chats']} chat(s).")


@dp.message(Command("resolve"))
async def cmd_resolve(message: Message, bot: Bot) -> None:
    """Look up members' real names via the Bot API, replacing export labels.

    Run inside the group whose members you want to resolve. Only people still in
    the group can be looked up; anyone who left keeps their export label.
    """
    if message.from_user.id not in config.ADMIN_USER_IDS:
        await message.reply("Admins only.")
        return
    if message.chat.type == ChatType.PRIVATE:
        await message.reply("Run /resolve inside the group whose members to resolve.")
        return

    ids = await _db(
        lambda: [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT sender_id FROM messages WHERE chat_id=? AND sender_id IS NOT NULL",
                (message.chat.id,),
            )
        ]
    )
    await message.reply(f"Resolving {len(ids)} people via the API — this can take a while…")

    done = 0
    for uid in ids:
        try:
            member = await bot.get_chat_member(message.chat.id, uid)
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
    text = message.text or message.caption
    if not text:
        return

    await _ingest_group_message(message)

    if await _mentions_bot(message, bot):
        me = await bot.me()
        question = text.replace(f"@{me.username}", "").strip()
        await respond(message, question, message.chat.id)


@dp.edited_message(F.chat.type.in_({ChatType.GROUP, ChatType.SUPERGROUP}))
async def on_group_edit(message: Message) -> None:
    await _ingest_group_message(message, edited=True)


# --- Direct messages ------------------------------------------------------

@dp.message(F.chat.type == ChatType.PRIVATE)
async def on_private_message(message: Message, bot: Bot) -> None:
    if not message.text:
        return
    allowed = await dm_scope(bot, message.from_user.id)
    if not allowed:
        await message.reply("You need to be a member of a chat I've indexed to ask me things.")
        return
    await respond(message, message.text, allowed)


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
    log.info("bot is up")
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(logconfig.asyncio_handler)
    await _align_export_chat_ids(bot)
    _admin_errors.attach(loop, lambda text: _notify_admins(bot, text))
    await _notify_admins(bot, "Bot is up")


async def _on_shutdown(bot: Bot) -> None:
    log.info("bot is down")
    await _notify_admins(bot, "Bot is down")
    _admin_errors.detach()


dp.startup.register(_on_startup)
dp.shutdown.register(_on_shutdown)


async def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("set TELEGRAM_BOT_TOKEN (see .env.example)")
    bot = Bot(config.TELEGRAM_BOT_TOKEN)
    log.info("starting polling")
    asyncio.create_task(_periodic_lookback())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
