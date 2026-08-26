"""The Telegram bot: aiogram wrapper around retrieve + answer, plus live ingest.

Pinned to one supergroup (`TELEGRAM_CHAT_ID`). Replies when @mentioned or when
someone replies to one of its own messages. Requires privacy mode OFF in
BotFather, or it receives no group messages to index at all.

DM behaviour: any plain message is a question, if the sender is a member of
that group. Search always runs against `TELEGRAM_CHAT_ID`.
"""

import asyncio
import logging
import socket
import threading
import time

from collections import defaultdict, deque

from aiogram import Bot, Dispatcher, F, html
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.session.middlewares.base import (
    BaseRequestMiddleware,
    NextRequestMiddlewareType,
)
from aiogram.enums import ChatType
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
)
from aiogram.filters import Command
from aiogram.methods.base import TelegramMethod, TelegramType
from aiogram.types import (
    BotCommand,
    BotCommandScopeChat,
    BotCommandScopeChatMember,
    BotCommandScopeDefault,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)

from . import adminlog, answer, config, cooldown, db, embed, followup, i18n, index, logconfig, membership, people, retrieve
from .info import format_info, format_latency, format_stats, last_update
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
_user_quota = cooldown.Quota(config.ANSWER_MAX_PER_USER_PER_HOUR)
_global_quota = cooldown.Quota(config.ANSWER_MAX_PER_HOUR)
_members = membership.MembershipCache(config.MEMBERSHIP_CACHE_SECONDS)
_history: dict[tuple[int, int], deque[str]] = defaultdict(lambda: deque(maxlen=4))
_admin_errors = adminlog.AdminErrorHandler()
_lang_cache: dict[int, str] = {}
# (chat_id, user_id) -> monotonic time /ask was issued with no question yet.
_pending_ask: dict[tuple[int, int], float] = {}
_ASK_PENDING_SECONDS = 5 * 60
# In-flight retrieve+answer tasks, cancelled by /cancel.
_in_flight: dict[tuple[int, int], set[asyncio.Task]] = defaultdict(set)
_chat_title = ""
_resolve_lock = asyncio.Lock()
_resolve_task: asyncio.Task | None = None
_resolve_progress: dict[str, int | str] = {}
_RESOLVE_PROGRESS_EVERY = 20


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


async def _refresh_chat_title(bot: Bot, message: Message | None = None) -> str:
    """Telegram title of TELEGRAM_CHAT_ID (same source as the startup stats DM)."""
    global _chat_title
    if (
        message is not None
        and message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
        and message.chat.id == _configured_chat()
        and message.chat.title
    ):
        _chat_title = message.chat.title
        return _chat_title
    try:
        chat = await bot.get_chat(_configured_chat())
        _chat_title = chat.title or str(_configured_chat())
    except Exception:
        if not _chat_title:
            _chat_title = str(_configured_chat())
    return _chat_title


def _ask_key(message: Message) -> tuple[int, int] | None:
    if message.from_user is None:
        return None
    return (message.chat.id, message.from_user.id)


def _cancel_pending_ask(message: Message) -> None:
    key = _ask_key(message)
    if key is not None:
        _pending_ask.pop(key, None)


def _arm_pending_ask(message: Message) -> None:
    key = _ask_key(message)
    if key is not None:
        _pending_ask[key] = time.monotonic()


def _pending_ask_ready(message: Message) -> bool:
    key = _ask_key(message)
    if key is None:
        return False
    started = _pending_ask.get(key)
    if started is None:
        return False
    if time.monotonic() - started > _ASK_PENDING_SECONDS:
        _pending_ask.pop(key, None)
        return False
    return True


def _track_in_flight(key: tuple[int, int]) -> asyncio.Task | None:
    task = asyncio.current_task()
    if task is not None:
        _in_flight[key].add(task)
    return task


def _untrack_in_flight(key: tuple[int, int], task: asyncio.Task | None) -> None:
    if task is None:
        return
    tasks = _in_flight.get(key)
    if not tasks:
        return
    tasks.discard(task)
    if not tasks:
        _in_flight.pop(key, None)


def _cancel_in_flight(key: tuple[int, int]) -> int:
    """Cancel this user's running searches. Returns how many tasks were cancelled."""
    tasks = _in_flight.pop(key, set())
    n = 0
    for task in tasks:
        if not task.done():
            task.cancel()
            n += 1
    return n


def _non_admin_in_flight() -> bool:
    """True if a non-admin retrieve+answer is running."""
    admins = config.ADMIN_USER_IDS
    for (_chat_id, user_id), tasks in _in_flight.items():
        if user_id in admins:
            continue
        if any(not t.done() for t in tasks):
            return True
    return False


def _span_lines(s: dict, lang: str) -> str:
    first, last = s.get("first_message"), s.get("last_message")
    if not first or not last:
        return ""
    return i18n.t(lang, "stats_span", first=first, last=last)


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


async def _lang_for(user_id: int | None) -> str:
    if not user_id:
        return i18n.DEFAULT_LANG
    cached = _lang_cache.get(user_id)
    if cached is not None:
        return cached
    lang = await _db(db.get_user_lang, conn, user_id)
    _lang_cache[user_id] = lang
    return lang


async def _set_lang(user_id: int, lang: str) -> str:
    lang = await _db(db.set_user_lang, conn, user_id, lang)
    _lang_cache[user_id] = lang
    return lang


def commands_for_user(lang: str, user_id: int) -> list[BotCommand]:
    lang = i18n.normalize_lang(lang)
    cmds = [BotCommand(command=name, description=desc) for name, desc in i18n.COMMAND_SPECS[lang]]
    if user_id not in config.ADMIN_USER_IDS:
        cmds = [c for c in cmds if c.command not in i18n.ADMIN_COMMANDS]
    return cmds


def _settings_keyboard(lang: str) -> InlineKeyboardMarkup:
    next_label = i18n.LANG_NATIVE_NAME[i18n.next_ui_lang(lang)]
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=i18n.t(lang, "settings_lang_btn", next_lang_label=next_label),
                    callback_data="settings:toggle_lang",
                )
            ]
        ]
    )


async def set_user_commands(bot: Bot, chat, user) -> None:
    """Push the command menu for this user in their saved language."""
    if chat is None or user is None:
        return
    lang = await _lang_for(user.id)
    menu = commands_for_user(lang, user.id)
    try:
        if chat.type == ChatType.PRIVATE:
            scope: BotCommandScopeChat | BotCommandScopeChatMember = BotCommandScopeChat(
                chat_id=chat.id
            )
        else:
            scope = BotCommandScopeChatMember(chat_id=chat.id, user_id=user.id)
        await bot.delete_my_commands(scope=scope)
        await bot.set_my_commands(menu, scope=scope)
    except Exception:
        log.warning("Could not set commands for user %s", user.id, exc_info=True)


async def _ensure_member(message: Message, bot: Bot) -> bool:
    """Allow the configured group, or a DM from a current member. Otherwise decline."""
    user = message.from_user
    if user is None:
        return False
    if message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        return _is_configured_chat(message.chat.id)
    if await _user_in_configured_chat(bot, user.id):
        return True
    lang = await _lang_for(user.id)
    await message.reply(i18n.t(lang, "not_member"))
    return False


async def _ensure_member_callback(query: CallbackQuery, bot: Bot) -> bool:
    user = query.from_user
    if user is None:
        return False
    chat = query.message.chat if query.message else None
    if chat is not None and chat.type in (ChatType.GROUP, ChatType.SUPERGROUP):
        return _is_configured_chat(chat.id)
    if await _user_in_configured_chat(bot, user.id):
        return True
    lang = await _lang_for(user.id)
    await query.answer(i18n.t(lang, "not_member"), show_alert=True)
    return False


def format_answer(result: answer.Answer, lang: str) -> str:
    # Markdown → Telegram HTML, then [W#] → t.me links (brackets survive escaping).
    body = answer.format_answer_body(result)

    # A direct jump to the first message the answer is grounded in.
    link = result.primary_link()
    if link:
        body += f'\n\n➡️ <a href="{link}">{i18n.t(lang, "go_to_first")}</a>'

    sources = result.grouped_sources()
    if sources:
        lines = "\n".join(
            f'<a href="{h.link()}">{" ".join(f"[W{i}]" for i in idxs)}</a>'
            f'{" ✓" if was_cited else ""} '
            f'{html.quote(h.when())} · {html.quote(h.speakers)}'
            for idxs, h, was_cited in sources
        )
        body += f'\n\n<b>{i18n.t(lang, "sources")}</b>\n' + lines
    return body


def _record_person(sender_id, name, username, source="live") -> None:
    people.record(conn, sender_id, name, username, source)
    conn.commit()


def _mark_miss(sender_id, reason="left") -> None:
    people.mark_miss(conn, sender_id, reason)
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


def _fmt_quota_wait(seconds: float, lang: str) -> str:
    if seconds < 60:
        return i18n.t(lang, "wait_seconds", n=int(seconds) + 1)
    return i18n.t(lang, "wait_minutes", n=(int(seconds) + 59) // 60)


def _quota_block(user_id: int, lang: str) -> str | None:
    """If this LLM call would exceed a quota, the reply to send; else consume a slot."""
    exempt = user_id in config.ADMIN_USER_IDS
    wait = _user_quota.remaining((user_id,), exempt=exempt)
    if wait > 0:
        return i18n.t(lang, "quota_user", wait=_fmt_quota_wait(wait, lang))
    wait = _global_quota.remaining((), exempt=exempt)
    if wait > 0:
        return i18n.t(lang, "quota_global", wait=_fmt_quota_wait(wait, lang))
    if not exempt:
        _user_quota.touch((user_id,))
        _global_quota.touch(())
    return None


_THINKING_INTERVAL = 2.5


def _status_html(text: str) -> str:
    return f"<i>{html.quote(text)}</i>"


async def _spin_thinking(
    msg: Message, lang: str, stop: asyncio.Event, last: str
) -> None:
    """Overwrite the placeholder with a fresh synonym until `stop` is set."""
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=_THINKING_INTERVAL)
            return
        except TimeoutError:
            pass
        nxt = i18n.thinking_phrase(lang, last)
        try:
            await msg.edit_text(_status_html(nxt), parse_mode="HTML")
            last = nxt
        except Exception:
            log.debug("thinking status edit failed", exc_info=True)


async def _stop_thinking(stop: asyncio.Event, task: asyncio.Task | None) -> None:
    stop.set()
    if task is None:
        return
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _abandon_search(
    thinking: Message | None,
    stop: asyncio.Event,
    spinner: asyncio.Task | None,
    lang: str,
) -> None:
    """Stop the spinner and drop the placeholder so a cancelled search leaves no status."""
    await _stop_thinking(stop, spinner)
    if thinking is None:
        return
    try:
        await thinking.delete()
    except Exception:
        try:
            await thinking.edit_text(
                _status_html(i18n.t(lang, "search_cancelled")), parse_mode="HTML"
            )
        except Exception:
            log.debug("cancel status update failed", exc_info=True)


async def respond(message: Message, question: str, chat_id: int | list[int]) -> None:
    user_id = message.from_user.id if message.from_user else 0
    lang = await _lang_for(user_id)
    if not question.strip():
        await message.reply(i18n.t(lang, "ask_empty"))
        return
    wait = _answers.remaining(
        (user_id, message.chat.id),
        exempt=user_id in config.ADMIN_USER_IDS,
    )
    if wait > 0:
        await message.reply(
            i18n.t(lang, "cooldown", wait=_fmt_quota_wait(wait, lang))
        )
        return
    _answers.touch((user_id, message.chat.id))

    key = (message.chat.id, user_id)
    thinking: Message | None = None
    spinner: asyncio.Task | None = None
    stop = asyncio.Event()
    tracked = _track_in_flight(key)
    t0 = time.monotonic()
    try:
        prior = _history[key][-1] if _history[key] else None
        force = False
        reply = message.reply_to_message
        if reply and reply.from_user:
            me = await message.bot.me()
            force = reply.from_user.id == me.id
        search_q = followup.rewrite(question, prior, force=force)

        phrase = i18n.thinking_phrase(lang)
        thinking = await message.reply(_status_html(phrase), parse_mode="HTML")
        spinner = asyncio.create_task(_spin_thinking(thinking, lang, stop, phrase))
        # Search the last scheduled index. Live ingest + periodic lookback
        # (and /reindex) refresh windows; doing that on every question pegs CPU.
        hits = await _db(retrieve.search, conn, search_q, chat_id)
        if hits:
            blocked = _quota_block(user_id, lang)
            if blocked:
                await _stop_thinking(stop, spinner)
                spinner = None
                await thinking.edit_text(blocked)
                return

        def complete():
            return answer.complete_answer(question, hits)

        result = await asyncio.to_thread(complete)
        await _db(answer._record, conn, question, chat_id, result, t0, None, user_id or None)
        _history[key].append(question)
        await _stop_thinking(stop, spinner)
        spinner = None
        await thinking.edit_text(
            format_answer(result, lang), parse_mode="HTML", disable_web_page_preview=True
        )
    except asyncio.CancelledError:
        log.info("search cancelled user=%s chat=%s", user_id, message.chat.id)
        await _abandon_search(thinking, stop, spinner, lang)
        spinner = None
        return
    except Exception:
        log.exception("failed to answer")
        await _stop_thinking(stop, spinner)
        spinner = None
        if thinking is not None:
            await thinking.edit_text(i18n.t(lang, "answer_failed"))
    finally:
        await _stop_thinking(stop, spinner)
        _untrack_in_flight(key, tracked)


async def _consume_pending_ask(message: Message) -> bool:
    """If this user was prompted by bare /ask, treat the message as the question."""
    if not _pending_ask_ready(message):
        return False
    text = (message.text or message.caption or "").strip()
    if not text:
        return False
    _cancel_pending_ask(message)
    await respond(message, text, _configured_chat())
    return True


# --- Commands -------------------------------------------------------------

@dp.message(Command("start", "help"))
async def cmd_help(message: Message, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    _cancel_pending_ask(message)
    await set_user_commands(bot, message.chat, message.from_user)
    lang = await _lang_for(message.from_user.id if message.from_user else None)
    text = i18n.t(lang, "help")
    if message.from_user and message.from_user.id in config.ADMIN_USER_IDS:
        text += i18n.t(lang, "help_admin")
    await message.reply(text)


@dp.message(Command("info"))
async def cmd_info(message: Message, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    _cancel_pending_ask(message)
    lang = await _lang_for(message.from_user.id if message.from_user else None)
    s = await _db(db.stats, conn)
    await message.reply(
        format_info(last_update(), lang, stats=s),
        parse_mode="HTML",
        disable_web_page_preview=True,
    )


@dp.message(Command("settings"))
async def cmd_settings(message: Message, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    _cancel_pending_ask(message)
    lang = await _lang_for(message.from_user.id if message.from_user else None)
    await message.reply(
        i18n.settings_text(lang),
        reply_markup=_settings_keyboard(lang),
        parse_mode="HTML",
    )


@dp.callback_query(F.data == "settings:toggle_lang")
async def settings_toggle_lang(query: CallbackQuery, bot: Bot) -> None:
    if not await _ensure_member_callback(query, bot):
        return
    user = query.from_user
    current = await _lang_for(user.id)
    lang = await _set_lang(user.id, i18n.next_ui_lang(current))
    if isinstance(query.message, Message):
        await set_user_commands(bot, query.message.chat, user)
    await query.answer(i18n.t(lang, "lang_set"))
    if isinstance(query.message, Message):
        await query.message.edit_text(
            i18n.settings_text(lang),
            reply_markup=_settings_keyboard(lang),
            parse_mode="HTML",
        )


@dp.message(Command("ask"))
async def cmd_ask(message: Message, command, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    question = (command.args or "").strip()
    if question:
        _cancel_pending_ask(message)
        await respond(message, question, _configured_chat())
        return
    _arm_pending_ask(message)
    lang = await _lang_for(message.from_user.id if message.from_user else None)
    name = await _refresh_chat_title(bot, message)
    await message.reply(
        i18n.t(lang, "ask_prompt", name=html.quote(name)),
        parse_mode="HTML",
    )


@dp.message(Command("cancel"))
async def cmd_cancel(message: Message, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    lang = await _lang_for(message.from_user.id if message.from_user else None)
    key = _ask_key(message)
    had_pending = False
    n = 0
    if key is not None:
        had_pending = _pending_ask.pop(key, None) is not None
        n = _cancel_in_flight(key)
    if n or had_pending:
        await message.reply(i18n.t(lang, "search_cancelled"))
    else:
        await message.reply(i18n.t(lang, "nothing_to_cancel"))


@dp.message(Command("stats"))
async def cmd_stats(message: Message, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    _cancel_pending_ask(message)
    lang = await _lang_for(message.from_user.id if message.from_user else None)
    if not message.from_user or message.from_user.id not in config.ADMIN_USER_IDS:
        await message.reply(i18n.t(lang, "admins_only"))
        return
    s = await _db(db.stats, conn)
    s["user_in_use"] = _non_admin_in_flight()
    await message.reply(format_stats(s, lang, questions=True))


@dp.message(Command("reindex"))
async def cmd_reindex(message: Message, command, bot: Bot) -> None:
    if not await _ensure_member(message, bot):
        return
    _cancel_pending_ask(message)
    lang = await _lang_for(message.from_user.id if message.from_user else None)
    if message.from_user.id not in config.ADMIN_USER_IDS:
        await message.reply(i18n.t(lang, "admins_only"))
        return
    chat_id = _configured_chat()
    full = (command.args or "").strip().lower() == "full"
    if full:
        await message.reply(i18n.t(lang, "reindex_full"))
        result = await index_chats(chat_id, full=True)
    else:
        await message.reply(i18n.t(lang, "reindex_recent"))
        result = await index_chats(chat_id, lookback=config.UPDATE_LOOKBACK_DAYS)
    await message.reply(
        i18n.t(lang, "reindex_done", windows=result["windows"], chats=result["chats"])
    )


def _resolve_html(lang: str, key: str, *, last_name: str = "", **kwargs) -> str:
    """Resolve status HTML. Names are quoted so emoji survive and markup in a name cannot break parse_mode."""
    text = i18n.t(lang, key, **kwargs)
    if last_name:
        text += "\n" + i18n.t(lang, "resolve_last", last=html.quote(last_name))
    return text


def _resolve_snapshot() -> dict:
    return {
        "done": int(_resolve_progress.get("done") or 0),
        "missed": int(_resolve_progress.get("missed") or 0),
        "failed": int(_resolve_progress.get("failed") or 0),
        "pending": int(_resolve_progress.get("pending") or 0),
    }


async def _edit_resolve_status(msg: Message, html_text: str) -> None:
    try:
        await msg.edit_text(html_text, parse_mode="HTML")
    except TelegramBadRequest:
        pass
    except Exception:
        log.debug("resolve status edit failed", exc_info=True)


async def _resolve_one(bot: Bot, chat_id: int, uid: int) -> tuple[str, str]:
    """Look up one member. Returns (ok|miss|error|pause, display_name_or_empty)."""
    while True:
        try:
            member = await bot.get_chat_member(chat_id, uid)
            user = member.user
            name = people.name_from_user(user)
            if not name:
                await _db(_mark_miss, uid, "left")
                return "miss", ""
            await _db(_record_person, uid, name, user.username, "api")
            return "ok", name
        except TelegramRetryAfter as e:
            wait = max(float(getattr(e, "retry_after", 1) or 1), 1.0) + 0.25
            if wait > config.RESOLVE_MAX_FLOOD_WAIT:
                _resolve_progress["pause_wait"] = int(wait)
                log.warning("resolve: flood wait %.0fs too long; pausing", wait)
                return "pause", ""
            log.info("resolve: flood wait %.1fs (user %s)", wait, uid)
            await asyncio.sleep(wait)
        except (TelegramBadRequest, TelegramForbiddenError) as e:
            await _db(_mark_miss, uid, people.miss_reason_from_error(str(e)))
            return "miss", ""
        except Exception:
            log.warning("resolve: lookup failed user=%s", uid, exc_info=True)
            return "error", ""


async def _resolve_job(bot: Bot, chat_id: int, ids: list[int], status_msg: Message, lang: str) -> None:
    done = missed = failed = 0
    pending = len(ids)
    last_name = ""
    last_edit = 0.0
    paused = False
    _resolve_progress.update(done=0, missed=0, failed=0, pending=pending, last_name="")

    async def publish(key: str, **extra) -> None:
        await _edit_resolve_status(
            status_msg,
            _resolve_html(lang, key, last_name=last_name, **extra),
        )

    try:
        for i, uid in enumerate(ids):
            outcome, name = await _resolve_one(bot, chat_id, uid)
            if outcome == "pause":
                paused = True
                pending = len(ids) - i
                _resolve_progress["pending"] = pending
                wait = int(_resolve_progress.get("pause_wait") or 0)
                await publish("resolve_paused", wait=wait, done=done, pending=pending)
                return
            if outcome == "ok":
                done += 1
                last_name = name
            elif outcome == "miss":
                missed += 1
            else:
                failed += 1
            pending = len(ids) - i - 1
            _resolve_progress.update(
                done=done, missed=missed, failed=failed, pending=pending, last_name=last_name
            )
            now = time.monotonic()
            if (i + 1) % _RESOLVE_PROGRESS_EVERY == 0 or now - last_edit >= 15:
                await publish("resolve_progress", done=done, missed=missed, pending=pending)
                last_edit = now
            await asyncio.sleep(config.RESOLVE_DELAY_SECONDS)
        await publish("resolve_done", done=done, missed=missed, failed=failed, pending=pending)
    except asyncio.CancelledError:
        _resolve_progress.update(done=done, missed=missed, failed=failed, pending=pending)
        await publish("resolve_stopped", done=done, missed=missed, pending=pending)
        raise
    finally:
        log.info(
            "resolve finished chat_id=%s named=%s missed=%s failed=%s pending=%s paused=%s",
            chat_id, done, missed, failed, pending, paused,
        )


@dp.message(Command("resolve"))
async def cmd_resolve(message: Message, command, bot: Bot) -> None:
    """Background getChatMember backfill. Resumes across runs; honours flood waits.

    Only people the API can see get a name (current members if the bot is an
    admin). Anyone who left is remembered as a miss and skipped next time.
    """
    if not await _ensure_member(message, bot):
        return
    _cancel_pending_ask(message)
    lang = await _lang_for(message.from_user.id if message.from_user else None)
    if not message.from_user or message.from_user.id not in config.ADMIN_USER_IDS:
        await message.reply(i18n.t(lang, "admins_only"))
        return

    global _resolve_task
    arg = (command.args or "").strip().lower()
    if arg not in ("", "continue", "retry", "full", "stop", "cancel"):
        await message.reply(i18n.t(lang, "resolve_usage"))
        return

    chat_id = _configured_chat()
    async with _resolve_lock:
        running = _resolve_task is not None and not _resolve_task.done()
        if arg in ("stop", "cancel"):
            if running:
                _resolve_task.cancel()
            else:
                await message.reply(i18n.t(lang, "nothing_to_cancel"))
            return
        if running:
            snap = _resolve_snapshot()
            last = str(_resolve_progress.get("last_name") or "")
            await message.reply(
                _resolve_html(lang, "resolve_running", last_name=last, **snap),
                parse_mode="HTML",
            )
            return

        retry = arg in ("retry", "full")
        ids = await _db(people.pending_ids, conn, chat_id, retry)
        if not ids:
            stats = await _db(people.lookup_stats, conn, chat_id)
            await message.reply(
                i18n.t(lang, "resolve_none", resolved=stats["resolved"], missed=stats["missed"]),
                parse_mode="HTML",
            )
            return

        status_msg = await message.reply(
            i18n.t(lang, "resolve_start", n=len(ids)),
            parse_mode="HTML",
        )
        _resolve_task = asyncio.create_task(_resolve_job(bot, chat_id, ids, status_msg, lang))


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
        people.name_from_user(message.from_user) or "Unknown",
        message.from_user.id if message.from_user else None,
        int(message.date.timestamp()),
        text,
        message.reply_to_message.message_id if message.reply_to_message else None,
    )
    # The sender's own full name is their real public name — record it so it
    # overrides any private export label on the next reindex.
    if message.from_user:
        u = message.from_user
        await _db(_record_person, u.id, people.name_from_user(u), u.username)
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

    if await _consume_pending_ask(message):
        return

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
    if await _consume_pending_ask(message):
        return
    await respond(message, message.text, _configured_chat())


async def _dm_admin(bot: Bot, uid: int, text: str) -> None:
    """DM one admin. Telegram only delivers if they have already /start'd the bot."""
    try:
        await bot.send_message(uid, text)
    except TelegramForbiddenError:
        log.warning("admin %s has not started a chat with the bot; cannot send %r", uid, text)
    except Exception:
        log.warning("failed to notify admin %s (%s)", uid, text, exc_info=True)


async def _notify_admins(bot: Bot, text: str) -> None:
    for uid in sorted(config.ADMIN_USER_IDS):
        await _dm_admin(bot, uid, text)


async def _notify_status(bot: Bot, key: str, **kwargs) -> None:
    for uid in sorted(config.ADMIN_USER_IDS):
        lang = await _lang_for(uid)
        await _dm_admin(bot, uid, i18n.t(lang, key, **kwargs))


async def _on_startup(bot: Bot) -> None:
    global _chat_title
    log.info("bot is starting")
    await _notify_status(bot, "bot_starting")
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(logconfig.asyncio_handler)
    await _align_export_chat_ids()
    try:
        chat = await bot.get_chat(_configured_chat())
        title = chat.title or str(_configured_chat())
    except Exception:
        title = str(_configured_chat())
        log.warning("cannot reach TELEGRAM_CHAT_ID=%s", _configured_chat())
    _chat_title = title
    log.info("configured chat %s (%s)", _configured_chat(), title)
    _admin_errors.attach(loop, lambda text: _notify_admins(bot, text))
    try:
        await bot.set_my_commands(
            commands_for_user(i18n.DEFAULT_LANG, user_id=0),
            scope=BotCommandScopeDefault(),
        )
    except Exception:
        log.warning("Could not set default commands", exc_info=True)
    for admin_id in config.ADMIN_USER_IDS:
        lang = await _lang_for(admin_id)
        try:
            scope = BotCommandScopeChat(chat_id=admin_id)
            await bot.delete_my_commands(scope=scope)
            await bot.set_my_commands(commands_for_user(lang, admin_id), scope=scope)
        except Exception:
            log.warning("Could not set admin commands for user %s", admin_id, exc_info=True)
    log.info("warming embedding model (%s threads)", config.EMBED_THREADS)
    try:
        await asyncio.to_thread(embed.warmup)
        log.info("embedding model ready")
    except Exception:
        log.exception("embedding warmup failed; the first search will load the model")
    s = db.stats(conn)
    log.info("bot is up; database %s: %s", config.DB_PATH, s)
    for uid in sorted(config.ADMIN_USER_IDS):
        lang = await _lang_for(uid)
        await _dm_admin(
            bot,
            uid,
            i18n.t(
                lang,
                "bot_up",
                db=config.DB_PATH,
                messages=s["messages"],
                windows=s["windows"],
                span=_span_lines(s, lang),
                latency=format_latency(s, lang),
                title=title,
                chat_id=_configured_chat(),
            ),
        )


async def _on_shutdown(bot: Bot) -> None:
    log.info("bot is down")
    task = _resolve_task
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    await _notify_status(bot, "bot_down")
    _admin_errors.detach()
    await _db(db.checkpoint, conn)


dp.startup.register(_on_startup)
dp.shutdown.register(_on_shutdown)


# Long-poll getUpdates holds a TCP socket to api.telegram.org. Telegram, Docker
# NAT, and broken IPv6 routes all drop that socket; aiohttp then raises
# ClientOSError 104 (connection reset). aiogram already retries; we just need a
# fresh connector and to stop treating the blip as an application error.
_POLL_TIMEOUT = 20
_SESSION_TIMEOUT = 70.0


class TelegramNetworkGuard(BaseRequestMiddleware):
    """Drop a dead aiohttp session after a reset so the next call opens a new socket.

    getUpdates is retried by the dispatcher; other methods get one extra attempt.
    """

    async def __call__(
        self,
        make_request: NextRequestMiddlewareType[TelegramType],
        bot: Bot,
        method: TelegramMethod[TelegramType],
    ):
        try:
            return await make_request(bot, method)
        except TelegramNetworkError:
            session = bot.session
            if hasattr(session, "_should_reset_connector"):
                session._should_reset_connector = True
            if getattr(method, "__api_method__", "") == "getUpdates":
                raise
            log.warning(
                "Telegram %s: connection dropped; retrying once",
                type(method).__name__,
            )
            return await make_request(bot, method)


def _telegram_session() -> AiohttpSession:
    session = AiohttpSession(timeout=_SESSION_TIMEOUT)
    # Telegram's IPv6 is a common source of getUpdates resets; stay on IPv4.
    session._connector_init["family"] = socket.AF_INET
    # Recycle idle sockets before typical Docker/NAT timeouts (~30–60s).
    session._connector_init["keepalive_timeout"] = 20
    session.middleware(TelegramNetworkGuard())
    return session


async def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("set TELEGRAM_BOT_TOKEN (see .env.example)")
    if config.TELEGRAM_CHAT_ID is None:
        raise SystemExit("set TELEGRAM_CHAT_ID (the supergroup's Bot API id, see .env.example)")
    bot = Bot(config.TELEGRAM_BOT_TOKEN, session=_telegram_session())
    log.info("starting polling")
    asyncio.create_task(_periodic_lookback())
    await dp.start_polling(bot, polling_timeout=_POLL_TIMEOUT)


if __name__ == "__main__":
    asyncio.run(main())
