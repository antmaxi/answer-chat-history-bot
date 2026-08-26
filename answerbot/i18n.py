"""User-facing strings. Default UI language is Russian; /settings toggles EN."""

from __future__ import annotations

import random

DEFAULT_LANG = "ru"
SUPPORTED_LANGS: tuple[str, ...] = ("ru", "en")
LANG_NATIVE_NAME: dict[str, str] = {
    "ru": "Русский",
    "en": "English",
}
ADMIN_COMMANDS = frozenset({"stats", "reindex", "resolve"})

COMMAND_SPECS: dict[str, list[tuple[str, str]]] = {
    "ru": [
        ("ask", "Задать вопрос"),
        ("cancel", "Отменить поиск"),
        ("settings", "⚙️ Настройки"),
        ("stats", "Индекс и вопросы"),
        ("reindex", "Обновить индекс"),
        ("resolve", "Имена участников"),
        ("info", "ℹ️ О боте"),
        ("help", "Как пользоваться ботом"),
    ],
    "en": [
        ("ask", "Ask a question"),
        ("cancel", "Stop the current search"),
        ("settings", "⚙️ Settings"),
        ("stats", "Index and questions"),
        ("reindex", "Rebuild recent index"),
        ("resolve", "Fix member names"),
        ("info", "ℹ️ About the bot"),
        ("help", "How to use the bot"),
    ],
}

T: dict[str, dict[str, str]] = {
    "en": {
        "not_member": "You're not a member of the group this bot serves.",
        "help": (
            "I answer questions from the group's history.\n"
            "In the group, @mention me, reply to my messages, or /ask. In DM, just ask.\n"
            "Commands: /ask, /ask <question>, /cancel, /settings, /info"
        ),
        "help_admin": (
            "\nAdmins: /stats, /reindex (recent), /reindex full, /resolve (names; retry/stop)"
        ),
        "ask_empty": "Ask me a question about this chat's history.",
        "ask_prompt": "What is your question on the chat {name}?",
        "search_cancelled": "Search cancelled.",
        "nothing_to_cancel": "Nothing to cancel.",
        "cooldown": "Wait {wait} before asking again.",
        "quota_user": "Hourly limit reached. Try again in {wait}.",
        "quota_global": "The bot's hourly limit is reached. Try again in {wait}.",
        "wait_seconds": "{n}s",
        "wait_minutes": "{n} min",
        "answer_failed": "Something went wrong answering that.",
        "go_to_first": "Go to the first message",
        "sources": "Sources",
        "admins_only": "Admins only.",
        "reindex_full": "Full reindex…",
        "reindex_recent": "Updating recent history…",
        "reindex_done": "Done: {windows} windows across {chats} chat(s).",
        "resolve_start": (
            "Resolving <b>{n}</b> people in the background — Telegram rate-limits "
            "lookups, so this can take a while.\n"
            "/resolve — status · /resolve stop — pause · /resolve retry — try skipped people again"
        ),
        "resolve_running": (
            "Already running: <b>{done}</b> named, <b>{missed}</b> not in the group, "
            "<b>{pending}</b> left to try."
        ),
        "resolve_progress": (
            "Resolve: <b>{done}</b> named, <b>{missed}</b> not in the group, "
            "<b>{pending}</b> remaining…"
        ),
        "resolve_last": "Last: {last}",
        "resolve_done": (
            "Finished: <b>{done}</b> named, <b>{missed}</b> not in the group, "
            "<b>{failed}</b> errors. Still pending: <b>{pending}</b>.\n"
            "Run /reindex to rewrite history with new names. "
            "/resolve continues; /resolve retry tries skipped people again."
        ),
        "resolve_none": (
            "Nothing to look up. Named: <b>{resolved}</b>. "
            "Skipped (left / not found): <b>{missed}</b>. "
            "/resolve retry to try skipped people again."
        ),
        "resolve_stopped": (
            "Paused. <b>{done}</b> named, <b>{missed}</b> not in the group, "
            "<b>{pending}</b> remaining. /resolve to continue."
        ),
        "resolve_paused": (
            "Telegram asked to wait {wait}s — pausing so we don't burn the quota.\n"
            "<b>{done}</b> named so far, <b>{pending}</b> remaining. /resolve to continue."
        ),
        "resolve_usage": "Usage: /resolve, /resolve retry, /resolve stop",
        "stats": (
            "messages: {messages}\nwindows: {windows}\n"
            "embedded: {embedded}\nchats: {chats}"
        ),
        "stats_span": "\nfirst: {first}\nlast: {last}",
        "stats_queries": (
            "\n\nquestions:\n"
            "last day: {day} (admin: {day_admin}, others: {day_other})\n"
            "last week: {week} (admin: {week_admin}, others: {week_other})\n"
            "last month: {month} (admin: {month_admin}, others: {month_other})"
        ),
        "stats_latency": (
            "\n\nask time:\n"
            "last day: {day}\n"
            "last week: {week}\n"
            "last month: {month}"
        ),
        "stats_latency_range": "{median} ± {std} (min {min} / max {max})",
        "stats_latency_none": "n/a",
        "bot_starting": "Bot is starting",
        "bot_up": (
            "Bot started, stats:\n{db}: {messages} messages, {windows} windows{span}"
            "{latency}\n"
            "chat: {title} (`{chat_id}`)"
        ),
        "bot_down": "Bot is down",
        "settings_title": "⚙️ <b>Settings</b>",
        "settings_lang_label": "Language:",
        "settings_lang_btn": "🌐 {next_lang_label}",
        "lang_set": "🇬🇧 Language set to English.",
        "bot_name": "Chat History Bot",
        "info_msg": (
            "🤖 <b>{bot_name}</b>\n\n"
            "📅 <b>Last update:</b> {last_commit}\n"
            "🧠 <b>Model:</b> {model} ({provider}){retention}\n"
            "🔗 <b>Source code:</b> {github_repo}\n\n"
            "💬 Feel free to contact @antmaxi for suggestions on what to improve "
            "or if you run into issues with the bot."
        ),
        "info_retention_cursor": (
            "\n🔒 The model provider has a zero-retention policy: your question "
            "and the chat excerpts used to answer it are not stored after the "
            "request and are not used to train models."
        ),
        "unknown": "unknown",
    },
    "ru": {
        "not_member": "Вы не состоите в группе, которую обслуживает этот бот.",
        "help": (
            "Я отвечаю на вопросы по истории группы.\n"
            "В группе упомяните меня, ответьте на моё сообщение или /ask. В личке просто спросите.\n"
            "Команды: /ask, /ask <вопрос>, /cancel, /settings, /info"
        ),
        "help_admin": (
            "\nАдминам: /stats, /reindex (недавнее), /reindex full, /resolve (имена; retry/stop)"
        ),
        "ask_empty": "Задайте вопрос об истории этого чата.",
        "ask_prompt": "Какой у вас вопрос по чату {name}?",
        "search_cancelled": "Поиск отменён.",
        "nothing_to_cancel": "Сейчас нечего отменять.",
        "cooldown": "Подождите {wait}, прежде чем спросить снова.",
        "quota_user": "Часовой лимит исчерпан. Попробуйте снова через {wait}.",
        "quota_global": "Часовой лимит бота исчерпан. Попробуйте снова через {wait}.",
        "wait_seconds": "{n} с",
        "wait_minutes": "{n} мин",
        "answer_failed": "Не получилось ответить на этот вопрос.",
        "go_to_first": "К первому сообщению",
        "sources": "Источники",
        "admins_only": "Только для админов.",
        "reindex_full": "Полная переиндексация…",
        "reindex_recent": "Обновляю недавнюю историю…",
        "reindex_done": "Готово: {windows} окон в {chats} чат(ах).",
        "resolve_start": (
            "Уточняю <b>{n}</b> человек в фоне — Telegram ограничивает частоту запросов, "
            "это может занять время.\n"
            "/resolve — статус · /resolve stop — пауза · /resolve retry — повторить пропущенных"
        ),
        "resolve_running": (
            "Уже идёт: <b>{done}</b> имён, <b>{missed}</b> не в группе, "
            "осталось <b>{pending}</b>."
        ),
        "resolve_progress": (
            "Имена: <b>{done}</b> уточнено, <b>{missed}</b> не в группе, "
            "осталось <b>{pending}</b>…"
        ),
        "resolve_last": "Последнее: {last}",
        "resolve_done": (
            "Готово: <b>{done}</b> имён, <b>{missed}</b> не в группе, "
            "<b>{failed}</b> ошибок. Осталось: <b>{pending}</b>.\n"
            "Запустите /reindex, чтобы переписать историю. "
            "/resolve продолжит; /resolve retry — повторить пропущенных."
        ),
        "resolve_none": (
            "Некого уточнять. Имён: <b>{resolved}</b>. "
            "Пропущено (ушли / не найдены): <b>{missed}</b>. "
            "/resolve retry — повторить пропущенных."
        ),
        "resolve_stopped": (
            "Пауза. <b>{done}</b> имён, <b>{missed}</b> не в группе, "
            "осталось <b>{pending}</b>. /resolve чтобы продолжить."
        ),
        "resolve_paused": (
            "Telegram просит подождать {wait} с — ставлю на паузу, чтобы не сжечь лимит.\n"
            "Пока <b>{done}</b> имён, осталось <b>{pending}</b>. /resolve чтобы продолжить."
        ),
        "resolve_usage": "Использование: /resolve, /resolve retry, /resolve stop",
        "stats": (
            "сообщений: {messages}\nокон: {windows}\n"
            "с эмбеддингами: {embedded}\nчатов: {chats}"
        ),
        "stats_span": "\nпервое: {first}\nпоследнее: {last}",
        "stats_queries": (
            "\n\nвопросов:\n"
            "за сутки: {day} (админы: {day_admin}, остальные: {day_other})\n"
            "за неделю: {week} (админы: {week_admin}, остальные: {week_other})\n"
            "за месяц: {month} (админы: {month_admin}, остальные: {month_other})"
        ),
        "stats_latency": (
            "\n\nвремя запроса:\n"
            "за сутки: {day}\n"
            "за неделю: {week}\n"
            "за месяц: {month}"
        ),
        "stats_latency_range": "{median} ± {std} (мин {min} / макс {max})",
        "stats_latency_none": "нет данных",
        "bot_starting": "Бот запускается",
        "bot_up": (
            "Бот запущен, статистика:\n{db}: {messages} сообщений, {windows} окон{span}"
            "{latency}\n"
            "чат: {title} (`{chat_id}`)"
        ),
        "bot_down": "Бот остановлен",
        "settings_title": "⚙️ <b>Настройки</b>",
        "settings_lang_label": "Язык:",
        "settings_lang_btn": "🌐 {next_lang_label}",
        "lang_set": "🇷🇺 Язык установлен: Русский.",
        "bot_name": "Бот истории чата",
        "info_msg": (
            "🤖 <b>{bot_name}</b>\n\n"
            "📅 <b>Последнее обновление:</b> {last_commit}\n"
            "🧠 <b>Модель:</b> {model} ({provider}){retention}\n"
            "🔗 <b>Исходный код:</b> {github_repo}\n\n"
            "💬 Пишите @antmaxi с предложениями по улучшению бота или если что-то "
            "не работает."
        ),
        "info_retention_cursor": (
            "\n🔒 Провайдер модели соблюдает политику нулевого хранения данных: "
            "ваш вопрос и фрагменты чата, по которым строится ответ, не "
            "сохраняются после запроса и не используются для обучения моделей."
        ),
        "unknown": "неизвестно",
    },
}

# Overwritten in place on the placeholder reply while retrieve+LLM run.
THINKING: dict[str, tuple[str, ...]] = {
    "en": (
        "Thinking…",
        "Searching…",
        "Looking…",
        "Digging…",
        "Scanning…",
        "Reading…",
        "Checking…",
        "Hunting…",
        "Sifting…",
        "Browsing…",
    ),
    "ru": (
        "Думаю…",
        "Ищу…",
        "Смотрю…",
        "Копаюсь…",
        "Просматриваю…",
        "Читаю…",
        "Проверяю…",
        "Рыщу…",
        "Перебираю…",
        "Листаю…",
    ),
}


def normalize_lang(lang: str | None) -> str:
    if lang in T:
        return lang
    return DEFAULT_LANG


def next_ui_lang(current: str) -> str:
    current = normalize_lang(current)
    idx = SUPPORTED_LANGS.index(current)
    return SUPPORTED_LANGS[(idx + 1) % len(SUPPORTED_LANGS)]


def t(lang: str | None, key: str, **kwargs: object) -> str:
    lang = normalize_lang(lang)
    text = T[lang][key]
    if kwargs:
        return text.format(**kwargs)
    return text


def thinking_phrase(lang: str | None, previous: str = "") -> str:
    """Random waiting synonym in `lang`, different from `previous` when possible."""
    lang = normalize_lang(lang)
    choices = THINKING[lang]
    others = tuple(p for p in choices if p != previous)
    return random.choice(others or choices)


def settings_text(lang: str) -> str:
    lang = normalize_lang(lang)
    return (
        f"{t(lang, 'settings_title')}\n\n"
        f"{t(lang, 'settings_lang_label')} {LANG_NATIVE_NAME[lang]}"
    )
