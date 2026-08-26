"""UI language: Russian default, /settings toggles EN."""

from answerbot import i18n


class TestNormalizeLang:
    def test_default_is_russian(self):
        assert i18n.DEFAULT_LANG == "ru"
        assert i18n.normalize_lang(None) == "ru"
        assert i18n.normalize_lang("de") == "ru"
        assert i18n.normalize_lang("en") == "en"
        assert i18n.normalize_lang("ru") == "ru"


class TestNextUiLang:
    def test_toggles_ru_en(self):
        assert i18n.next_ui_lang("ru") == "en"
        assert i18n.next_ui_lang("en") == "ru"
        assert i18n.next_ui_lang("de") == "en"


class TestStrings:
    def test_en_and_ru_have_the_same_keys(self):
        assert set(i18n.T["en"]) == set(i18n.T["ru"])

    def test_help_omits_admin_commands(self):
        for lang in ("ru", "en"):
            text = i18n.t(lang, "help")
            assert "/stats" not in text
            assert "/reindex" not in text
            assert "/who" not in text
            assert "/settings" in text
            assert "/info" in text
            assert "/cancel" in text
        assert "/stats" in i18n.t("en", "help_admin")
        assert "/stats <a> <b>" in i18n.t("en", "help_admin")
        assert "/who" in i18n.t("en", "help_admin")

    def test_settings_text_shows_current_language(self):
        ru = i18n.settings_text("ru")
        assert "Настройки" in ru
        assert "Русский" in ru
        en = i18n.settings_text("en")
        assert "Settings" in en
        assert "English" in en

    def test_lang_set_toast_matches_the_new_language(self):
        assert "Русский" in i18n.t("ru", "lang_set")
        assert "English" in i18n.t("en", "lang_set")

    def test_ask_prompt_includes_chat_name(self):
        assert i18n.t("en", "ask_prompt", name="RUPR") == (
            "What is your question on the chat RUPR?"
        )
        assert i18n.t("ru", "ask_prompt", name="RUPR") == (
            "Какой у вас вопрос по чату RUPR?"
        )
        assert "other indexed chats" in i18n.t("en", "ask_prompt_all", name="RUPR")
        assert "другим проиндексированным" in i18n.t(
            "ru", "ask_prompt_all", name="RUPR"
        )

    def test_quota_wait_format(self):
        assert i18n.t("ru", "wait_seconds", n=5) == "5 с"
        assert i18n.t("en", "wait_seconds", n=5) == "5s"
        assert i18n.t("ru", "wait_minutes", n=3) == "3 мин"
        assert i18n.t("en", "wait_minutes", n=3) == "3 min"

    def test_cancel_feedback(self):
        assert i18n.t("en", "search_cancelled") == "Search cancelled."
        assert i18n.t("ru", "search_cancelled") == "Поиск отменён."
        assert i18n.t("en", "nothing_to_cancel") == "Nothing to cancel."
        assert i18n.t("ru", "nothing_to_cancel") == "Сейчас нечего отменять."

    def test_admin_startup_dms(self):
        assert i18n.t("en", "bot_starting") == "Bot is starting"
        assert i18n.t("ru", "bot_starting") == "Бот запускается"
        en = i18n.t(
            "en",
            "bot_up",
            db="chat.db",
            messages=10,
            windows=3,
            span="",
            latency="",
            title="RUPR",
            chat_id=-100,
        )
        assert en.startswith("Bot started, stats:")
        assert "chat.db: 10 messages, 3 windows" in en
        ru = i18n.t(
            "ru",
            "bot_up",
            db="chat.db",
            messages=10,
            windows=3,
            span="",
            latency="",
            title="RUPR",
            chat_id=-100,
        )
        assert ru.startswith("Бот запущен, статистика:")
        extra = i18n.t("en", "bot_up_more", chats="Offtopic (`-1002`)")
        assert extra.startswith("\nalso indexing:")
        assert "Offtopic" in extra


class TestThinking:
    def test_both_languages_have_matching_phrase_counts(self):
        assert set(i18n.THINKING) == set(i18n.SUPPORTED_LANGS)
        counts = {lang: len(i18n.THINKING[lang]) for lang in i18n.SUPPORTED_LANGS}
        assert len(set(counts.values())) == 1
        assert next(iter(counts.values())) >= 2

    def test_phrase_comes_from_the_language_list(self):
        for lang in i18n.SUPPORTED_LANGS:
            assert i18n.thinking_phrase(lang) in i18n.THINKING[lang]
        assert i18n.thinking_phrase("de") in i18n.THINKING[i18n.DEFAULT_LANG]

    def test_avoids_the_previous_phrase(self):
        for lang in i18n.SUPPORTED_LANGS:
            for prev in i18n.THINKING[lang]:
                for _ in range(20):
                    assert i18n.thinking_phrase(lang, prev) != prev


class TestProgress:
    def test_elapsed_format(self):
        assert i18n.fmt_elapsed(0) == "0:00"
        assert i18n.fmt_elapsed(12.9) == "0:12"
        assert i18n.fmt_elapsed(65) == "1:05"
        assert i18n.fmt_elapsed(3723) == "1:02:03"

    def test_bar_and_status(self):
        assert i18n.progress_bar(0) == "░░░░░░░░░░"
        assert i18n.progress_bar(100) == "██████████"
        assert i18n.progress_bar(50).count("█") == 5
        assert i18n.progress_status("Head", None, "0:08") == "Head\n0:08"
        text = i18n.progress_status("Head", 42, "1:05")
        assert text.startswith("Head\n")
        assert "42%" in text
        assert "1:05" in text

    def test_reindex_pct(self):
        assert i18n.reindex_pct(0, None) == 0
        assert i18n.reindex_pct(0, 0) == 100
        assert i18n.reindex_pct(25, 100) == 25
        assert i18n.reindex_pct(100, 100) == 100

    def test_estimated_pct_only_when_typical_is_long_enough(self):
        assert i18n.estimated_pct(5, None) is None
        assert i18n.estimated_pct(5, 2.0) is None
        assert i18n.estimated_pct(4, 8.0) == 50
        assert i18n.estimated_pct(80, 8.0) == 95

    def test_reindex_done_includes_elapsed(self):
        en = i18n.t("en", "reindex_done", windows=3, chats=1, elapsed="1:05")
        assert "3 windows" in en
        assert "1:05" in en
        ru = i18n.t("ru", "reindex_done", windows=3, chats=1, elapsed="1:05")
        assert "3 окон" in ru
        assert "1:05" in ru

    def test_reindex_failed(self):
        assert i18n.t("en", "reindex_failed") == "Something went wrong updating the index."
        assert i18n.t("ru", "reindex_failed") == "Не получилось обновить индекс."


class TestCommandSpecs:
    def test_same_commands_in_both_languages(self):
        ru = [name for name, _ in i18n.COMMAND_SPECS["ru"]]
        en = [name for name, _ in i18n.COMMAND_SPECS["en"]]
        assert ru == en

    def test_settings_and_info_are_listed(self):
        for lang in i18n.SUPPORTED_LANGS:
            cmds = [name for name, _ in i18n.COMMAND_SPECS[lang]]
            assert "settings" in cmds
            assert "info" in cmds
            assert "help" in cmds
            visible = [c for c in cmds if c not in i18n.ADMIN_COMMANDS]
            assert visible == ["ask", "cancel", "settings", "info", "help"]
            assert "stats" in i18n.ADMIN_COMMANDS
            assert "who" in i18n.ADMIN_COMMANDS
