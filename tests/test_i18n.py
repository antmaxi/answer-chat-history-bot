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

    def test_help_mentions_settings(self):
        assert "/settings" in i18n.t("ru", "help")
        assert "/settings" in i18n.t("en", "help")

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

    def test_quota_wait_format(self):
        assert i18n.t("ru", "wait_seconds", n=5) == "5 с"
        assert i18n.t("en", "wait_seconds", n=5) == "5s"
        assert i18n.t("ru", "wait_minutes", n=3) == "3 мин"
        assert i18n.t("en", "wait_minutes", n=3) == "3 min"


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
            assert cmds[-2:] == ["help", "info"]
