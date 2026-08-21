"""Isolate tests from the developer's .env (e.g. SPEAKER_LABEL=id)."""

import pytest

from answerbot import config


@pytest.fixture(autouse=True)
def _default_speaker_label(monkeypatch):
    monkeypatch.setattr(config, "SPEAKER_LABEL", "name")
