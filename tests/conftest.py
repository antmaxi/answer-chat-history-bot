"""Isolate tests from the developer's .env (e.g. SPEAKER_LABEL=id)."""

import pytest

from answerbot import config


@pytest.fixture(autouse=True)
def _default_speaker_label(monkeypatch):
    monkeypatch.setattr(config, "SPEAKER_LABEL", "name")
    # Production caps excerpts at MAX_K; tests assert ranking, not the cap.
    monkeypatch.setattr(config, "MIN_K", 1)
    monkeypatch.setattr(config, "MAX_K", config.TOP_K)
    monkeypatch.setattr(config, "COSINE_MIN", 0.0)
    monkeypatch.setattr(config, "RECENCY_HALF_LIFE_DAYS", 0.0)
    # FAQ fetch is network; tests that need it turn it on and stub ingest.
    monkeypatch.setattr(config, "FAQ_ENABLED", False)
    monkeypatch.setattr(config, "FAQ_COSINE_MIN", 0.0)
