"""Cross-encoder rerank of the fused thread pool.

One CPU pass per question: the question against each expanded thread already
retrieved. This does not read or write the vector store, so changing the
reranker does not need a reindex. It cannot invent a message the pool missed.
"""

from __future__ import annotations

import logging

from . import config

log = logging.getLogger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder

        nthreads = max(1, config.EMBED_THREADS)
        torch.set_num_threads(nthreads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _model = CrossEncoder(
            config.RERANK_MODEL,
            max_length=config.RERANK_MAX_LENGTH,
            token=config.HF_TOKEN,
        )
    return _model


def score(question: str, texts: list[str]) -> list[float] | None:
    """Logits for `(question, text)` pairs, or None if scoring did not run."""
    if not texts:
        return []
    try:
        model = _get_model()
        pairs = [(question, text or "") for text in texts]
        raw = model.predict(
            pairs,
            batch_size=max(1, config.RERANK_BATCH),
            show_progress_bar=False,
        )
        return [float(s) for s in raw]
    except Exception:
        log.exception("cross-encoder rerank failed; keeping fusion order")
        return None


def warmup() -> None:
    """Load weights and score a dummy pair so the first question is not the stall."""
    score("warmup", ["warmup"])
