"""Embedding backend. Local by default; the model is loaded lazily on first use.

The e5 family expects asymmetric prefixes — "passage: " for indexed text and
"query: " for searches. Getting this wrong quietly costs a chunk of retrieval
quality, so the prefixes live here rather than at the call sites.
"""

import numpy as np

from . import config

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def _needs_e5_prefix() -> bool:
    return "e5" in config.EMBED_MODEL.lower()


def encode_passages(texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
    if _needs_e5_prefix():
        texts = [f"passage: {t}" for t in texts]
    vecs = _get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=progress,
    )
    return np.asarray(vecs, dtype=np.float32)


def encode_query(text: str) -> np.ndarray:
    if _needs_e5_prefix():
        text = f"query: {text}"
    vec = _get_model().encode([text], normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)[0]


def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
