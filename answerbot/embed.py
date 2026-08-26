"""Embedding backend. Local by default; the model is loaded lazily on first use.

The e5 family expects asymmetric prefixes — "passage: " for indexed text and
"query: " for searches. Getting this wrong quietly costs a chunk of retrieval
quality, so the prefixes live here rather than at the call sites.
"""

import os

import numpy as np

from . import config

# Cap BLAS/OpenMP *before* torch is imported. After import these env vars are ignored.
_nthreads = max(1, config.EMBED_THREADS)
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_key, str(_nthreads))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(_nthreads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _model = SentenceTransformer(config.EMBED_MODEL, token=config.HF_TOKEN)
    return _model


def _needs_e5_prefix() -> bool:
    return "e5" in config.EMBED_MODEL.lower()


def encode_passages(
    texts: list[str],
    batch_size: int = 64,
    progress: bool = False,
    *,
    on_progress=None,
) -> np.ndarray:
    if _needs_e5_prefix():
        texts = [f"passage: {t}" for t in texts]
    if not texts:
        if on_progress is not None:
            on_progress(0, 0)
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    model = _get_model()
    if on_progress is None:
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=progress,
        )
        return np.asarray(vecs, dtype=np.float32)

    parts = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        parts.append(np.asarray(vecs, dtype=np.float32))
        on_progress(min(i + len(batch), n), n)
    return np.vstack(parts)


def encode_query(text: str) -> np.ndarray:
    if _needs_e5_prefix():
        text = f"query: {text}"
    vec = _get_model().encode([text], normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)[0]


def warmup() -> None:
    """Load weights and run a dummy encode so the first user query is not the stall."""
    encode_query("warmup")


def pack(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def unpack(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
