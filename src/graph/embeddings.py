"""Cached TextEncoderV1 embeddings for graph nodes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.text.encoder import TextEncoderV1, load_encoder_from_card
from src.utils import io as uio

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def embedding_cache_path(repo_root: Path, graph_version: str, stem: str) -> Path:
    return (
        repo_root
        / "data_processed"
        / "embeddings"
        / graph_version
        / f"{stem}__embeddings.npy"
    )


def embedding_ids_path(repo_root: Path, graph_version: str, stem: str) -> Path:
    return (
        repo_root
        / "data_processed"
        / "embeddings"
        / graph_version
        / f"{stem}__embedding_ids.json"
    )


def encode_segments(
    encoder: TextEncoderV1,
    segments: Sequence[dict[str, Any]],
) -> np.ndarray:
    texts = [str(s.get("corrected_text") or "") for s in segments]
    if not texts:
        return np.zeros((0, encoder.embedding_dim), dtype=np.float32)
    return encoder.encode(texts)


def load_or_build_embeddings(
    repo_root: Path,
    *,
    stem: str,
    segments: Sequence[dict[str, Any]],
    graph_version: str,
    encoder: Optional[TextEncoderV1] = None,
    force: bool = False,
) -> np.ndarray:
    """Return (n_seg, dim) L2-normalised embeddings aligned to *segments* order."""
    npy_path = embedding_cache_path(repo_root, graph_version, stem)
    ids_path = embedding_ids_path(repo_root, graph_version, stem)
    # Legacy parquet cache
    parquet_path = npy_path.with_suffix(".parquet")
    ids = [s["segment_id"] for s in segments]

    if npy_path.is_file() and ids_path.is_file() and not force:
        cached_ids = uio.read_json(ids_path)
        if list(cached_ids) == ids:
            return np.load(npy_path).astype(np.float32)

    if parquet_path.is_file() and not force:
        df = pd.read_parquet(parquet_path)
        by_id = {
            str(r["segment_id"]): np.asarray(r["embedding"], dtype=np.float32)
            for _, r in df.iterrows()
        }
        if all(i in by_id for i in ids):
            emb = np.stack([by_id[i] for i in ids], axis=0)
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, emb)
            uio.write_json(ids_path, ids)
            return emb

    if encoder is None:
        card = repo_root / "artifacts" / "text_encoder_v1_card.json"
        encoder = load_encoder_from_card(card, device="cpu")

    emb = encode_segments(encoder, segments)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, emb.astype(np.float32))
    uio.write_json(ids_path, ids)
    uio.write_json(
        npy_path.with_name(f"{stem}__embeddings_meta.json"),
        {
            "stem": stem,
            "n_segments": len(ids),
            "embedding_dim": int(emb.shape[1]) if len(emb) else encoder.embedding_dim,
            "model_name": encoder.model_name,
        },
    )
    return emb
