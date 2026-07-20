"""Frozen TextEncoderV1 wrapper (pinned after M2 bake-off)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
from omegaconf import OmegaConf

from src.text.hf_encoder import HFMeanPoolEncoder, load_text_encoder
from src.utils import io as uio


class TextEncoderV1:
    """Deterministic HF mean-pool wrapper with L2-normalised embeddings."""

    def __init__(
        self,
        model_name: str,
        *,
        text_prefix: str = "",
        batch_size: int = 32,
        normalise_l2: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.text_prefix = text_prefix or ""
        self.batch_size = int(batch_size)
        self.normalise_l2 = bool(normalise_l2)
        self._model: HFMeanPoolEncoder = load_text_encoder(
            model_name,
            text_prefix=self.text_prefix,
            batch_size=self.batch_size,
            normalise_l2=self.normalise_l2,
            device=device,
        )
        self.embedding_dim = int(self._model.embedding_dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(texts)

    def card(self) -> dict[str, Any]:
        return {
            "name": "TextEncoderV1",
            "model_name": self.model_name,
            "text_prefix": self.text_prefix,
            "pooling": "mean",
            "normalise_l2": self.normalise_l2,
            "embedding_dim": self.embedding_dim,
            "batch_size": self.batch_size,
            "backend": "transformers.AutoModel+mean_pool",
        }

    def save_card(self, path: Union[str, Path], *, extra: Optional[dict[str, Any]] = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.card()
        if extra:
            payload.update(extra)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        payload["card_sha256"] = hashlib.sha256(blob).hexdigest()
        uio.write_json(path, payload)
        return path


def load_encoder_from_card(card_path: Path, *, device: Optional[str] = None) -> TextEncoderV1:
    card = uio.read_json(card_path)
    return TextEncoderV1(
        str(card["model_name"]),
        text_prefix=str(card.get("text_prefix") or ""),
        batch_size=int(card.get("batch_size") or 32),
        normalise_l2=bool(card.get("normalise_l2", True)),
        device=device,
    )


def resolve_model_revision(model_name: str) -> str:
    """HF commit SHA for the pinned model revision (empty string if unavailable)."""
    import os

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import model_info

        info = model_info(model_name)
        return str(getattr(info, "sha", "") or "")
    except Exception:
        return ""


def freeze_encoder_id(
    repo_root: Path,
    encoder_id: str,
    *,
    bakeoff_summary: Optional[dict[str, Any]] = None,
    selection_note: Optional[str] = None,
) -> Path:
    """Freeze TextEncoderV1 card for a named candidate id (e.g. after owner override)."""
    cfg = OmegaConf.load(repo_root / "configs" / "encoder_selection.yaml")
    cand = next(c for c in cfg.candidates if str(c.id) == encoder_id)
    model_name = str(cand.model_name)
    text_prefix = str(cand.get("text_prefix") or "")
    enc = TextEncoderV1(
        model_name,
        text_prefix=text_prefix,
        batch_size=int(cfg.batch_size),
        normalise_l2=bool(cfg.normalise_l2),
    )
    revision = resolve_model_revision(model_name)
    extra: dict[str, Any] = {
        "bakeoff_winner_id": encoder_id,
        "model_revision": revision,
        "metric": "ranking_accuracy",
        "tie_breaker": "ranking_accuracy_hard",
        "selection_note": selection_note
        or "Selected from bake-off (see reports/encoder_bakeoff_v1.md).",
    }
    if bakeoff_summary:
        # Prefer per-candidate metrics when overriding
        cand_row = None
        for r in bakeoff_summary.get("results") or []:
            if r.get("id") == encoder_id:
                cand_row = r
                break
        if cand_row:
            extra["bakeoff_ranking_accuracy"] = cand_row.get("ranking_accuracy")
            extra["bakeoff_ranking_accuracy_hard"] = cand_row.get("ranking_accuracy_hard")
            extra["bakeoff_ranking_accuracy_easy"] = cand_row.get("ranking_accuracy_easy")
            extra["bakeoff_per_category"] = cand_row.get("per_category")
        else:
            extra["bakeoff_ranking_accuracy"] = bakeoff_summary.get("winner_accuracy")
            extra["bakeoff_ranking_accuracy_hard"] = bakeoff_summary.get("winner_accuracy_hard")
            extra["bakeoff_ranking_accuracy_easy"] = bakeoff_summary.get("winner_accuracy_easy")
        extra["bakeoff_n_triples"] = bakeoff_summary.get("n_triples")
    card_path = repo_root / str(cfg.paths.encoder_card)
    return enc.save_card(card_path, extra=extra)


def freeze_winner(repo_root: Path, bakeoff_summary: dict[str, Any]) -> Path:
    """Write TextEncoderV1 card from bake-off winner."""
    return freeze_encoder_id(
        repo_root,
        str(bakeoff_summary["winner_id"]),
        bakeoff_summary=bakeoff_summary,
    )
