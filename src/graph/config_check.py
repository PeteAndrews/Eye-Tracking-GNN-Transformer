"""Assert TextEncoderV1 card dims match configs/graph.yaml before graph build."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src.utils import io as uio


def load_graph_config(repo_root: Path) -> Any:
    return OmegaConf.load(repo_root / "configs" / "graph.yaml")


def assert_encoder_graph_dim_match(repo_root: Path) -> dict[str, Any]:
    """Raise if frozen encoder dim ≠ graph.yaml text_embedding_dim."""
    cfg = load_graph_config(repo_root)
    card_path = repo_root / str(cfg.paths.encoder_card)
    if not card_path.is_file():
        raise FileNotFoundError(f"TextEncoderV1 card missing: {card_path}")
    card = uio.read_json(card_path)
    card_dim = int(card["embedding_dim"])
    cfg_dim = int(cfg.text_embedding_dim)
    if card_dim != cfg_dim:
        raise ValueError(
            f"Encoder dim {card_dim} != graph.yaml text_embedding_dim {cfg_dim}. "
            "Re-freeze card or update configs/graph.yaml."
        )
    return {
        "ok": True,
        "embedding_dim": card_dim,
        "model_name": card.get("model_name"),
        "model_revision": card.get("model_revision"),
        "graph_version": str(cfg.graph_version),
    }
