"""Scroll-feature dropout and seed helpers for M6 training."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch

from src.models.tokens import SCROLL_SIDE_SLICE, SIDE_FEATURE_DIM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_scroll_dropout(
    tokens: torch.Tensor,
    *,
    gnn_out_dim: int = 128,
    p: float = 0.3,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """With prob p, zero all scroll side-features for the whole episode (per batch row).

    tokens: [B, T, D] where D = 2*gnn_out_dim + SIDE_FEATURE_DIM.
    """
    if p <= 0:
        return tokens
    b = tokens.size(0)
    out = tokens.clone()
    start = 2 * gnn_out_dim + SCROLL_SIDE_SLICE.start
    end = 2 * gnn_out_dim + SCROLL_SIDE_SLICE.stop
    assert end - start <= SIDE_FEATURE_DIM
    drop = torch.rand(b, generator=rng, device=tokens.device) < p
    if drop.any():
        out[drop, :, start:end] = 0.0
    return out
