"""Fixation token assembly: concat[x_v, h_v, fix features, scroll, visit/loop, conf]."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.graph.features import PANEL_VOCAB


SCROLL_KEYS = [
    "scroll_direction",  # encoded separately
    "scroll_displacement_px",
    "scroll_velocity_px_s",
    "scroll_t_since_scroll_onset_ms",
    "scroll_t_since_scroll_offset_ms",
    "scroll_during_scroll",
    "scroll_viewport_doc_position",
    "scroll_gaze_viewport_y",
]

SCROLL_DIR_VOCAB = ["none", "up", "down", "left", "right", "unknown"]

# Fixed layout of fixation_side_features (keep in sync with that function)
# [base 11 | loop_role 4 | scroll_dir 6 | scroll_num 7]
SIDE_FEATURE_DIM = 11 + 4 + len(SCROLL_DIR_VOCAB) + 7
# Indices within the side vector that are scroll features (zero-masked by train dropout)
SCROLL_SIDE_SLICE = slice(11 + 4, SIDE_FEATURE_DIM)  # scroll_dir one-hot + scroll_num



def _f(row: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for k in keys:
        if k in row and row[k] is not None:
            v = row[k]
            if isinstance(v, (bool, np.bool_)):
                return float(v)
            try:
                out = float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(out) or np.isinf(out):
                return default
            return out
    return default


def flatten_fixation_row(row: dict[str, Any]) -> dict[str, Any]:
    """Accept nested schema or flat parquet columns."""
    out = dict(row)
    scroll = row.get("scroll")
    if isinstance(scroll, dict):
        out.setdefault("scroll_direction", scroll.get("direction", "none"))
        out.setdefault("scroll_displacement_px", scroll.get("displacement_px", 0.0))
        out.setdefault("scroll_velocity_px_s", scroll.get("velocity_px_s", 0.0))
        out.setdefault("scroll_t_since_scroll_onset_ms", scroll.get("t_since_scroll_onset_ms", 0.0))
        out.setdefault("scroll_t_since_scroll_offset_ms", scroll.get("t_since_scroll_offset_ms", 0.0))
        out.setdefault("scroll_during_scroll", scroll.get("during_scroll", False))
        out.setdefault("scroll_viewport_doc_position", scroll.get("viewport_doc_position", 0.0))
        out.setdefault("scroll_gaze_viewport_y", scroll.get("gaze_viewport_y", 0.0))
    sacc = row.get("prev_saccade")
    if isinstance(sacc, dict):
        out.setdefault("prev_saccade_amplitude", sacc.get("amplitude", 0.0))
        out.setdefault("prev_saccade_direction_deg", sacc.get("direction_deg", 0.0))
    return out


def fixation_side_features(row: dict[str, Any], *, episode_duration_ms: float) -> np.ndarray:
    """Numeric/bool side features (no graph embeddings)."""
    row = flatten_fixation_row(row)
    dur = _f(row, "duration_ms")
    t = _f(row, "t_start_ms")
    rel_t = t / max(episode_duration_ms, 1.0)
    conf = _f(row, "assignment_confidence")
    amb = 1.0 if row.get("ambiguous") else 0.0
    visit = _f(row, "visit_count", default=1.0)
    is_ret = 1.0 if row.get("is_return") else 0.0
    gap_e = _f(row, "return_gap_events", default=0.0)
    gap_ms = _f(row, "return_gap_ms", default=0.0)
    short = 1.0 if row.get("short_loop_return") else 0.0
    # loop_role one-hot
    role = str(row.get("loop_role") or "none")
    role_oh = np.array(
        [1.0 if role == r else 0.0 for r in ("none", "origin", "pivot", "closure")],
        dtype=np.float32,
    )
    sacc_amp = _f(row, "prev_saccade_amplitude")
    sacc_dir = _f(row, "prev_saccade_direction_deg") / 180.0
    sdir = str(row.get("scroll_direction") or "none").lower()
    if sdir not in SCROLL_DIR_VOCAB:
        sdir = "unknown"
    sdir_oh = np.array([1.0 if sdir == d else 0.0 for d in SCROLL_DIR_VOCAB], dtype=np.float32)
    scroll_num = np.array(
        [
            _f(row, "scroll_displacement_px"),
            _f(row, "scroll_velocity_px_s"),
            _f(row, "scroll_t_since_scroll_onset_ms") / 1000.0,
            _f(row, "scroll_t_since_scroll_offset_ms") / 1000.0,
            1.0 if row.get("scroll_during_scroll") else 0.0,
            _f(row, "scroll_viewport_doc_position"),
            _f(row, "scroll_gaze_viewport_y"),
        ],
        dtype=np.float32,
    )
    base = np.array(
        [dur / 1000.0, rel_t, conf, amb, visit / 10.0, is_ret, gap_e / 20.0, gap_ms / 1000.0, short, sacc_amp / 500.0, sacc_dir],
        dtype=np.float32,
    )
    return np.concatenate([base, role_oh, sdir_oh, scroll_num])


class EmptySpaceEmbedding(nn.Module):
    """Learned embeddings for empty-space fixations (panel-specific or generic)."""

    def __init__(self, *, mode: str = "panel_specific", dim: int = 128, n_panels: int = 7) -> None:
        super().__init__()
        self.mode = mode
        self.dim = dim
        if mode == "drop":
            self.emb = None
        elif mode == "generic":
            self.emb = nn.Embedding(1, dim)
        else:
            self.emb = nn.Embedding(n_panels, dim)

    def forward(self, panel_ids: torch.Tensor) -> torch.Tensor:
        if self.emb is None:
            return torch.zeros(panel_ids.size(0), self.dim, device=panel_ids.device)
        if self.mode == "generic":
            idx = torch.zeros_like(panel_ids)
            return self.emb(idx)
        return self.emb(panel_ids.clamp(min=0, max=self.emb.num_embeddings - 1))


def panel_id_for_row(row: dict[str, Any], panel_classes: Sequence[str]) -> int:
    pl = str(row.get("panel_label") or "outside_document")
    if pl in panel_classes:
        return list(panel_classes).index(pl)
    if pl in PANEL_VOCAB:
        # map into extended list if present
        return list(panel_classes).index(pl) if pl in panel_classes else len(panel_classes) - 1
    return list(panel_classes).index("outside_document") if "outside_document" in panel_classes else 0


def assemble_token(
    *,
    x_v: np.ndarray,
    h_v: np.ndarray,
    side: np.ndarray,
    is_empty: bool,
    empty_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Concat graph slots + side features; empty-space may replace x_v/h_v."""
    if is_empty and empty_vec is not None:
        # Replace both slots with the same learned empty embedding (doubled)
        xv = empty_vec
        hv = empty_vec
    else:
        xv, hv = x_v, h_v
    return np.concatenate([xv, hv, side]).astype(np.float32)
