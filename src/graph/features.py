"""Node feature assembly for M3 graphs."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


PANEL_VOCAB = ["question", "response", "mark_scheme", "commentary", "star_chart", "ui"]
BOOL_KEYS = [
    "command_word",
    "domain_term",
    "is_bullet_point",
    "is_level_descriptor",
    "is_mark_scheme_point",
    "is_commentary",
    "is_star_chart",
    "requires_calculation",
    "contains_data_reference",
    "contains_allow_instruction",
    "contains_reject_instruction",
    "contains_comparison",
]


def one_hot(value: str, vocab: Sequence[str]) -> np.ndarray:
    v = np.zeros(len(vocab), dtype=np.float32)
    if value in vocab:
        v[list(vocab).index(value)] = 1.0
    return v


def geometry_features(seg: dict[str, Any], doc_w: float, doc_h: float) -> np.ndarray:
    g = seg.get("geometry") or {}
    x, y = float(g.get("x", 0)), float(g.get("y", 0))
    w, h = float(g.get("w", 0)), float(g.get("h", 0))
    dw, dh = max(doc_w, 1.0), max(doc_h, 1.0)
    return np.array([x / dw, y / dh, w / dw, h / dh], dtype=np.float32)


def segment_side_features(
    seg: dict[str, Any],
    *,
    doc_w: float,
    doc_h: float,
    n_segments_in_panel: int,
) -> np.ndarray:
    """Non-text features for a semantic segment (excludes embedding)."""
    panel = one_hot(str(seg.get("panel_label")), PANEL_VOCAB)
    bools = seg.get("bools") or {}
    bvec = np.array([1.0 if bools.get(k) else 0.0 for k in BOOL_KEYS], dtype=np.float32)
    fmt = seg.get("formatting") or {}
    fvec = np.array(
        [
            1.0 if fmt.get("bold") else 0.0,
            1.0 if fmt.get("italic") else 0.0,
            float(fmt.get("formatted_prop") or 0.0),
        ],
        dtype=np.float32,
    )
    geom = geometry_features(seg, doc_w, doc_h)
    order = float(seg.get("segment_order") or 0)
    order_n = order / max(n_segments_in_panel - 1, 1)
    star_flag = np.array(
        [1.0 if seg.get("is_star_conditional") else 0.0], dtype=np.float32
    )
    return np.concatenate([panel, bvec, fvec, geom, [order_n], star_flag])


def assemble_node_features(
    segments: Sequence[dict[str, Any]],
    text_embeddings: np.ndarray,
    *,
    doc_w: float,
    doc_h: float,
    panel_nodes: Optional[Sequence[str]] = None,
    text_dim: int = 1024,
) -> tuple[np.ndarray, list[str]]:
    """Concat text emb + side features for segments; zero text for abstract panels.

    Returns (X, node_ids) where panel nodes are appended after segments.
    """
    emb = np.asarray(text_embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(segments):
        raise ValueError("text_embeddings must be (n_segments, dim)")
    if emb.shape[1] != text_dim:
        raise ValueError(f"expected text_dim={text_dim}, got {emb.shape[1]}")

    by_panel_count: dict[str, int] = {}
    for s in segments:
        by_panel_count[str(s["panel_label"])] = by_panel_count.get(str(s["panel_label"]), 0) + 1

    rows: list[np.ndarray] = []
    node_ids: list[str] = []
    side_dim = None
    for i, s in enumerate(segments):
        side = segment_side_features(
            s,
            doc_w=doc_w,
            doc_h=doc_h,
            n_segments_in_panel=by_panel_count[str(s["panel_label"])],
        )
        side_dim = len(side)
        rows.append(np.concatenate([emb[i], side]))
        node_ids.append(s["segment_id"])

    if panel_nodes is None:
        panel_nodes = sorted({str(s["panel_label"]) for s in segments})
    assert side_dim is not None
    for pl in panel_nodes:
        from src.graph.edges import panel_node_id

        zero_text = np.zeros(text_dim, dtype=np.float32)
        # Panel identity one-hot in the panel slot of side features
        side = np.zeros(side_dim, dtype=np.float32)
        if pl in PANEL_VOCAB:
            side[PANEL_VOCAB.index(pl)] = 1.0
        rows.append(np.concatenate([zero_text, side]))
        node_ids.append(panel_node_id(pl))

    return np.stack(rows, axis=0), node_ids
