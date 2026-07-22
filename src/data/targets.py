"""Next-relation multi-hot targets and ranking labels for M5 episodes."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from src.graph.build import RELATION_TO_ID

# Ordered relation vocabulary for multi-hot targets
GRAPH_RELATION_NAMES = [
    "NEXT_SEGMENT",
    "PREVIOUS_SEGMENT",
    "BELONGS_TO",
    "SPATIAL_NEIGHBOUR",
    "SEMANTIC_CANDIDATE",
]
NO_DIRECT_RELATION = "NO_DIRECT_RELATION"
EMPTY_SPACE_TRANSITION = "EMPTY_SPACE_TRANSITION"

RELATION_VOCAB = GRAPH_RELATION_NAMES + [NO_DIRECT_RELATION, EMPTY_SPACE_TRANSITION]
RELATION_NAME_TO_IDX = {n: i for i, n in enumerate(RELATION_VOCAB)}


def build_edge_relation_lookup(
    edge_index: np.ndarray | Any,
    edge_type: np.ndarray | Any,
) -> dict[tuple[int, int], set[int]]:
    """Map (src, dst) → set of relation-type ids present on that directed edge."""
    if hasattr(edge_index, "cpu"):
        edge_index = edge_index.cpu().numpy()
    if hasattr(edge_type, "cpu"):
        edge_type = edge_type.cpu().numpy()
    edge_index = np.asarray(edge_index)
    edge_type = np.asarray(edge_type).astype(int)
    lookup: dict[tuple[int, int], set[int]] = {}
    if edge_index.size == 0:
        return lookup
    for e in range(edge_index.shape[1]):
        key = (int(edge_index[0, e]), int(edge_index[1, e]))
        lookup.setdefault(key, set()).add(int(edge_type[e]))
    return lookup


def next_relation_multihot(
    src_node: Optional[int],
    dst_node: Optional[int],
    *,
    edge_lookup: dict[tuple[int, int], set[int]],
    src_is_empty: bool,
    dst_is_empty: bool,
    include_no_direct: bool = True,
    empty_label: str = EMPTY_SPACE_TRANSITION,
) -> np.ndarray:
    """Multi-hot vector over RELATION_VOCAB for the transition src→dst.

    Empty-space involvement → EMPTY_SPACE_TRANSITION (and not NO_DIRECT).
    Otherwise all graph relations on the directed edge; if none, NO_DIRECT_RELATION.
    """
    vec = np.zeros(len(RELATION_VOCAB), dtype=np.float32)
    if src_is_empty or dst_is_empty or src_node is None or dst_node is None:
        vec[RELATION_NAME_TO_IDX[empty_label]] = 1.0
        return vec

    rels = edge_lookup.get((int(src_node), int(dst_node)), set())
    # Plan: relations holding between consecutive viewed nodes — directed src→dst.
    id_to_name = {v: k for k, v in RELATION_TO_ID.items()}
    if rels:
        for rid in rels:
            name = id_to_name.get(int(rid))
            if name and name in RELATION_NAME_TO_IDX:
                vec[RELATION_NAME_TO_IDX[name]] = 1.0
    elif include_no_direct:
        vec[RELATION_NAME_TO_IDX[NO_DIRECT_RELATION]] = 1.0
    return vec


def sample_ranking_candidates(
    *,
    positive_node: Optional[int],
    n_segments: int,
    visited: set[int],
    text_emb: np.ndarray,
    query_emb: Optional[np.ndarray],
    n_easy: int = 8,
    n_hard: int = 4,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Return (candidate_node_ids, labels) with 1 for positive, 0 for negatives.

    Negatives: easy = random unvisited segments; hard = top-cosine unvisited.
    If no positive (empty-space next), returns empty lists.
    """
    if positive_node is None or positive_node < 0 or positive_node >= n_segments:
        return [], []
    unvisited = [i for i in range(n_segments) if i not in visited and i != positive_node]
    hard: list[int] = []
    if query_emb is not None and len(unvisited) and text_emb.shape[0] >= n_segments:
        q = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        q_n = float(np.linalg.norm(q)) + 1e-12
        q = q / q_n
        idx = np.asarray(unvisited, dtype=np.int64)
        mat = np.asarray(text_emb[idx], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        sims = (mat / norms) @ q
        # stable top-k without Python loop over dots
        k = min(n_hard, len(unvisited))
        if k > 0:
            top = np.argpartition(-sims, kth=k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            hard = [int(idx[j]) for j in top]
    remaining = [i for i in unvisited if i not in hard]
    rng.shuffle(remaining)
    easy = remaining[:n_easy]
    cands = [positive_node] + hard + easy
    labels = [1] + [0] * (len(cands) - 1)
    return cands, labels
