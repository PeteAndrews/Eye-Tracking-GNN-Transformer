"""Graph edge construction (M3 / PLAN S2-T2).

Edge types: NEXT_SEGMENT, PREVIOUS_SEGMENT, BELONGS_TO, SPATIAL_NEIGHBOUR,
SEMANTIC_CANDIDATE. SAME_MARK_POINT / SAME_STAR are intentionally absent.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

Edge = tuple[str, str]  # (source_id, target_id)
EdgeAttr = dict[str, Any]


def panel_node_id(panel_label: str) -> str:
    return f"panel_{panel_label}"


def build_next_previous(
    segments: Sequence[dict[str, Any]],
) -> tuple[list[Edge], list[Edge]]:
    """Panel-grouped NEXT_SEGMENT / PREVIOUS_SEGMENT by segment_order."""
    by_panel: dict[str, list[dict[str, Any]]] = {}
    for s in segments:
        by_panel.setdefault(str(s["panel_label"]), []).append(s)
    next_edges: list[Edge] = []
    prev_edges: list[Edge] = []
    for _panel, group in by_panel.items():
        ordered = sorted(group, key=lambda x: int(x.get("segment_order") or 0))
        for a, b in zip(ordered, ordered[1:]):
            next_edges.append((a["segment_id"], b["segment_id"]))
            prev_edges.append((b["segment_id"], a["segment_id"]))
    return next_edges, prev_edges


def build_belongs_to(segments: Sequence[dict[str, Any]]) -> list[Edge]:
    """Semantic segment → abstract panel node."""
    return [(s["segment_id"], panel_node_id(str(s["panel_label"]))) for s in segments]


def _centroid(s: dict[str, Any]) -> tuple[float, float]:
    g = s.get("geometry") or {}
    if "x" in g and "y" in g:
        return float(g["x"]), float(g["y"])
    return (
        0.5 * (float(g.get("x_min", 0)) + float(g.get("x_max", 0))),
        0.5 * (float(g.get("y_min", 0)) + float(g.get("y_max", 0))),
    )


def build_spatial_neighbour(
    segments: Sequence[dict[str, Any]],
    *,
    k: int = 3,
    within_panel_only: bool = True,
) -> tuple[list[Edge], list[EdgeAttr]]:
    """k-NN spatial edges within panel (undirected → two directed edges)."""
    by_panel: dict[str, list[dict[str, Any]]] = {}
    for s in segments:
        if not s.get("geometry"):
            continue
        by_panel.setdefault(str(s["panel_label"]), []).append(s)

    edges: list[Edge] = []
    attrs: list[EdgeAttr] = []
    seen: set[tuple[str, str]] = set()

    groups = list(by_panel.values()) if within_panel_only else [list(segments)]
    for group in groups:
        if len(group) < 2:
            continue
        ids = [s["segment_id"] for s in group]
        xy = np.array([_centroid(s) for s in group], dtype=np.float64)
        diff = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))
        np.fill_diagonal(dist, np.inf)
        for i, sid in enumerate(ids):
            order = np.argsort(dist[i])
            for j in order[:k]:
                if not np.isfinite(dist[i, j]):
                    continue
                tid = ids[int(j)]
                pair = tuple(sorted((sid, tid)))
                if pair in seen:
                    continue
                seen.add(pair)
                d = float(dist[i, j])
                dx = float(xy[int(j), 0] - xy[i, 0])
                dy = float(xy[int(j), 1] - xy[i, 1])
                same_col = abs(dx) < abs(dy)
                for src, tgt, sdx, sdy in (
                    (sid, tid, dx, dy),
                    (tid, sid, -dx, -dy),
                ):
                    edges.append((src, tgt))
                    attrs.append(
                        {
                            "relation": "SPATIAL_NEIGHBOUR",
                            "distance": d,
                            "dx": sdx,
                            "dy": sdy,
                            "same_column": bool(same_col),
                        }
                    )
    return edges, attrs


def _panel_pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def _is_ms_content_bullet(s: dict[str, Any]) -> bool:
    if str(s.get("panel_label")) != "mark_scheme":
        return False
    b = s.get("bools") or {}
    if b.get("is_level_descriptor"):
        return False
    if b.get("is_mark_scheme_point"):
        return True
    role = str(s.get("segment_role") or "")
    return role in {"mark_scheme_point", "bullet_point", "answers", "clause", "sentence"}


def build_semantic_candidate(
    segments: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    allowed_panel_pairs: Sequence[Sequence[str]],
    min_similarity: float = 0.25,
    k_response_mark_scheme: int = 3,
    k_other: int = 2,
    coverage_floor: bool = True,
) -> tuple[list[Edge], list[EdgeAttr]]:
    """Cross-panel SEMANTIC_CANDIDATE edges from cosine similarity."""
    if len(segments) != len(embeddings):
        raise ValueError("segments and embeddings length mismatch")
    emb = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb = emb / norms

    allowed = {_panel_pair_key(str(a), str(b)) for a, b in allowed_panel_pairs}
    ids = [s["segment_id"] for s in segments]
    panels = [str(s["panel_label"]) for s in segments]
    n = len(segments)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)

    edge_map: dict[Edge, EdgeAttr] = {}

    def _put(src_i: int, tgt_i: int, sim: float, rank: int, below: bool) -> None:
        key = (ids[src_i], ids[tgt_i])
        attr = {
            "relation": "SEMANTIC_CANDIDATE",
            "cosine": float(sim),
            "rank": int(rank),
            "panel_pair": f"{panels[src_i]}|{panels[tgt_i]}",
            "below_threshold": bool(below),
        }
        prev = edge_map.get(key)
        if prev is None or float(prev["cosine"]) < float(sim):
            edge_map[key] = attr

    for pair in allowed:
        pa, pb = pair
        idx_a = [i for i, p in enumerate(panels) if p == pa]
        idx_b = [i for i, p in enumerate(panels) if p == pb]
        if not idx_a or not idx_b:
            continue
        prioritised = pair == _panel_pair_key("response", "mark_scheme")
        k = k_response_mark_scheme if prioritised else k_other

        for src_idx, tgt_idx in ((idx_a, idx_b), (idx_b, idx_a)):
            for i in src_idx:
                scored = sorted(
                    ((float(sims[i, j]), j) for j in tgt_idx),
                    key=lambda x: x[0],
                    reverse=True,
                )
                for rank, (sim, j) in enumerate(scored[:k], start=1):
                    if sim < min_similarity:
                        continue
                    _put(i, j, sim, rank, below=False)

        if prioritised and coverage_floor:
            ms_idx = [i for i, s in enumerate(segments) if _is_ms_content_bullet(s)]
            resp_idx = [i for i, p in enumerate(panels) if p == "response"]
            for mi in ms_idx:
                if not resp_idx:
                    continue
                best_j = max(resp_idx, key=lambda j: float(sims[mi, j]))
                sim = float(sims[mi, best_j])
                # Ensure at least one directed edge involving this MS bullet
                has = any(ids[mi] in e for e in edge_map)
                if not has:
                    _put(mi, best_j, sim, rank=1, below=sim < min_similarity)
                    _put(best_j, mi, sim, rank=1, below=sim < min_similarity)

    edges = list(edge_map.keys())
    attrs = [edge_map[e] for e in edges]
    return edges, attrs


def assert_no_same_panel_semantic(
    edges: Iterable[Edge],
    segments: Sequence[dict[str, Any]],
) -> None:
    panel_of = {s["segment_id"]: str(s["panel_label"]) for s in segments}
    for a, b in edges:
        if a.startswith("panel_") or b.startswith("panel_"):
            continue
        if a not in panel_of or b not in panel_of:
            continue
        if panel_of[a] == panel_of[b]:
            raise AssertionError(f"same-panel SEMANTIC_CANDIDATE forbidden: {a}->{b}")
