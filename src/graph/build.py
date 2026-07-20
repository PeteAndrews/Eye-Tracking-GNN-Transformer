"""Build one trial graph (segments + edges + node features) as a PyG-ready dict."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf

from src.graph.edges import (
    build_belongs_to,
    build_next_previous,
    build_semantic_candidate,
    build_spatial_neighbour,
)
from src.graph.features import assemble_node_features


RELATION_TO_ID = {
    "NEXT_SEGMENT": 0,
    "PREVIOUS_SEGMENT": 1,
    "BELONGS_TO": 2,
    "SPATIAL_NEIGHBOUR": 3,
    "SEMANTIC_CANDIDATE": 4,
}


def build_graph_dict(
    segments: Sequence[dict[str, Any]],
    text_embeddings: np.ndarray,
    *,
    trial_id: str,
    star_condition: str,
    graph_cfg: Any,
    doc_w: float = 1.0,
    doc_h: float = 1.0,
) -> dict[str, Any]:
    """Assemble node features and typed edges for one (trial, star) graph."""
    text_dim = int(graph_cfg.text_embedding_dim)
    X, node_ids = assemble_node_features(
        segments,
        text_embeddings,
        doc_w=doc_w,
        doc_h=doc_h,
        text_dim=text_dim,
    )
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    e_src: list[int] = []
    e_tgt: list[int] = []
    e_type: list[int] = []
    e_attr_rows: list[list[float]] = []

    def _add_edges(
        edges: list[tuple[str, str]],
        relation: str,
        attrs: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        rid = RELATION_TO_ID[relation]
        for i, (a, b) in enumerate(edges):
            if a not in id_to_idx or b not in id_to_idx:
                continue
            e_src.append(id_to_idx[a])
            e_tgt.append(id_to_idx[b])
            e_type.append(rid)
            if attrs and i < len(attrs):
                at = attrs[i]
                e_attr_rows.append(
                    [
                        float(at.get("cosine", at.get("distance", 0.0))),
                        float(at.get("rank", 0)),
                        1.0 if at.get("below_threshold") else 0.0,
                        float(at.get("dx", 0.0)),
                        float(at.get("dy", 0.0)),
                    ]
                )
            else:
                e_attr_rows.append([0.0, 0.0, 0.0, 0.0, 0.0])

    nxt, prev = build_next_previous(segments)
    _add_edges(nxt, "NEXT_SEGMENT")
    _add_edges(prev, "PREVIOUS_SEGMENT")
    _add_edges(build_belongs_to(segments), "BELONGS_TO")

    sp_cfg = graph_cfg.edges.spatial_neighbour
    if sp_cfg.enabled:
        sp_e, sp_a = build_spatial_neighbour(
            segments,
            k=int(sp_cfg.k),
            within_panel_only=bool(sp_cfg.within_panel_only),
        )
        _add_edges(sp_e, "SPATIAL_NEIGHBOUR", sp_a)

    sem_cfg = graph_cfg.edges.semantic_candidate
    if sem_cfg.enabled:
        allowed = [list(p) for p in sem_cfg.allowed_panel_pairs]
        sem_e, sem_a = build_semantic_candidate(
            segments,
            text_embeddings,
            allowed_panel_pairs=allowed,
            min_similarity=float(sem_cfg.min_similarity),
            k_response_mark_scheme=int(sem_cfg.response_mark_scheme.k),
            k_other=int(sem_cfg.other_pairs.k),
            coverage_floor=bool(sem_cfg.response_mark_scheme.coverage_floor),
        )
        _add_edges(sem_e, "SEMANTIC_CANDIDATE", sem_a)

    edge_index = torch.tensor([e_src, e_tgt], dtype=torch.long) if e_src else torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(e_type, dtype=torch.long) if e_type else torch.zeros((0,), dtype=torch.long)
    edge_attr = (
        torch.tensor(e_attr_rows, dtype=torch.float32)
        if e_attr_rows
        else torch.zeros((0, 5), dtype=torch.float32)
    )

    return {
        "trial_id": trial_id,
        "star_condition": star_condition,
        "graph_version": str(graph_cfg.graph_version),
        "x": torch.from_numpy(X),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "edge_attr": edge_attr,
        "node_ids": node_ids,
        "relation_to_id": dict(RELATION_TO_ID),
        "n_segments": len(segments),
        "text_embedding_dim": text_dim,
    }


def save_graph_pt(graph: dict[str, Any], path: Any) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, path)
