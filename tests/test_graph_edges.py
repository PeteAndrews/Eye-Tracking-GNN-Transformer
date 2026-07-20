"""Unit tests for M3 edge builders (fixtures; no HF downloads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.graph.config_check import assert_encoder_graph_dim_match
from src.graph.edges import (
    assert_no_same_panel_semantic,
    build_belongs_to,
    build_next_previous,
    build_semantic_candidate,
    build_spatial_neighbour,
    panel_node_id,
)
from src.utils import io as uio

ROOT = Path(__file__).resolve().parents[1]
FX01 = ROOT / "fixtures" / "trials" / "fx01_T99"


def _segments():
    return uio.read_json(FX01 / "segments.json")


def _expected():
    return uio.read_json(FX01 / "expected_edges.json")


def test_next_previous_matches_fixture():
    segs = _segments()
    nxt, prev = build_next_previous(segs)
    exp = _expected()
    assert sorted(nxt) == sorted(tuple(e) for e in exp["NEXT_SEGMENT"])
    assert sorted(prev) == sorted(tuple(e) for e in exp["PREVIOUS_SEGMENT"])


def test_belongs_to_matches_fixture():
    segs = _segments()
    edges = build_belongs_to(segs)
    exp = _expected()
    assert sorted(edges) == sorted(tuple(e) for e in exp["BELONGS_TO"])
    assert panel_node_id("response") == "panel_response"


def test_spatial_within_panel_only():
    segs = _segments()
    edges, attrs = build_spatial_neighbour(segs, k=3, within_panel_only=True)
    panel_of = {s["segment_id"]: s["panel_label"] for s in segs}
    for a, b in edges:
        assert panel_of[a] == panel_of[b]
    # Undirected pairs stored as 2 directed edges
    undirected = {tuple(sorted(e)) for e in edges}
    # Fixture within-panel neighbours among response / mark_scheme stacks
    assert ("seg_r1", "seg_r2") in undirected or ("seg_r2", "seg_r1") in undirected
    # Cross-panel must not appear
    assert ("seg_r2", "seg_ms1") not in undirected


def test_semantic_excludes_same_panel():
    segs = _segments()
    n = len(segs)
    # One-hot-ish random embeddings; boost expected pairs
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, 16)).astype(np.float64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    id_to_i = {s["segment_id"]: i for i, s in enumerate(segs)}
    for a, b in (("seg_r2", "seg_ms1"), ("seg_r1", "seg_ms2"), ("seg_r3", "seg_c1")):
        i, j = id_to_i[a], id_to_i[b]
        emb[i] = emb[j]  # identical → cosine 1
    allowed = [
        ["question", "response"],
        ["response", "mark_scheme"],
        ["response", "commentary"],
        ["response", "star_chart"],
        ["mark_scheme", "commentary"],
    ]
    edges, attrs = build_semantic_candidate(
        segs,
        emb,
        allowed_panel_pairs=allowed,
        min_similarity=0.5,
        k_response_mark_scheme=3,
        k_other=2,
        coverage_floor=True,
    )
    assert_no_same_panel_semantic(edges, segs)
    undirected = {tuple(sorted(e)) for e in edges}
    assert tuple(sorted(("seg_r2", "seg_ms1"))) in undirected
    # Same-panel pair must never appear
    assert tuple(sorted(("seg_r1", "seg_r2"))) not in undirected


def test_semantic_coverage_floor_below_threshold():
    segs = [
        {
            "segment_id": "r1",
            "panel_label": "response",
            "bools": {},
            "segment_role": "answers",
        },
        {
            "segment_id": "ms1",
            "panel_label": "mark_scheme",
            "bools": {"is_mark_scheme_point": True},
            "segment_role": "bullet_point",
        },
        {
            "segment_id": "ms2",
            "panel_label": "mark_scheme",
            "bools": {"is_mark_scheme_point": True},
            "segment_role": "bullet_point",
        },
    ]
    # ms2 only weakly related to r1 (near-orthogonal after L2)
    emb = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    edges, attrs = build_semantic_candidate(
        segs,
        emb,
        allowed_panel_pairs=[["response", "mark_scheme"]],
        min_similarity=0.9,
        k_response_mark_scheme=1,
        k_other=2,
        coverage_floor=True,
    )
    # ms2→r1 (or reverse) should exist with below_threshold
    pairs = set(edges)
    assert ("ms2", "r1") in pairs or ("r1", "ms2") in pairs
    below = [a for e, a in zip(edges, attrs) if set(e) == {"ms2", "r1"}]
    assert any(a.get("below_threshold") for a in below)


def test_encoder_graph_dim_match():
    info = assert_encoder_graph_dim_match(ROOT)
    assert info["ok"]
    assert info["embedding_dim"] == 1024
    assert "bge" in str(info["model_name"]).lower()
