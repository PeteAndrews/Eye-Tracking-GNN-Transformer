"""Tests for M3 correspondence + feature/graph assembly (no HF)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from src.graph.build import build_graph_dict
from src.graph.correspondence import match_ns_s, normalise_text
from src.graph.features import assemble_node_features
from src.utils import io as uio

ROOT = Path(__file__).resolve().parents[1]
FX01 = ROOT / "fixtures" / "trials" / "fx01_T99"


def test_normalise_text():
    assert normalise_text("  Foo   BAR ") == "foo bar"


def test_ns_s_correspondence_identical():
    segs = uio.read_json(FX01 / "segments.json")
    result = match_ns_s(segs, segs, star_conditional_ids=set())
    assert result["ok"]
    assert result["n_matched"] == len(segs)


def test_assemble_and_build_graph_shapes():
    segs = uio.read_json(FX01 / "segments.json")
    cfg = OmegaConf.load(ROOT / "configs" / "graph.yaml")
    dim = int(cfg.text_embedding_dim)
    emb = np.random.default_rng(0).normal(size=(len(segs), dim)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    X, node_ids = assemble_node_features(segs, emb, doc_w=800, doc_h=600, text_dim=dim)
    n_panels = len({s["panel_label"] for s in segs})
    assert X.shape[0] == len(segs) + n_panels
    assert X.shape[1] > dim
    assert len(node_ids) == X.shape[0]

    g = build_graph_dict(
        segs,
        emb,
        trial_id="T99",
        star_condition="not_eligible",
        graph_cfg=cfg,
        doc_w=800,
        doc_h=600,
    )
    assert g["x"].shape[0] == X.shape[0]
    assert g["text_embedding_dim"] == 1024
    assert g["edge_index"].shape[0] == 2
    assert g["edge_type"].shape[0] == g["edge_index"].shape[1]
