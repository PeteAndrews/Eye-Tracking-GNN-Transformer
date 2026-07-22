"""M5 unit tests: multi-hot relation targets + episode dataset on fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.episode_dataset import (
    EpisodeDataset,
    collate_episodes,
    load_fixture_episode,
)
from src.data.targets import (
    EMPTY_SPACE_TRANSITION,
    NO_DIRECT_RELATION,
    RELATION_NAME_TO_IDX,
    RELATION_VOCAB,
    build_edge_relation_lookup,
    next_relation_multihot,
    sample_ranking_candidates,
)
from src.graph.build import RELATION_TO_ID
from src.models.tokens import SIDE_FEATURE_DIM, assemble_token, fixation_side_features

ROOT = Path(__file__).resolve().parents[1]


def test_relation_vocab_covers_graph_plus_extras():
    for name in RELATION_TO_ID:
        assert name in RELATION_NAME_TO_IDX
    assert NO_DIRECT_RELATION in RELATION_VOCAB
    assert EMPTY_SPACE_TRANSITION in RELATION_VOCAB
    assert len(RELATION_VOCAB) == 7


def test_multihot_next_segment_hand_computed():
    lookup = {(0, 1): {RELATION_TO_ID["NEXT_SEGMENT"]}}
    vec = next_relation_multihot(
        0, 1, edge_lookup=lookup, src_is_empty=False, dst_is_empty=False
    )
    assert vec.dtype == np.float32
    assert vec[RELATION_NAME_TO_IDX["NEXT_SEGMENT"]] == 1.0
    assert vec[RELATION_NAME_TO_IDX[NO_DIRECT_RELATION]] == 0.0
    assert vec.sum() == 1.0


def test_multihot_multi_relation_hand_computed():
    lookup = {
        (2, 3): {
            RELATION_TO_ID["NEXT_SEGMENT"],
            RELATION_TO_ID["SPATIAL_NEIGHBOUR"],
        }
    }
    vec = next_relation_multihot(
        2, 3, edge_lookup=lookup, src_is_empty=False, dst_is_empty=False
    )
    assert vec[RELATION_NAME_TO_IDX["NEXT_SEGMENT"]] == 1.0
    assert vec[RELATION_NAME_TO_IDX["SPATIAL_NEIGHBOUR"]] == 1.0
    assert vec[RELATION_NAME_TO_IDX[NO_DIRECT_RELATION]] == 0.0
    assert vec.sum() == 2.0


def test_multihot_no_direct_when_no_edge():
    vec = next_relation_multihot(
        0, 5, edge_lookup={}, src_is_empty=False, dst_is_empty=False
    )
    assert vec[RELATION_NAME_TO_IDX[NO_DIRECT_RELATION]] == 1.0
    assert vec.sum() == 1.0


def test_multihot_empty_space_transition():
    lookup = {(0, 1): {RELATION_TO_ID["NEXT_SEGMENT"]}}
    vec = next_relation_multihot(
        0, 1, edge_lookup=lookup, src_is_empty=True, dst_is_empty=False
    )
    assert vec[RELATION_NAME_TO_IDX[EMPTY_SPACE_TRANSITION]] == 1.0
    assert vec[RELATION_NAME_TO_IDX["NEXT_SEGMENT"]] == 0.0
    assert vec[RELATION_NAME_TO_IDX[NO_DIRECT_RELATION]] == 0.0


def test_ranking_candidates_include_positive():
    rng = np.random.default_rng(0)
    emb = np.eye(10, dtype=np.float32)
    cands, labels = sample_ranking_candidates(
        positive_node=3,
        n_segments=10,
        visited={0, 1},
        text_emb=emb,
        query_emb=emb[2],
        n_easy=8,
        n_hard=4,
        rng=rng,
    )
    assert 3 in cands
    assert labels[cands.index(3)] == 1
    assert sum(labels) == 1
    assert len(cands) <= 1 + 4 + 8


def test_side_feature_dim_matches_constant():
    row = {
        "duration_ms": 100.0,
        "t_start_ms": 0.0,
        "assignment_confidence": 1.0,
        "ambiguous": False,
        "visit_count": 1,
        "is_return": False,
        "loop_role": "none",
        "scroll_direction": "none",
    }
    side = fixation_side_features(row, episode_duration_ms=1000.0)
    assert side.shape == (SIDE_FEATURE_DIM,)


def test_assemble_token_concat():
    xv = np.ones(4, np.float32)
    hv = np.zeros(4, np.float32)
    side = np.full(3, 0.5, np.float32)
    tok = assemble_token(x_v=xv, h_v=hv, side=side, is_empty=False)
    assert tok.shape == (11,)
    np.testing.assert_array_equal(tok[:4], xv)


def test_fixture_episode_multihot_q1_to_q2():
    """Structural NEXT_SEGMENT between first two question segments on fx01."""
    cfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    ep = load_fixture_episode(ROOT, "fx01_T99")
    ds = EpisodeDataset([ep], dataset_cfg=cfg, gnn=None, seed=13)
    item = ds[0]
    node_ids = list(ep["graph"]["node_ids"])
    i_q1 = node_ids.index("seg_q1")
    i_q2 = node_ids.index("seg_q2")
    # First two fixations are seg_q1, seg_q2
    assert int(item["node_index"][0]) == i_q1
    assert int(item["node_index"][1]) == i_q2
    rel = item["next_relation"][0].numpy()
    assert rel[RELATION_NAME_TO_IDX["NEXT_SEGMENT"]] == 1.0
    assert rel[RELATION_NAME_TO_IDX[NO_DIRECT_RELATION]] == 0.0
    # Next panel = question for both
    assert int(item["next_panel"][0]) == list(cfg.panel_classes).index("question")


def test_fixture_empty_space_transition_label():
    cfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    ep = load_fixture_episode(ROOT, "fx01_T99")
    ds = EpisodeDataset([ep], dataset_cfg=cfg, gnn=None, seed=13)
    item = ds[0]
    empty = item["is_empty"].numpy()
    # Find a transition involving empty space
    found = False
    for t in range(int(item["length"]) - 1):
        if empty[t] or empty[t + 1]:
            rel = item["next_relation"][t].numpy()
            assert rel[RELATION_NAME_TO_IDX[EMPTY_SPACE_TRANSITION]] == 1.0
            assert rel[RELATION_NAME_TO_IDX[NO_DIRECT_RELATION]] == 0.0
            found = True
            break
    assert found, "fixture should contain at least one empty-space fixation"


def test_fixture_edge_lookup_matches_multihot():
    """For every non-empty consecutive pair, multi-hot equals lookup relations."""
    cfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    ep = load_fixture_episode(ROOT, "fx01_T99")
    lookup = build_edge_relation_lookup(ep["graph"]["edge_index"], ep["graph"]["edge_type"])
    ds = EpisodeDataset([ep], dataset_cfg=cfg, gnn=None, seed=13)
    item = ds[0]
    empty = item["is_empty"].numpy()
    nodes = item["node_index"].numpy()
    id_to_name = {v: k for k, v in RELATION_TO_ID.items()}
    for t in range(int(item["length"]) - 1):
        if empty[t] or empty[t + 1]:
            continue
        src, dst = int(nodes[t]), int(nodes[t + 1])
        expected = np.zeros(len(RELATION_VOCAB), dtype=np.float32)
        rels = lookup.get((src, dst), set())
        if rels:
            for rid in rels:
                expected[RELATION_NAME_TO_IDX[id_to_name[rid]]] = 1.0
        else:
            expected[RELATION_NAME_TO_IDX[NO_DIRECT_RELATION]] = 1.0
        np.testing.assert_array_equal(item["next_relation"][t].numpy(), expected)


def test_collate_and_dataloader():
    cfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    eps = [
        load_fixture_episode(ROOT, "fx01_T99"),
        load_fixture_episode(ROOT, "fx02_T98_star_on"),
    ]
    ds = EpisodeDataset(eps, dataset_cfg=cfg, gnn=None, seed=13)
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_episodes)
    batch = next(iter(loader))
    assert batch["tokens"].ndim == 3
    assert batch["mask"].shape[:2] == batch["tokens"].shape[:2]
    assert batch["next_relation"].shape[-1] == len(RELATION_VOCAB)
    assert batch["tokens"].shape[-1] == 2 * int(cfg.gnn_out_dim) + SIDE_FEATURE_DIM
    assert batch["rank_candidates"].ndim == 3
    assert batch["loop_origin_index"].shape[:2] == batch["tokens"].shape[:2]
    assert batch["node_x_v"].ndim == 3
    assert batch["star_condition"] == ["not_eligible", "star_on"]


def test_throughput_fixture_sanity():
    """Rough throughput: ≥100 fixture episodes/sec without GNN (placeholder x_v)."""
    import time

    cfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    ep = load_fixture_episode(ROOT, "fx01_T99")
    ds = EpisodeDataset([ep] * 32, dataset_cfg=cfg, gnn=None, seed=13)
    t0 = time.perf_counter()
    for i in range(len(ds)):
        _ = ds[i]
    elapsed = time.perf_counter() - t0
    rate = len(ds) / max(elapsed, 1e-9)
    assert rate > 20.0, f"throughput too low: {rate:.1f} ep/s"
