"""Lazy real-episode dataset smoke test."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from src.data.episode_dataset import LazyRealEpisodeDataset, discover_real_episodes

ROOT = Path(__file__).resolve().parents[1]


def test_lazy_dataset_loads_one_real_episode():
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    fix_root = ROOT / str(dcfg.paths.fixations_root)
    if not fix_root.is_dir():
        return
    keys = discover_real_episodes(fix_root, max_episodes=3)
    if not keys:
        return
    ds = LazyRealEpisodeDataset(
        keys,
        dataset_cfg=dcfg,
        fixations_root=fix_root,
        graphs_root=ROOT / str(dcfg.paths.graphs_root),
        graph_version=str(dcfg.graph_version),
        gnn=None,
        seed=13,
    )
    item = ds[0]
    assert item["tokens"].ndim == 2
    assert item["length"] >= 1
    # Second access should hit graph cache
    _ = ds[1] if len(ds) > 1 else ds[0]
    assert len(ds._graph_cache) >= 1
