#!/usr/bin/env python
"""M6 dry-run: fixture overfit path + short train loop (not full corpus training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.episode_dataset import EpisodeDataset, collate_episodes, load_fixture_episode
from src.train.loop import train_run
from src.utils import io as uio


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "m6_dryrun")
    args = ap.parse_args()

    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    eps = [
        load_fixture_episode(ROOT, "fx01_T99"),
        load_fixture_episode(ROOT, "fx02_T98_star_on"),
    ]
    ds = EpisodeDataset(eps, dataset_cfg=dcfg, gnn=None, seed=args.seed)
    # Tiny "grouped" split: both in train and val for dry-run smoke
    loader = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=collate_episodes)
    val_loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_episodes)

    result = train_run(
        repo=ROOT,
        train_loader=loader,
        val_loader=val_loader,
        run_dir=args.out,
        seed=args.seed,
        fold=0,
        max_epochs=args.epochs,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        tracker_backend="none",
    )
    summary = {
        "best_val": result["best_val"],
        "epochs": result["epochs"],
        "run_dir": result["run_dir"],
        "note": "Dry-run on 2 fixtures only — owner must run full grouped 5-fold training.",
    }
    uio.write_json(Path(args.out) / "dryrun_summary.json", summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
