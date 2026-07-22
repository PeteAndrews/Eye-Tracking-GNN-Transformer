#!/usr/bin/env python
"""M6 owner training entrypoint — grouped 5-fold over participants.

Agent does not run full training (`.cursorrules`). This script runs one
fold × seed on real episodes with **lazy** disk loading (shared graph cache).

Example (GPU):

  python scripts/run_m6_train.py --fold 0 --seed 13 --epochs 50 --device cuda

Always ``conda activate gnn-gaze`` first.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# Cursor / some IDE shells set CUDA_VISIBLE_DEVICES=-1 which hides the GPU.
# Must clear before importing torch.
_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if _cvd is not None and _cvd.strip() in ("-1",):
    del os.environ["CUDA_VISIBLE_DEVICES"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# CRITICAL (Windows + CUDA torch): warm pandas/pyarrow *before* any torch.cuda
# API. Otherwise the first pd.read_parquet after CUDA init can AV (0xC0000005).
from src.utils.arrow_cuda import warmup_parquet_io

_warmup_sample = (
    ROOT / "data_processed" / "v0_p0" / "fixations" / "P01" / "T01__not_eligible.parquet"
)
warmup_parquet_io(_warmup_sample if _warmup_sample.is_file() else None)

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from src.data.episode_dataset import (
    LazyRealEpisodeDataset,
    collate_episodes,
    discover_real_episodes,
)
from src.data.splits import grouped_participant_folds
from src.train.loop import train_run
from src.train.truncation_stats import truncation_stats_for_keys
from src.utils import io as uio


def _resolve_device(spec: str) -> torch.device:
    spec = (spec or "auto").strip().lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Requested CUDA but torch.cuda.is_available() is False. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}. "
                "Unset it (or set to 0) and retry."
            )
        return torch.device(spec if ":" in spec else "cuda")
    return torch.device(spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--epochs",
        "--epoch",
        type=int,
        default=None,
        dest="epochs",
        help="Max epochs (default: configs/train.yaml max_epochs, else 100).",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Train at most N steps then checkpoint+exit (resume by re-running). "
        "Only auto-enabled on Windows when CUDA is unavailable.",
    )
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoint_last.pt and start a new run (weights from scratch).",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | cuda:0 (default auto)",
    )
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs" / "m6")
    args = ap.parse_args()

    try:
        # Defer CUDA device resolution until after dataset smoke.
        print(
            f"torch {torch.__version__} | cuda_built={torch.version.cuda} | "
            f"(device resolved after smoke)",
            flush=True,
        )

        dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
        tcfg = OmegaConf.load(ROOT / "configs" / "model_transformer.yaml")
        dcfg.max_seq_len = int(tcfg.max_seq_len)
        dcfg.build_pair_relations = bool(tcfg.biases.graph_relation.enabled)
        scfg = OmegaConf.load(ROOT / "configs" / "splits.yaml")
        triples = discover_real_episodes(
            ROOT / str(dcfg.paths.fixations_root),
            max_episodes=args.max_episodes,
        )
        if not triples:
            print("No episodes found", file=sys.stderr)
            return 1

        print(f"Discovered {len(triples)} episodes (lazy load; graph cache <=36).", flush=True)

        graphs_root = ROOT / str(dcfg.paths.graphs_root) / str(dcfg.graph_version)
        keys: list[tuple[str, str, str]] = []
        missing = 0
        for pid, tid, sc in triples:
            if (graphs_root / f"{tid}__{sc}.pt").is_file() and (
                ROOT / str(dcfg.paths.fixations_root) / pid / f"{tid}__{sc}.parquet"
            ).is_file():
                keys.append((pid, tid, sc))
            else:
                missing += 1
        if missing:
            print(f"Skipped {missing} episodes missing graph or parquet.", flush=True)
        print(f"Usable episodes: {len(keys)}", flush=True)

        folds = grouped_participant_folds(
            keys, n_folds=int(scfg.n_folds), seed=int(scfg.seed)
        )
        if args.fold < 0 or args.fold >= len(folds):
            print(f"fold {args.fold} out of range 0..{len(folds)-1}", file=sys.stderr)
            return 1
        fold = folds[args.fold]
        print(
            f"Fold {args.fold}: train n={len(fold['train_idx'])} "
            f"({len(fold['train_participants'])} participants), "
            f"val n={len(fold['val_idx'])} "
            f"({len(fold['val_participants'])} participants)",
            flush=True,
        )

        ds = LazyRealEpisodeDataset(
            keys,
            dataset_cfg=dcfg,
            fixations_root=ROOT / str(dcfg.paths.fixations_root),
            graphs_root=ROOT / str(dcfg.paths.graphs_root),
            graph_version=str(dcfg.graph_version),
            gnn=None,
            seed=args.seed,
        )
        smoke_i = fold["train_idx"][0]
        print(f"Building smoke item keys[{smoke_i}]={keys[smoke_i]} ...", flush=True)
        t_smoke = time.perf_counter()
        try:
            item = ds[smoke_i]
        except Exception:
            print("Smoke item FAILED:", file=sys.stderr)
            traceback.print_exc()
            return 1
        print(
            f"Smoke item OK in {time.perf_counter() - t_smoke:.2f}s "
            f"(T={int(item['tokens'].shape[0])}).",
            flush=True,
        )

        device = _resolve_device(args.device)
        print(
            f"cuda_available={torch.cuda.is_available()} | device={device}",
            flush=True,
        )
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

        train_ds = Subset(ds, fold["train_idx"])
        val_ds = Subset(ds, fold["val_idx"])
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_episodes,
            num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_episodes,
            num_workers=args.num_workers,
        )

        print(f"Starting train setup on {device} ...", flush=True)

        run_dir = args.out_root / f"fold{args.fold}_seed{args.seed}"
        fold_keys = [keys[i] for i in list(fold["train_idx"]) + list(fold["val_idx"])]
        trunc = truncation_stats_for_keys(
            ROOT,
            fold_keys,
            max_seq_len=int(tcfg.max_seq_len),
            fixations_root=ROOT / str(dcfg.paths.fixations_root),
        )
        print(
            f"Truncation guard (cap={trunc['max_seq_len']}): "
            f"episodes {trunc['n_episodes_truncated']}/{trunc['n_episodes']} "
            f"({100 * trunc['frac_episodes_truncated']:.1f}%) truncated; "
            f"fixations discarded {trunc['n_fixations_discarded']}/"
            f"{trunc['n_fixations_total']} "
            f"({100 * trunc['frac_fixations_discarded']:.1f}%)",
            flush=True,
        )

        max_steps = args.max_steps
        if max_steps is None and (os.name == "nt") and (device.type != "cuda"):
            max_steps = 200
            print(
                "Note: CPU on Windows — using --max-steps 200 (AV workaround). "
                "Re-run to resume. With --device cuda this is disabled.",
                flush=True,
            )

        # Resolve epoch ceiling from CLI or configs/train.yaml
        epochs = args.epochs
        if epochs is None:
            t_yml = OmegaConf.load(ROOT / "configs" / "train.yaml")
            epochs = int(getattr(t_yml, "max_epochs", 100) or 100)
        print(f"Starting train ({epochs} epochs) on {device} ...", flush=True)

        result = train_run(
            repo=ROOT,
            train_loader=train_loader,
            val_loader=val_loader,
            run_dir=run_dir,
            seed=args.seed,
            fold=args.fold,
            max_epochs=epochs,
            device=device,
            max_steps=max_steps,
            checkpoint_every=args.checkpoint_every,
            resume=not args.fresh,
            batch_size=args.batch_size,
            truncation_stats=trunc,
        )
        uio.write_json(
            run_dir / "train_summary.json",
            {
                "fold": args.fold,
                "seed": args.seed,
                "n_train": len(fold["train_idx"]),
                "n_val": len(fold["val_idx"]),
                "train_participants": fold["train_participants"],
                "val_participants": fold["val_participants"],
                "best_val": result["best_val"],
                "epochs": result["epochs"],
                "device": str(device),
                "chunked": bool(result.get("chunked")),
                "global_step": result.get("global_step"),
                "truncation": trunc,
                "max_seq_len": int(tcfg.max_seq_len),
                "graph_relation_bias": bool(tcfg.biases.graph_relation.enabled),
            },
        )
        if result.get("chunked"):
            print(
                f"Chunk checkpointed at step {result.get('global_step')} -> {run_dir}",
                flush=True,
            )
        else:
            print(f"Done. best_val={result['best_val']:.4f}  run_dir={run_dir}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
