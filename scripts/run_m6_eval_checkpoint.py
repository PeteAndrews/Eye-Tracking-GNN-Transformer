#!/usr/bin/env python
"""Evaluate a frozen M6 checkpoint on its grouped-val fold (go/no-go metrics)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
if _cvd is not None and _cvd.strip() in ("-1",):
    del os.environ["CUDA_VISIBLE_DEVICES"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.arrow_cuda import warmup_parquet_io

_sample = ROOT / "data_processed" / "v0_p0" / "fixations" / "P01" / "T01__not_eligible.parquet"
warmup_parquet_io(_sample if _sample.is_file() else None)

import torch

from src.eval.m6_predictive import evaluate_checkpoint, write_eval_report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs" / "m6" / "fold0_seed13" / "checkpoint_best.pt",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs" / "m6" / "fold0_seed13" / "eval",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device={device} torch={torch.__version__}", flush=True)
    summary = evaluate_checkpoint(
        ROOT,
        args.checkpoint,
        fold=args.fold,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
        operating_threshold=args.threshold,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_eval_report(
        summary,
        args.out_dir / "predictive_metrics.json",
        args.out_dir / "predictive_metrics.md",
    )
    gate = summary.get("semantic_candidate_gate") or {}
    print(f"Wrote {args.out_dir / 'predictive_metrics.md'}", flush=True)
    print(
        f"GO/NO-GO={summary['go_nogo']}  "
        f"SEMANTIC AP={gate.get('ap')} baseline={gate.get('ap_baseline')}",
        flush=True,
    )
    return 0 if summary["go_nogo"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
