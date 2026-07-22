#!/usr/bin/env python
"""Launch M6 grouped 5-fold × seeds {13,42,1337} matrix, then V5 compare_runs.

Each cell: ``python scripts/run_m6_train.py --fold F --seed S --device cuda --fresh``
(epochs from ``configs/train.yaml`` max_epochs=100). Skips cells that already
have a finished ``train_summary.json`` unless ``--force``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 1337])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs" / "m6")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if train_summary.json exists.",
    )
    ap.add_argument(
        "--dry-plan",
        action="store_true",
        help="Print planned cells and exit.",
    )
    args = ap.parse_args()

    cells = [(f, s) for f in args.folds for s in args.seeds]
    print(f"Matrix: {len(cells)} cells  folds={args.folds} seeds={args.seeds}", flush=True)
    if args.dry_plan:
        for f, s in cells:
            print(f"  fold{f}_seed{s}", flush=True)
        return 0

    log_path = args.out_root / "matrix_launch.log"
    args.out_root.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        for i, (fold, seed) in enumerate(cells, start=1):
            run_dir = args.out_root / f"fold{fold}_seed{seed}"
            summary = run_dir / "train_summary.json"
            if summary.is_file() and not args.force:
                msg = f"[{i}/{len(cells)}] SKIP fold={fold} seed={seed} (summary exists)"
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
                continue
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_m6_train.py"),
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--batch-size",
                str(args.batch_size),
                "--device",
                args.device,
                "--fresh",
                "--out-root",
                str(args.out_root),
            ]
            msg = f"[{i}/{len(cells)}] RUN {' '.join(cmd)}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
            t0 = time.perf_counter()
            env = os.environ.copy()
            # Ensure parquet-before-CUDA path inside child as well
            if env.get("CUDA_VISIBLE_DEVICES", "").strip() == "-1":
                del env["CUDA_VISIBLE_DEVICES"]
            proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
            elapsed = time.perf_counter() - t0
            done = f"  exit={proc.returncode} elapsed_h={elapsed/3600:.2f} -> {run_dir}"
            print(done, flush=True)
            log.write(done + "\n")
            log.flush()
            if proc.returncode != 0:
                failures.append(f"fold{fold}_seed{seed}")

    from src.eval.viz.compare_runs import write_v5_compare_runs

    v5 = write_v5_compare_runs(args.out_root)
    print(f"V5 compare_runs: {v5}", flush=True)
    if failures:
        print(f"FAILED cells: {failures}", flush=True)
        return 1
    print("Matrix complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
