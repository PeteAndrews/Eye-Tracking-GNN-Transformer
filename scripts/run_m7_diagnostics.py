#!/usr/bin/env python
"""M7 diagnostic gate on a frozen M6 checkpoint (D1–D3 + fixation/visit slice)."""

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

from src.eval.loop_diagnostics import run_m7_gate, write_m7_report
from src.utils import io as uio


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
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <checkpoint_dir>/m7_diagnostics",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or (args.checkpoint.parent / "m7_diagnostics")
    device = torch.device(args.device)
    print(f"device={device} ckpt={args.checkpoint}", flush=True)

    summary = run_m7_gate(
        ROOT,
        args.checkpoint,
        fold=args.fold,
        seed=args.seed,
        batch_size=args.batch_size,
        device=device,
    )
    json_path, md_path = write_m7_report(summary, out_dir)
    # Prefer a distinct reports/ name when out-dir is not the default m7_diagnostics/
    report_tag = "diagnostics"
    if out_dir.name and out_dir.name != "m7_diagnostics":
        report_tag = out_dir.name.replace("m7_", "")
    reports_md = ROOT / "reports" / f"m7_fold{args.fold}_seed{args.seed}_{report_tag}.md"
    uio.write_text(reports_md, uio.read_text(md_path))
    uio.write_json(
        ROOT / "reports" / f"m7_fold{args.fold}_seed{args.seed}_{report_tag}.json",
        summary,
    )
    print(f"Wrote {md_path}", flush=True)
    print(f"Copied {reports_md}", flush=True)
    print(f"Decision: {summary['decision']}", flush=True)
    print(
        f"Gates D1={summary['gates']['D1']} D2={summary['gates']['D2']} "
        f"D3={summary['gates']['D3']}",
        flush=True,
    )
    return 0 if summary["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
