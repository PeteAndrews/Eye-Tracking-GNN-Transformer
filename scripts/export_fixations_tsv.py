#!/usr/bin/env python3
"""Emit UTF-8 TSV companions next to existing P6 fixation parquet files.

Use after a parquet-only P6 run, or to refresh TSVs without rebuilding events.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.aoi_injection import write_gaze_table  # noqa: E402


def main() -> int:
    cfg = OmegaConf.load(ROOT / "configs" / "data.yaml")
    out_dir = ROOT / str(cfg.paths.processed_root) / str(cfg.data_version) / "fixations"
    if not out_dir.is_dir():
        print(f"Missing {out_dir}", file=sys.stderr)
        return 1
    n = 0
    for pq in sorted(out_dir.rglob("*.parquet")):
        # Skip derived tables that are not episode fixations? Export all.
        df = pd.read_parquet(pq)
        write_gaze_table(pq.parent / pq.stem, df)
        n += 1
        rel = pq.relative_to(out_dir)
        print(f"wrote {rel.with_suffix('.tsv')} ({len(df)} rows)")
    print(f"done: {n} tables")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
