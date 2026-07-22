"""Report return-within-H positive rates at candidate horizons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.eval.loop_diagnostics import return_within_horizon_labels
from src.utils.arrow_cuda import read_parquet, warmup_parquet_io

FR = ROOT / "data_processed" / "v0_p0" / "fixations"


def main() -> None:
    warmup_parquet_io(FR / "P01" / "T01__not_eligible.parquet")
    horizons = [5, 8, 10, 12, 15, 20, 30]
    stats = {h: {"n": 0, "pos": 0} for h in horizons}
    for pq in sorted(FR.glob("*/*.parquet")):
        df = read_parquet(pq)
        sid = df["segment_id"].astype(str).fillna("").to_numpy()
        empty = np.array([s in ("", "None", "nan", "NaN") for s in sid], dtype=bool)
        if "empty_space_category" in df.columns:
            esc = df["empty_space_category"].astype(str).fillna("").to_numpy()
            empty |= (esc != "") & (~np.isin(esc, ["None", "nan", "NaN"]))
        codes: dict[str, int] = {}
        arr = np.empty(len(sid), dtype=np.int64)
        for i, s in enumerate(sid):
            if empty[i]:
                arr[i] = -1
            else:
                if s not in codes:
                    codes[s] = len(codes)
                arr[i] = codes[s]
        for h in horizons:
            lab = return_within_horizon_labels(arr, horizon=h)
            ok = lab >= 0
            stats[h]["n"] += int(ok.sum())
            stats[h]["pos"] += int((lab[ok] == 1).sum())

    rows = []
    for h in horizons:
        n = stats[h]["n"]
        p = stats[h]["pos"] / n if n else float("nan")
        pw = ((1.0 - p) / p) if p and p < 1 else float("nan")
        rows.append(
            {
                "horizon": h,
                "n": n,
                "pos_rate": round(float(p), 4),
                "balance_pos_weight": round(float(pw), 4),
            }
        )
        print(
            f"H={h:2d}  n={n}  pos_rate={p:.4f}  balance_pos_weight={pw:.4f}",
            flush=True,
        )
    out = ROOT / "reports" / "return_horizon_balance.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
