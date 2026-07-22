"""V5 cross-run comparison from ``runs/m6/fold*_seed*/`` artefacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.utils import io as uio


def _load_run(run_dir: Path) -> Optional[dict[str, Any]]:
    summary_p = run_dir / "train_summary.json"
    metrics_p = run_dir / "metrics.jsonl"
    if not summary_p.is_file() or not metrics_p.is_file():
        return None
    summary = uio.read_json(summary_p)
    rows = []
    for line in metrics_p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return None
    best = min(rows, key=lambda r: float(r["val"]["loss_total"]))
    last = rows[-1]
    return {
        "run_dir": str(run_dir),
        "fold": summary.get("fold"),
        "seed": summary.get("seed"),
        "best_epoch": best["epoch"],
        "best_val_total": float(best["val"]["loss_total"]),
        "best_val_panel": float(best["val"]["loss_panel"]),
        "best_val_relation": float(best["val"]["loss_relation"]),
        "best_val_ranking": float(best["val"]["loss_ranking"]),
        "final_epoch": last["epoch"],
        "final_train_total": float(last["train"]["loss_total"]),
        "final_val_total": float(last["val"]["loss_total"]),
        "train_val_gap": float(last["val"]["loss_total"]) - float(last["train"]["loss_total"]),
        "n_epochs_logged": len(rows),
        "device": summary.get("device"),
    }


def collect_m6_runs(runs_root: Path) -> list[dict[str, Any]]:
    runs_root = Path(runs_root)
    out = []
    for d in sorted(runs_root.glob("fold*_seed*")):
        if d.is_dir():
            row = _load_run(d)
            if row is not None:
                out.append(row)
    return out


def write_v5_compare_runs(runs_root: Path, out_dir: Optional[Path] = None) -> Path:
    """Emit ``viz/v5_compare_runs.html`` (+ JSON) across completed fold×seed runs."""
    runs_root = Path(runs_root)
    out_dir = Path(out_dir) if out_dir else runs_root / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_m6_runs(runs_root)
    uio.write_json(out_dir / "v5_compare_runs.json", {"n": len(rows), "runs": rows})

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    by_fold: dict[Any, list[float]] = {}
    by_seed: dict[Any, list[float]] = {}
    for r in rows:
        by_fold.setdefault(r["fold"], []).append(r["best_val_total"])
        by_seed.setdefault(r["seed"], []).append(r["best_val_total"])

    tr_rows = []
    for r in rows:
        tr_rows.append(
            "<tr>"
            f"<td>{r['fold']}</td><td>{r['seed']}</td>"
            f"<td>{r['best_epoch']}</td>"
            f"<td>{r['best_val_total']:.4f}</td>"
            f"<td>{r['best_val_panel']:.4f}</td>"
            f"<td>{r['best_val_relation']:.4f}</td>"
            f"<td>{r['best_val_ranking']:.4f}</td>"
            f"<td>{r['train_val_gap']:.4f}</td>"
            f"<td>{r['final_epoch']}</td>"
            "</tr>"
        )

    fold_sum = "".join(
        f"<li>fold {k}: mean best val = {mean(v):.4f} (n={len(v)})</li>"
        for k, v in sorted(by_fold.items(), key=lambda x: (x[0] is None, x[0]))
    )
    seed_sum = "".join(
        f"<li>seed {k}: mean best val = {mean(v):.4f} (n={len(v)})</li>"
        for k, v in sorted(by_seed.items(), key=lambda x: (x[0] is None, x[0]))
    )
    overall = mean([r["best_val_total"] for r in rows])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>V5 compare_runs</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ccc; padding: 0.35rem 0.7rem; }}
</style></head><body>
<h1>V5 — Cross-run comparison (M6 matrix)</h1>
<p>Root: <code>{runs_root.as_posix()}</code> · completed runs: <b>{len(rows)}</b>
· overall mean best val: <b>{overall:.4f}</b></p>
<h2>By fold</h2><ul>{fold_sum or "<li>none</li>"}</ul>
<h2>By seed</h2><ul>{seed_sum or "<li>none</li>"}</ul>
<h2>All runs</h2>
<table>
<tr><th>fold</th><th>seed</th><th>best_ep</th><th>best_val</th>
<th>panel</th><th>relation</th><th>ranking</th><th>gap@final</th><th>final_ep</th></tr>
{"".join(tr_rows)}
</table>
<p><em>Per-run V1+V2 panels live under each <code>fold*_seed*/viz/report.html</code>.</em></p>
</body></html>
"""
    out = out_dir / "v5_compare_runs.html"
    uio.write_text(out, html)
    return out
