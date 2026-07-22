"""Minimal V1/V2 HTML report from metrics.jsonl (M6 acceptance smoke)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.utils import io as uio


def write_v1_v2_report(run_dir: Path, history: Sequence[dict[str, Any]]) -> Path:
    """Emit ``viz/report.html`` with training curves + final val losses."""
    run_dir = Path(run_dir)
    viz = run_dir / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    epochs = [h["epoch"] for h in history]
    keys = ["loss_total", "loss_panel", "loss_relation", "loss_ranking"]

    def series(split: str, key: str) -> list[float]:
        return [float(h[split][key]) for h in history]

    rows = []
    for k in keys:
        tr = series("train", k)
        va = series("val", k)
        tr_s = f"{tr[-1]:.4f}" if tr else "—"
        va_s = f"{va[-1]:.4f}" if va else "—"
        rows.append(f"<tr><td>{k}</td><td>{tr_s}</td><td>{va_s}</td></tr>")

    # Simple SVG polylines for total loss
    def svg_curve(vals: list[float], color: str, w: int = 420, h: int = 160) -> str:
        if not vals:
            return ""
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        pts = []
        for i, v in enumerate(vals):
            x = 10 + (w - 20) * (i / max(len(vals) - 1, 1))
            y = h - 10 - (h - 20) * ((v - lo) / span)
            pts.append(f"{x:.1f},{y:.1f}")
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}" />'

    svg = (
        f'<svg width="440" height="180" xmlns="http://www.w3.org/2000/svg">'
        f"{svg_curve(series('train', 'loss_total'), '#1f77b4')}"
        f"{svg_curve(series('val', 'loss_total'), '#ff7f0e')}"
        f"</svg>"
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><title>M6 run report</title>
<style>
body {{ font-family: Georgia, serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ccc; padding: 0.35rem 0.7rem; }}
.legend span {{ display: inline-block; width: 12px; height: 12px; margin-right: 4px; }}
</style></head><body>
<h1>M6 training report (V1 + V2 smoke)</h1>
<p>Run dir: <code>{run_dir.as_posix()}</code> · epochs: {len(epochs)}</p>
<h2>V1 — Training dynamics</h2>
<p class="legend"><span style="background:#1f77b4"></span>train total
<span style="background:#ff7f0e;margin-left:1rem"></span>val total</p>
{svg}
<h2>V2 — Final predictive losses (proxy)</h2>
<table><tr><th>loss</th><th>train</th><th>val</th></tr>
{''.join(rows)}
</table>
<p><em>Full PR / confusion / ranking panels land after owner full training runs.</em></p>
</body></html>
"""
    out = viz / "report.html"
    uio.write_text(out, html)
    return out
