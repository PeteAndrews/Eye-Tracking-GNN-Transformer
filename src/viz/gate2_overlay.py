"""P7 Visual Gate 2 — assignment validation overlays (extends Gate 1 tooling)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from omegaconf import OmegaConf

from src.utils import io as uio
from src.viz.overlay_check import (
    encode_image_jpeg,
    image_stem,
    select_stratified_sample,
    _rect_shape,
)


PANEL_COLORS = {
    "question": "#1f77b4",
    "response": "#2ca02c",
    "mark_scheme": "#ff7f0e",
    "commentary": "#9467bd",
    "star_chart": "#d62728",
    "ui": "#7f7f7f",
    "answer_scroll_bar": "#17becf",
    "commentary_scroll_bar": "#e377c2",
    "ui_general": "#bcbd22",
    "outside_document": "#aaaaaa",
}


def flag_qc_episodes(
    qc: pd.DataFrame,
    *,
    pct_empty: float = 40.0,
    pct_ambiguous: float = 40.0,
    mean_confidence_below: float = 0.2,
) -> list[dict[str, str]]:
    mask = (
        (qc["pct_empty_space"] > pct_empty)
        | (qc["pct_ambiguous"] > pct_ambiguous)
        | (qc["mean_confidence"] < mean_confidence_below)
    )
    rows = []
    for r in qc.loc[mask].itertuples(index=False):
        rows.append(
            {
                "participant_id": str(r.participant_id),
                "trial_id": str(r.trial_id),
                "star_condition": str(r.star_condition),
                "flag_reason": "p6_qc",
            }
        )
    return rows


def distance_to_edge_hist(
    fix: pd.DataFrame,
    segments: list[dict[str, Any]],
    *,
    bins: Optional[list[float]] = None,
) -> dict[str, int]:
    """Histogram of min distance-to-edge for assigned (non-empty) fixations."""
    bins = bins or [0, 2, 5, 10, 20, 40, 80, 1e9]
    seg_g = {
        s["segment_id"]: s["geometry"]
        for s in segments
        if s.get("segment_id") and s.get("geometry")
    }
    dists = []
    for r in fix.itertuples(index=False):
        sid = getattr(r, "segment_id", None)
        if sid is None or (isinstance(sid, float) and np.isnan(sid)) or str(sid) == "None":
            continue
        g = seg_g.get(str(sid))
        if not g:
            continue
        x, y = float(r.x_doc), float(r.y_doc)
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        d = min(
            x - float(g["x_min"]),
            float(g["x_max"]) - x,
            y - float(g["y_min"]),
            float(g["y_max"]) - y,
        )
        dists.append(max(0.0, d))
    hist = {f"{bins[i]}-{bins[i+1]}": 0 for i in range(len(bins) - 1)}
    keys = list(hist.keys())
    for d in dists:
        for i in range(len(bins) - 1):
            if bins[i] <= d < bins[i + 1]:
                hist[keys[i]] += 1
                break
    return hist


def panel_vs_aoi_counts(
    fix: pd.DataFrame,
    gaze: pd.DataFrame,
    aoi_panel_map: dict[str, str],
) -> dict[str, Any]:
    fix_panels = fix["panel_label"].astype(str).value_counts().to_dict()
    raw = gaze["aoi_label"].astype(str)
    mapped_counts: dict[str, int] = {}
    for lab, n in raw.value_counts().items():
        key = aoi_panel_map.get(str(lab), "unmapped")
        mapped_counts[str(key)] = mapped_counts.get(str(key), 0) + int(n)
    return {
        "assignment_panel_counts": {str(k): int(v) for k, v in fix_panels.items()},
        "export_aoi_mapped_panel_counts": mapped_counts,
        "export_aoi_label_counts": {str(k): int(v) for k, v in raw.value_counts().items()},
    }


def build_gate2_figure(
    *,
    image_uri: str,
    w_doc: int,
    h_doc: int,
    segments: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    fix: pd.DataFrame,
    epsilon: float,
    panel_colors: dict[str, str],
    title: str,
) -> go.Figure:
    seg_by_id = {s["segment_id"]: s for s in segments if s.get("segment_id")}
    shapes: list[dict[str, Any]] = []
    for s in segments:
        g = s["geometry"]
        color = panel_colors.get(s.get("panel_label"), "#444")
        shapes.append(
            _rect_shape(
                float(g["x_min"]),
                float(g["y_min"]),
                float(g["x_max"]),
                float(g["y_max"]),
                color=color,
                width=1.0,
            )
        )
    for p in panels:
        pl = str(p.get("panel_label") or "")
        color = panel_colors.get(pl, "#888")
        shapes.append(
            _rect_shape(
                float(p["x_min"]),
                float(p["y_min"]),
                float(p["x_max"]),
                float(p["y_max"]),
                color=color,
                width=1.5,
                dash="dash",
            )
        )

    assigned = fix[fix["segment_id"].notna()].copy() if "segment_id" in fix.columns else fix.iloc[0:0]
    empty = fix[fix["segment_id"].isna()].copy() if "segment_id" in fix.columns else fix
    if "ambiguous" in fix.columns:
        amb = fix[fix["ambiguous"].fillna(False).astype(bool)].copy()
    else:
        amb = fix.iloc[0:0]

    def _colors(df: pd.DataFrame) -> list[str]:
        return [panel_colors.get(str(p), "#333") for p in df["panel_label"].astype(str)]

    fig = go.Figure()
    if len(assigned):
        fig.add_trace(
            go.Scatter(
                x=assigned["x_doc"],
                y=assigned["y_doc"],
                mode="markers",
                name="Assigned",
                marker=dict(
                    size=np.clip(np.sqrt(assigned["duration_ms"].to_numpy(dtype=float) / 8.0), 5, 22),
                    color=_colors(assigned),
                    opacity=0.75,
                    line=dict(width=0.4, color="#222"),
                ),
                customdata=np.stack(
                    [
                        assigned["fixation_id"].astype(str),
                        assigned["segment_id"].astype(str),
                        assigned["assignment_confidence"].to_numpy(dtype=float),
                        assigned["panel_label"].astype(str),
                    ],
                    axis=1,
                ),
                hovertemplate=(
                    "%{customdata[0]}<br>(%{x:.0f},%{y:.0f})<br>"
                    "seg=%{customdata[1]} conf=%{customdata[2]:.2f}<br>"
                    "%{customdata[3]}<extra></extra>"
                ),
            )
        )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], name="Assigned", mode="markers"))

    if len(empty):
        fig.add_trace(
            go.Scatter(
                x=empty["x_doc"],
                y=empty["y_doc"],
                mode="markers",
                name="Empty-space",
                marker=dict(size=8, symbol="x", color=_colors(empty), opacity=0.85),
                customdata=empty["empty_space_category"].astype(str),
                hovertemplate="empty %{customdata}<br>(%{x:.0f},%{y:.0f})<extra></extra>",
            )
        )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], name="Empty-space", mode="markers"))

    if len(amb):
        fig.add_trace(
            go.Scatter(
                x=amb["x_doc"],
                y=amb["y_doc"],
                mode="markers",
                name="Ambiguous",
                marker=dict(size=12, symbol="diamond", color="#ff00aa", opacity=0.9, line=dict(width=1, color="#000")),
                customdata=np.stack(
                    [
                        amb["segment_id"].astype(str),
                        amb["segment_id_alt"].astype(str) if "segment_id_alt" in amb.columns else ["?"] * len(amb),
                    ],
                    axis=1,
                ),
                hovertemplate="ambig %{customdata[0]} / alt %{customdata[1]}<extra></extra>",
            )
        )
    else:
        fig.add_trace(go.Scatter(x=[], y=[], name="Ambiguous", mode="markers"))

    # Current-fixation highlight layer (updated by slider frames)
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode="markers",
            name="Current",
            marker=dict(size=16, symbol="circle-open", color="#ffff00", line=dict(width=3, color="#ffff00")),
        )
    )

    # Slider frames: highlight current fixation + assigned/alt boxes + ε ring
    n = len(fix)
    max_steps = min(n, 80)
    idxs = np.linspace(0, n - 1, max_steps).astype(int) if n else []
    frames = []
    for step_i, fi in enumerate(idxs):
        row = fix.iloc[int(fi)]
        x, y = float(row.x_doc), float(row.y_doc)
        extra_shapes = list(shapes)
        sid = row.segment_id if pd.notna(row.segment_id) else None
        if sid and str(sid) in seg_by_id:
            g = seg_by_id[str(sid)]["geometry"]
            extra_shapes.append(
                _rect_shape(
                    float(g["x_min"]),
                    float(g["y_min"]),
                    float(g["x_max"]),
                    float(g["y_max"]),
                    color="#ffff00",
                    width=4.0,
                )
            )
        alt = getattr(row, "segment_id_alt", None)
        if alt is not None and pd.notna(alt) and str(alt) in seg_by_id:
            g = seg_by_id[str(alt)]["geometry"]
            extra_shapes.append(
                _rect_shape(
                    float(g["x_min"]),
                    float(g["y_min"]),
                    float(g["x_max"]),
                    float(g["y_max"]),
                    color="#ff00aa",
                    width=3.0,
                    dash="dot",
                )
            )
        if bool(getattr(row, "edge_zone", False)) and np.isfinite(x) and np.isfinite(y):
            extra_shapes.append(
                {
                    "type": "circle",
                    "xref": "x",
                    "yref": "y",
                    "x0": x - epsilon,
                    "y0": y - epsilon,
                    "x1": x + epsilon,
                    "y1": y + epsilon,
                    "line": {"color": "#ffcc00", "width": 2, "dash": "dash"},
                    "fillcolor": "rgba(255,204,0,0.08)",
                }
            )
        conf = float(row.assignment_confidence) if pd.notna(row.assignment_confidence) else 0.0
        scroll_dir = str(getattr(row, "scroll_direction", "none"))
        annot = (
            f"{row.fixation_id} | seg={sid} | conf={conf:.2f} | "
            f"ambig={bool(getattr(row, 'ambiguous', False))} | scroll={scroll_dir}"
        )
        frames.append(
            go.Frame(
                name=str(step_i),
                data=[go.Scatter(x=[x], y=[y])],
                traces=[3],
                layout=go.Layout(
                    shapes=extra_shapes,
                    annotations=[
                        dict(
                            text=annot,
                            xref="paper",
                            yref="paper",
                            x=0.0,
                            y=1.08,
                            showarrow=False,
                            font=dict(size=12),
                            align="left",
                        )
                    ],
                ),
            )
        )
    fig.frames = frames

    fig.update_layout(
        title=title,
        width=min(1200, w_doc + 80),
        height=min(900, int(h_doc * (1200 / max(w_doc, 1))) + 140),
        xaxis=dict(range=[0, w_doc], scaleanchor="y", scaleratio=1, title="x_doc (px)"),
        yaxis=dict(range=[h_doc, 0], title="y_doc (px)"),
        shapes=shapes,
        images=[
            dict(
                source=image_uri,
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=w_doc,
                sizey=h_doc,
                sizing="stretch",
                opacity=1.0,
                layer="below",
            )
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=100, b=40),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.16,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"mode": "immediate", "frame": {"duration": 0}}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                pad={"t": 30},
                currentvalue={"prefix": "Fixation step: "},
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            {"mode": "immediate", "frame": {"duration": 0, "redraw": True}},
                        ],
                        label=str(i + 1),
                    )
                    for i in range(len(frames))
                ],
            )
        ]
        if frames
        else [],
    )
    return fig


def render_gate2_html(
    *,
    participant_id: str,
    trial_id: str,
    star_condition: str,
    image_path: Path,
    segments: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    fix: pd.DataFrame,
    gaze: pd.DataFrame,
    episode_qc: Optional[dict[str, Any]],
    epsilon: float,
    gate_cfg: dict[str, str],
    aoi_panel_map: dict[str, str],
    out_path: Path,
    flagged: bool = False,
) -> dict[str, Any]:
    quality = int(gate_cfg.get("jpeg_quality") or 70)
    uri, w_doc, h_doc = encode_image_jpeg(image_path, quality=quality)
    panel_colors = dict(gate_cfg.get("panel_colors") or PANEL_COLORS)
    # merge defaults
    for k, v in PANEL_COLORS.items():
        panel_colors.setdefault(k, v)

    fig = build_gate2_figure(
        image_uri=uri,
        w_doc=w_doc,
        h_doc=h_doc,
        segments=segments,
        panels=panels,
        fix=fix,
        epsilon=epsilon,
        panel_colors=panel_colors,
        title=f"Gate 2 · {participant_id} · {trial_id} · {star_condition}",
    )
    hist = distance_to_edge_hist(fix, segments)
    compare = panel_vs_aoi_counts(fix, gaze, aoi_panel_map)

    qc = episode_qc or {}
    hist_rows = "".join(
        f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>" for k, v in hist.items()
    )
    assign_rows = "".join(
        f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>"
        for k, v in sorted(compare["assignment_panel_counts"].items(), key=lambda kv: -kv[1])
    )
    aoi_rows = "".join(
        f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>"
        for k, v in sorted(compare["export_aoi_mapped_panel_counts"].items(), key=lambda kv: -kv[1])
    )

    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    flag_note = "<p class='flag'>Flagged by P6 QC thresholds.</p>" if flagged else ""
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Gate 2 — {participant_id} {trial_id}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1rem 1.5rem;color:#222}}
.grid{{display:grid;grid-template-columns:1fr 340px;gap:1rem}}
table{{border-collapse:collapse;width:100%;margin:.4rem 0;font-size:.9rem}}
th,td{{border:1px solid #ccc;padding:.25rem .4rem}} th{{background:#f4f4f4;text-align:left}}
.note{{color:#666;font-size:.85rem}} .flag{{color:#a00;font-weight:600}}
h1{{font-size:1.2rem}} h2{{font-size:1rem;margin-top:1rem}}
</style></head><body>
<h1>Visual Gate 2 — fixation→segment assignment</h1>
{flag_note}
<p><strong>{participant_id}</strong> · <strong>{trial_id}</strong> · <strong>{star_condition}</strong>
 · ε={epsilon:.3f}px · <code>{image_path.name}</code></p>
<p class="note">Yellow highlight = current assigned segment; pink dotted = ambiguous alt;
yellow dashed circle = ε edge-zone; × = empty-space; diamond = ambiguous.</p>
<div class="grid">
<div>{plot_html}</div>
<aside>
<h2>Assignment QC</h2>
<table>
<tr><td>n fixations</td><td>{len(fix)}</td></tr>
<tr><td>% empty</td><td>{qc.get('pct_empty_space', float('nan')):.1f}</td></tr>
<tr><td>% ambiguous</td><td>{qc.get('pct_ambiguous', float('nan')):.1f}</td></tr>
<tr><td>% edge-zone</td><td>{qc.get('pct_edge_zone', float('nan')):.1f}</td></tr>
<tr><td>mean confidence</td><td>{qc.get('mean_confidence', float('nan')):.3f}</td></tr>
</table>
<h2>Distance-to-edge (px)</h2>
<table><tr><th>bin</th><th>n</th></tr>{hist_rows}</table>
<h2>Assigned panel counts</h2>
<table><tr><th>panel</th><th>n</th></tr>{assign_rows}</table>
<h2>Export AOI → panel</h2>
<table><tr><th>panel</th><th>n samples</th></tr>{aoi_rows}</table>
<p class="note">AOI hits and assignment panels should tell a consistent story;
systematic colour/box mismatch ⇒ upstream fix before model work.</p>
</aside></div>
</body></html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uio.write_text(out_path, page)
    return {
        "participant_id": participant_id,
        "trial_id": trial_id,
        "star_condition": star_condition,
        "html": str(out_path),
        "flagged": flagged,
        "n_fixations": int(len(fix)),
        "qc": qc,
        "edge_hist": hist,
        "panel_vs_aoi": compare,
    }


def run_gate2_batch(
    repo_root: Optional[Path] = None,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    data_cfg = OmegaConf.load(repo_root / "configs" / "data.yaml")
    pre_cfg = OmegaConf.load(repo_root / "configs" / "preprocessing.yaml")
    gate_cfg = OmegaConf.to_container(pre_cfg.gate1, resolve=True)
    assert isinstance(gate_cfg, dict)
    g2 = OmegaConf.to_container(getattr(pre_cfg, "gate2", {}), resolve=True) or {}
    data_version = str(data_cfg.data_version)
    processed = repo_root / str(data_cfg.paths.processed_root) / data_version
    out_dir = repo_root / "reports" / "gaze_checks" / "gate2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        return _gate2_smoke(repo_root, out_dir, gate_cfg, float(pre_cfg.gaze_assignment.epsilon_doc_px or 8.0))

    epsilon = float(pre_cfg.gaze_assignment.epsilon_doc_px)
    aoi_map = dict(pre_cfg.gaze_aoi_panel_map)

    star_tbl = pd.read_parquet(processed / "registry" / "star_conditions.parquet")
    eligible = list(data_cfg.star_eligible_trials)
    base = select_stratified_sample(
        star_tbl,
        eligible=eligible,
        trials_per_participant=int(gate_cfg.get("trials_per_participant") or 3),
        seed=0,
    )
    qc = pd.read_parquet(processed / "fixations" / "episode_qc.parquet")
    flagged = flag_qc_episodes(
        qc,
        pct_empty=float(g2.get("flag_pct_empty_space", 40)),
        pct_ambiguous=float(g2.get("flag_pct_ambiguous", 40)),
        mean_confidence_below=float(g2.get("flag_mean_confidence_below", 0.2)),
    )
    # Merge sample
    seen = {(e["participant_id"], e["trial_id"]) for e in base}
    sample = list(base)
    for f in flagged:
        key = (f["participant_id"], f["trial_id"])
        if key not in seen:
            sample.append(
                {
                    "participant_id": f["participant_id"],
                    "trial_id": f["trial_id"],
                    "star_condition": f["star_condition"],
                }
            )
            seen.add(key)
    flagged_keys = {(f["participant_id"], f["trial_id"]) for f in flagged}

    qc_lookup = {
        (str(r["participant_id"]), str(r["trial_id"]), str(r["star_condition"])): r.to_dict()
        for _, r in qc.iterrows()
    }

    img_dir = repo_root / str(data_cfg.paths.document_images_dir)
    meta_dir = processed / "metadata"
    fix_root = processed / "fixations"
    gaze_dir = processed / "gaze_coords"

    results = []
    errors = []
    for ep in sample:
        pid, tid, sc = ep["participant_id"], ep["trial_id"], ep["star_condition"]
        stem = image_stem(tid, sc)
        try:
            img = img_dir / f"{stem}.png"
            segments = uio.read_json(meta_dir / f"{stem}__segments.json")
            panels = uio.read_json(meta_dir / f"{stem}__panels.json")
            fix_path = fix_root / pid / f"{tid}__{sc}.parquet"
            fix = pd.read_parquet(fix_path)
            num = pid[1:] if pid.upper().startswith("P") else pid
            gaze_path = gaze_dir / f"p{int(num):02d}.parquet"
            gaze_all = pd.read_parquet(gaze_path)
            gaze = gaze_all[gaze_all["trial_id"].astype(str) == tid]
            ep_qc = qc_lookup.get((pid, tid, sc), {})
            info = render_gate2_html(
                participant_id=pid,
                trial_id=tid,
                star_condition=sc,
                image_path=img,
                segments=segments,
                panels=panels,
                fix=fix,
                gaze=gaze,
                episode_qc=ep_qc,
                epsilon=epsilon,
                gate_cfg=gate_cfg,
                aoi_panel_map=aoi_map,
                out_path=out_dir / f"{pid}_{tid}_{sc}.html",
                flagged=(pid, tid) in flagged_keys,
            )
            results.append(info)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{pid}/{tid}/{sc}: {e}")

    star_on = {r["trial_id"] for r in sample if r["star_condition"] == "star_on"}
    summary = {
        "data_version": data_version,
        "out_dir": str(out_dir),
        "n_reports": len(results),
        "n_sample": len(sample),
        "n_flagged_qc_added": len(flagged_keys),
        "star_on_trials_covered": sorted(star_on),
        "eligible_star_coverage_ok": set(eligible).issubset(star_on),
        "errors": errors,
        "ok": len(errors) == 0 and len(results) == len(sample) and set(eligible).issubset(star_on),
        "sign_off": "PENDING owner review in reports/DECISIONS.md — no model code (M2+) until signed off.",
    }
    uio.write_json(
        out_dir / "gate2_manifest.json",
        {"sample": sample, "flagged": flagged, "results": results, "summary": summary},
    )
    rows = "".join(
        f"<tr><td>{r['participant_id']}</td><td>{r['trial_id']}</td><td>{r['star_condition']}</td>"
        f"<td>{'QC' if r['flagged'] else ''}</td>"
        f"<td><a href='{Path(r['html']).name}'>{Path(r['html']).name}</a></td>"
        f"<td>{r['qc'].get('pct_empty_space', float('nan')):.1f}</td>"
        f"<td>{r['qc'].get('pct_ambiguous', float('nan')):.1f}</td>"
        f"<td>{r['qc'].get('mean_confidence', float('nan')):.3f}</td></tr>"
        for r in results
    )
    index = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Gate 2 index</title>
<style>body{{font-family:system-ui,sans-serif;margin:1.5rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:.3rem .5rem}}th{{background:#f4f4f4}}</style></head>
<body>
<h1>Visual Gate 2 — stratified sample + P6 QC flags</h1>
<p><strong>HARD STOP:</strong> owner sign-off required in <code>reports/DECISIONS.md</code>
before any Stage 2 / model code (M2+). Do not self-certify.</p>
<pre>{json.dumps(summary, indent=2)}</pre>
<table>
<tr><th>Participant</th><th>Trial</th><th>Star</th><th>Flag</th><th>Report</th>
<th>% empty</th><th>% ambig</th><th>mean conf</th></tr>
{rows}
</table>
</body></html>
"""
    uio.write_text(out_dir / "index.html", index)
    return summary


def _gate2_smoke(
    repo_root: Path, out_dir: Path, gate_cfg: dict[str, Any], epsilon: float
) -> dict[str, Any]:
    from PIL import Image

    fx = repo_root / "fixtures" / "trials" / "fx01_T99"
    segments = uio.read_json(fx / "segments.json")
    fix_json = uio.read_json(fx / "fixations.json")
    # Flatten fixture fixations to a DataFrame-like for the plotter
    rows = []
    for i, f in enumerate(fix_json):
        g = next((s["geometry"] for s in segments if s["segment_id"] == f.get("segment_id")), None)
        rows.append(
            {
                "fixation_id": f["fixation_id"],
                "x_doc": float(g["x"]) if g else 40.0 + i * 5,
                "y_doc": float(g["y"]) if g else 40.0 + i * 5,
                "duration_ms": float(f["duration_ms"]),
                "segment_id": f.get("segment_id"),
                "segment_id_alt": f.get("segment_id_alt"),
                "empty_space_category": f.get("empty_space_category"),
                "panel_label": f["panel_label"],
                "assignment_confidence": float(f["assignment_confidence"]),
                "ambiguous": bool(f.get("ambiguous", False)),
                "edge_zone": False,
                "scroll_direction": f["scroll"]["direction"],
            }
        )
    # Force one empty + one ambiguous for panel coverage
    rows[0]["ambiguous"] = True
    rows[0]["segment_id_alt"] = segments[1]["segment_id"] if len(segments) > 1 else rows[0]["segment_id"]
    rows[-1]["segment_id"] = None
    rows[-1]["empty_space_category"] = "question_background"
    rows[-1]["panel_label"] = "question"
    rows[-1]["assignment_confidence"] = 0.0
    fix = pd.DataFrame(rows)
    panels = [
        {
            "aoi_id": "q",
            "aoi_type": "question",
            "panel_label": "question",
            "x_min": 0,
            "y_min": 0,
            "x_max": 500,
            "y_max": 400,
            "area": 200000,
        }
    ]
    img = out_dir / "_smoke_gate2.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), (245, 245, 240)).save(img)
    gaze = pd.DataFrame({"aoi_label": ["Question"] * 20 + ["Outside"] * 5})
    info = render_gate2_html(
        participant_id="P99",
        trial_id="T99",
        star_condition="not_eligible",
        image_path=img,
        segments=segments,
        panels=panels,
        fix=fix,
        gaze=gaze,
        episode_qc={
            "pct_empty_space": 10.0,
            "pct_ambiguous": 5.0,
            "pct_edge_zone": 8.0,
            "mean_confidence": 0.9,
        },
        epsilon=epsilon,
        gate_cfg=gate_cfg,
        aoi_panel_map={"Question": "question", "Outside": "outside_document"},
        out_path=out_dir / "smoke_P99_T99_not_eligible.html",
        flagged=True,
    )
    html = uio.read_text(Path(info["html"]))
    checks = {
        "has_plotly": "plotly" in html.lower(),
        "has_assigned": "Assigned" in html,
        "has_empty": "Empty-space" in html,
        "has_ambiguous": "Ambiguous" in html,
        "has_qc": "Assignment QC" in html,
        "has_edge_hist": "Distance-to-edge" in html,
        "has_panel_compare": "Assigned panel counts" in html,
        "has_play": "Play" in html,
    }
    summary = {"ok": all(checks.values()), "checks": checks, "html": info["html"], "smoke": True}
    uio.write_json(out_dir / "smoke_summary.json", summary)
    return summary
