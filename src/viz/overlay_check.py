"""P4 Visual Gate 1 — metadata–gaze alignment overlays (raw document px)."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from omegaconf import OmegaConf
from PIL import Image

from src.utils import io as uio

DEFAULT_AOI_COLOR = "#333333"


def image_stem(trial_id: str, star_condition: str) -> str:
    if star_condition == "star_on":
        return f"{trial_id}S"
    if star_condition == "star_off":
        return f"{trial_id}NS"
    return trial_id


def encode_image_jpeg(path: Path, *, quality: int = 70) -> tuple[str, int, int]:
    """Return (data URI, width, height) for embedding in self-contained HTML."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality), optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", w, h


def aggregate_fixations(gaze: pd.DataFrame) -> pd.DataFrame:
    """Median doc position per fixation event index."""
    fix = gaze[gaze["eye_movement_type"].astype(str) == "Fixation"].copy()
    if fix.empty:
        return pd.DataFrame(
            columns=[
                "eye_movement_type_index",
                "x",
                "y",
                "duration_ms",
                "t_start",
                "aoi_label",
            ]
        )
    idx_col = "eye_movement_type_index"
    rows = []
    for eid, g in fix.groupby(idx_col, sort=True):
        labels = g["aoi_label"].dropna().astype(str)
        mode = labels.mode().iloc[0] if len(labels) else "Outside"
        rows.append(
            {
                "eye_movement_type_index": eid,
                "x": float(g["gaze_point_x_doc"].median()),
                "y": float(g["gaze_point_y_doc"].median()),
                "duration_ms": float(g["gaze_event_duration"].median()),
                "t_start": float(g["recording_timestamp"].min()),
                "aoi_label": mode,
            }
        )
    return pd.DataFrame(rows)


def downsample_scatter(gaze: pd.DataFrame, max_points: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(gaze) <= max_points:
        return gaze
    idx = rng.choice(len(gaze), size=max_points, replace=False)
    return gaze.iloc[np.sort(idx)]


def point_in_any_box(
    x: np.ndarray,
    y: np.ndarray,
    boxes: list[dict[str, float]],
    *,
    strict: bool = False,
) -> np.ndarray:
    hit = np.zeros(len(x), dtype=bool)
    for b in boxes:
        if strict:
            inside = (x > b["x_min"]) & (x < b["x_max"]) & (y > b["y_min"]) & (y < b["y_max"])
        else:
            inside = (x >= b["x_min"]) & (x <= b["x_max"]) & (y >= b["y_min"]) & (y <= b["y_max"])
        hit |= inside
    return hit


def compute_alignment_stats(
    gaze: pd.DataFrame,
    segments: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    *,
    w_doc: int,
    h_doc: int,
) -> dict[str, Any]:
    x = gaze["gaze_point_x_doc"].to_numpy(dtype=float)
    y = gaze["gaze_point_y_doc"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[valid], y[valid]
    n = len(xv)
    seg_boxes = [
        {
            "x_min": float(s["geometry"]["x_min"]),
            "y_min": float(s["geometry"]["y_min"]),
            "x_max": float(s["geometry"]["x_max"]),
            "y_max": float(s["geometry"]["y_max"]),
        }
        for s in segments
        if s.get("geometry")
    ]
    panel_boxes = [
        {
            "x_min": float(p["x_min"]),
            "y_min": float(p["y_min"]),
            "x_max": float(p["x_max"]),
            "y_max": float(p["y_max"]),
        }
        for p in panels
    ]
    in_seg = point_in_any_box(xv, yv, seg_boxes) if n and seg_boxes else np.zeros(n, dtype=bool)
    in_panel = point_in_any_box(xv, yv, panel_boxes) if n and panel_boxes else np.zeros(n, dtype=bool)
    in_doc = (xv >= 0) & (xv <= w_doc) & (yv >= 0) & (yv <= h_doc) if n else np.zeros(0, dtype=bool)
    label_counts = gaze.loc[valid, "aoi_label"].astype(str).value_counts().to_dict() if n else {}
    return {
        "n_samples": int(len(gaze)),
        "n_valid_xy": int(n),
        "pct_inside_segment": float(100.0 * in_seg.mean()) if n else 0.0,
        "pct_inside_panel": float(100.0 * in_panel.mean()) if n else 0.0,
        "pct_outside_document": float(100.0 * (~in_doc).mean()) if n else 0.0,
        "aoi_label_counts": {str(k): int(v) for k, v in label_counts.items()},
    }


def _rect_shape(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    color: str,
    width: float,
    dash: Optional[str] = None,
) -> dict[str, Any]:
    line: dict[str, Any] = {"color": color, "width": width}
    if dash:
        line["dash"] = dash
    return {
        "type": "rect",
        "xref": "x",
        "yref": "y",
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "line": line,
        "fillcolor": "rgba(0,0,0,0)",
    }


def build_episode_figure(
    *,
    image_uri: str,
    w_doc: int,
    h_doc: int,
    segments: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    gaze: pd.DataFrame,
    fixations: pd.DataFrame,
    gate_cfg: dict[str, Any],
    title: str,
    seed: int = 0,
) -> go.Figure:
    panel_colors = dict(gate_cfg.get("panel_colors") or {})
    aoi_colors = dict(gate_cfg.get("aoi_label_colors") or {})
    ui_types = set(gate_cfg.get("ui_panel_aoi_types") or [])
    max_scatter = int(gate_cfg.get("max_scatter_points") or 8000)
    rng = np.random.default_rng(seed)

    shapes: list[dict[str, Any]] = []
    for s in segments:
        g = s.get("geometry") or {}
        color = panel_colors.get(s.get("panel_label"), "#000000")
        shapes.append(
            _rect_shape(
                float(g["x_min"]),
                float(g["y_min"]),
                float(g["x_max"]),
                float(g["y_max"]),
                color=color,
                width=1.5,
            )
        )
    for p in panels:
        aoi_type = str(p.get("aoi_type") or "")
        panel_label = str(p.get("panel_label") or "")
        if aoi_type == "star_chart":
            color = panel_colors.get("star_chart", "#d62728")
            width, dash = 3.0, None
        elif aoi_type in ui_types:
            color = aoi_colors.get(
                {
                    "answer_scroll_bar": "Answer_Scroll_Bar",
                    "commentary_scroll_bar": "Commentary_Scroll_Bar",
                    "general_ui": "General_UI",
                }.get(aoi_type, ""),
                "#7f7f7f",
            )
            width, dash = 2.0, "dot"
        else:
            color = panel_colors.get(panel_label, "#444444")
            width, dash = 2.0, "dash"
        shapes.append(
            _rect_shape(
                float(p["x_min"]),
                float(p["y_min"]),
                float(p["x_max"]),
                float(p["y_max"]),
                color=color,
                width=width,
                dash=dash,
            )
        )

    samples = downsample_scatter(gaze.dropna(subset=["gaze_point_x_doc", "gaze_point_y_doc"]), max_scatter, rng)
    sample_colors = [aoi_colors.get(str(a), DEFAULT_AOI_COLOR) for a in samples["aoi_label"].astype(str)]
    fix_colors = [aoi_colors.get(str(a), DEFAULT_AOI_COLOR) for a in fixations["aoi_label"].astype(str)] if len(fixations) else []

    # Star-injection highlight layer (star_on samples with Star_Chart label)
    star_mask = samples["aoi_label"].astype(str) == "Star_Chart"
    star_pts = samples.loc[star_mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=samples["gaze_point_x_doc"],
            y=samples["gaze_point_y_doc"],
            mode="markers",
            name="Samples",
            marker=dict(size=4, color=sample_colors, opacity=0.35),
            customdata=samples["aoi_label"].astype(str),
            hovertemplate="(%{x:.0f}, %{y:.0f})<br>%{customdata}<extra></extra>",
            visible=True,
        )
    )
    if len(fixations):
        sizes = np.clip(np.sqrt(fixations["duration_ms"].to_numpy(dtype=float) / 10.0), 4, 28)
        fig.add_trace(
            go.Scatter(
                x=fixations["x"],
                y=fixations["y"],
                mode="markers",
                name="Fixations",
                marker=dict(size=sizes, color=fix_colors, opacity=0.75, line=dict(width=0.5, color="#222")),
                customdata=np.stack(
                    [
                        fixations["aoi_label"].astype(str),
                        fixations["duration_ms"].to_numpy(dtype=float),
                    ],
                    axis=1,
                ),
                hovertemplate="(%{x:.0f}, %{y:.0f})<br>%{customdata[0]}<br>%{customdata[1]:.0f} ms<extra></extra>",
                visible=False,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(x=[], y=[], mode="markers", name="Fixations", visible=False)
        )

    fig.add_trace(
        go.Scatter(
            x=star_pts["gaze_point_x_doc"] if len(star_pts) else [],
            y=star_pts["gaze_point_y_doc"] if len(star_pts) else [],
            mode="markers",
            name="Star_Chart hits",
            marker=dict(size=7, color=aoi_colors.get("Star_Chart", "#d62728"), symbol="x", opacity=0.9),
            visible=True,
            hoverinfo="skip",
        )
    )

    # Time-slider frames over fixations (cumulative reveal)
    n_steps = 12
    frames = []
    if len(fixations):
        tvals = fixations["t_start"].to_numpy(dtype=float)
        t_min, t_max = float(tvals.min()), float(tvals.max())
        edges = np.linspace(t_min, t_max, n_steps + 1)
        for i in range(n_steps):
            thr = edges[i + 1]
            m = tvals <= thr
            sub = fixations.loc[m]
            cols = [aoi_colors.get(str(a), DEFAULT_AOI_COLOR) for a in sub["aoi_label"].astype(str)]
            sz = np.clip(np.sqrt(sub["duration_ms"].to_numpy(dtype=float) / 10.0), 4, 28)
            frames.append(
                go.Frame(
                    name=str(i),
                    data=[
                        go.Scatter(
                            x=sub["x"],
                            y=sub["y"],
                            mode="markers",
                            marker=dict(size=sz, color=cols, opacity=0.75, line=dict(width=0.5, color="#222")),
                        )
                    ],
                    traces=[1],
                )
            )
        fig.frames = frames

    fig.update_layout(
        title=title,
        width=min(1200, w_doc + 80),
        height=min(900, int(h_doc * (1200 / max(w_doc, 1))) + 120),
        xaxis=dict(range=[0, w_doc], constrain="domain", scaleanchor="y", scaleratio=1, title="x_doc (px)"),
        yaxis=dict(range=[h_doc, 0], constrain="domain", title="y_doc (px)"),  # image coords: y down
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
        margin=dict(l=40, r=20, t=80, b=40),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.14,
                showactive=True,
                buttons=[
                    dict(
                        label="Samples",
                        method="update",
                        args=[{"visible": [True, False, True]}],
                    ),
                    dict(
                        label="Fixations",
                        method="update",
                        args=[{"visible": [False, True, False]}],
                    ),
                ],
            ),
            dict(
                type="buttons",
                direction="left",
                x=0.35,
                y=1.14,
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 400, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"mode": "immediate", "frame": {"duration": 0}}],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=n_steps - 1 if len(fixations) else 0,
                pad={"t": 30},
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                        label=f"{i + 1}/{n_steps}",
                    )
                    for i in range(n_steps)
                ]
                if len(fixations)
                else [],
                currentvalue={"prefix": "Time bin: "},
            )
        ]
        if len(fixations)
        else [],
    )
    return fig


def render_episode_html(
    *,
    participant_id: str,
    trial_id: str,
    star_condition: str,
    image_path: Path,
    segments: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    gaze: pd.DataFrame,
    injection_qc: Optional[dict[str, Any]],
    gate_cfg: dict[str, Any],
    out_path: Path,
    seed: int = 0,
) -> dict[str, Any]:
    quality = int(gate_cfg.get("jpeg_quality") or 70)
    uri, w_doc, h_doc = encode_image_jpeg(image_path, quality=quality)
    fixations = aggregate_fixations(gaze)
    stats = compute_alignment_stats(gaze, segments, panels, w_doc=w_doc, h_doc=h_doc)
    title = f"{participant_id} · {trial_id} · {star_condition}"
    fig = build_episode_figure(
        image_uri=uri,
        w_doc=w_doc,
        h_doc=h_doc,
        segments=segments,
        panels=panels,
        gaze=gaze,
        fixations=fixations,
        gate_cfg=gate_cfg,
        title=title,
        seed=seed,
    )

    # Side panels as HTML tables
    aoi_rows = "".join(
        f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>"
        for k, v in sorted(stats["aoi_label_counts"].items(), key=lambda kv: -kv[1])
    )
    inj_html = "<p><em>No injection QC row.</em></p>"
    if injection_qc:
        note = (
            "<p class='note'>Scrollbar hit rates are indicative "
            "(thin regions vs gaze precision).</p>"
        )
        inj_html = note + "<table><tr><th>Metric</th><th>Value</th></tr>"
        for k in (
            "n_star_hits",
            "n_star_relabel",
            "star_hit_proportion",
            "n_ui_label_updates",
            "n_hit_aoi__answer_scroll_bar",
            "n_hit_aoi__commentary_scroll_bar",
            "n_hit_aoi__general_ui",
        ):
            if k in injection_qc:
                inj_html += f"<tr><td>{k}</td><td>{injection_qc[k]}</td></tr>"
        inj_html += "</table>"

    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Gate 1 — {participant_id} {trial_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem 1.5rem; color: #222; }}
.grid {{ display: grid; grid-template-columns: 1fr 320px; gap: 1rem; }}
aside {{ font-size: 0.9rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.5rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.25rem 0.4rem; }}
th {{ background: #f4f4f4; text-align: left; }}
.note {{ color: #666; font-size: 0.85rem; }}
h1 {{ font-size: 1.2rem; }}
h2 {{ font-size: 1rem; margin-top: 1rem; }}
.legend-swatch {{ display:inline-block; width:12px; height:12px; margin-right:4px; vertical-align:middle; }}
</style>
</head>
<body>
<h1>Visual Gate 1 — metadata–gaze alignment</h1>
<p><strong>{participant_id}</strong> · <strong>{trial_id}</strong> · <strong>{star_condition}</strong>
 · image <code>{image_path.name}</code></p>
<p class="note">Raw document pixels. Solid boxes = segments (panel colour).
Dashed = content panels; dotted = UI regions; thick red = star chart.
Toggle Samples / Fixations; Play animates fixation time bins.</p>
<div class="grid">
  <div>{plot_html}</div>
  <aside>
    <h2>Alignment summary</h2>
    <table>
      <tr><td>Samples</td><td>{stats['n_samples']}</td></tr>
      <tr><td>Valid XY</td><td>{stats['n_valid_xy']}</td></tr>
      <tr><td>% in any segment</td><td>{stats['pct_inside_segment']:.1f}</td></tr>
      <tr><td>% in any panel</td><td>{stats['pct_inside_panel']:.1f}</td></tr>
      <tr><td>% outside document</td><td>{stats['pct_outside_document']:.1f}</td></tr>
      <tr><td>Fixation events</td><td>{len(fixations)}</td></tr>
    </table>
    <h2>AOI_label counts</h2>
    <table><tr><th>Label</th><th>n</th></tr>{aoi_rows}</table>
    <h2>P3 injection QC</h2>
    {inj_html}
  </aside>
</div>
</body>
</html>
"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    uio.write_text(out_path, page)
    return {
        "participant_id": participant_id,
        "trial_id": trial_id,
        "star_condition": star_condition,
        "html": str(out_path),
        "stats": stats,
        "n_fixations": int(len(fixations)),
    }


def select_stratified_sample(
    star_tbl: pd.DataFrame,
    *,
    eligible: list[str],
    trials_per_participant: int = 3,
    flagged: Optional[list[tuple[str, str]]] = None,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Every participant × ≥N trials; star_on covering all eligible; plus flagged."""
    rng = np.random.default_rng(seed)
    selected: set[tuple[str, str]] = set()
    meta: dict[tuple[str, str], str] = {}

    def add(pid: str, tid: str, sc: str) -> None:
        selected.add((pid, tid))
        meta[(pid, tid)] = sc

    # Cover all eligible star_on trials
    for tid in eligible:
        cand = star_tbl[(star_tbl["trial_id"] == tid) & (star_tbl["star_condition"] == "star_on")]
        if cand.empty:
            continue
        row = cand.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        add(str(row["participant_id"]), str(row["trial_id"]), "star_on")

    # Flagged episodes from audit (if any)
    for pid, tid in flagged or []:
        hit = star_tbl[(star_tbl["participant_id"] == pid) & (star_tbl["trial_id"] == tid)]
        if len(hit):
            add(pid, tid, str(hit.iloc[0]["star_condition"]))

    # Ensure ≥N trials per participant
    for pid, g in star_tbl.groupby("participant_id"):
        have = {t for (p, t) in selected if p == pid}
        need = trials_per_participant - len(have)
        if need <= 0:
            continue
        remaining = g[~g["trial_id"].isin(have)].copy()
        # Prefer mix: keep star_on if available, else sample
        take = min(need, len(remaining))
        if take <= 0:
            continue
        picked = remaining.sample(take, random_state=int(rng.integers(0, 1_000_000)))
        for row in picked.itertuples(index=False):
            add(str(row.participant_id), str(row.trial_id), str(row.star_condition))

    out = [
        {"participant_id": p, "trial_id": t, "star_condition": meta[(p, t)]}
        for p, t in sorted(selected)
    ]
    return out


def load_flagged_from_audit(issues_csv: Path) -> list[tuple[str, str]]:
    """Return (participant_id, trial_id) pairs if audit issues encode them; else []."""
    if not issues_csv.is_file():
        return []
    df = pd.read_csv(issues_csv, encoding="utf-8")
    if df.empty:
        return []
    # Audit is metadata-file level; no participant. Return empty — caller may
    # add all variants as extra review via stem list if needed.
    return []


def run_gate1_batch(
    repo_root: Optional[Path] = None,
    *,
    sample: Optional[list[dict[str, str]]] = None,
    smoke: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    data_cfg = OmegaConf.load(repo_root / "configs" / "data.yaml")
    pre_cfg = OmegaConf.load(repo_root / "configs" / "preprocessing.yaml")
    gate_cfg = OmegaConf.to_container(pre_cfg.gate1, resolve=True)
    assert isinstance(gate_cfg, dict)
    data_version = str(data_cfg.data_version)
    processed = repo_root / str(data_cfg.paths.processed_root) / data_version
    out_dir = repo_root / "reports" / "gaze_checks" / "gate1"
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        return _run_smoke(repo_root, out_dir, gate_cfg)

    star_tbl = pd.read_parquet(processed / "registry" / "star_conditions.parquet")
    eligible = list(data_cfg.star_eligible_trials)
    issues = repo_root / "reports" / "metadata_audit" / "metadata_audit_issues.csv"
    flagged = load_flagged_from_audit(issues)
    if sample is None:
        sample = select_stratified_sample(
            star_tbl,
            eligible=eligible,
            trials_per_participant=int(gate_cfg.get("trials_per_participant") or 3),
            flagged=flagged,
            seed=0,
        )

    inj_qc = pd.read_parquet(processed / "gaze_canonical" / "injection_qc.parquet")
    inj_lookup = {
        (str(r["participant_id"]), str(r["trial_id"])): r.to_dict()
        for _, r in inj_qc.iterrows()
    }

    img_dir = repo_root / str(data_cfg.paths.document_images_dir)
    meta_dir = processed / "metadata"
    gaze_dir = processed / "gaze_canonical"

    results = []
    errors = []
    for ep in sample:
        pid, tid, sc = ep["participant_id"], ep["trial_id"], ep["star_condition"]
        stem = image_stem(tid, sc)
        img = img_dir / f"{stem}.png"
        seg_path = meta_dir / f"{stem}__segments.json"
        pan_path = meta_dir / f"{stem}__panels.json"
        gaze_path = gaze_dir / f"{pid.lower()}.parquet"
        if not gaze_path.is_file():
            # p01 vs P01
            num = pid[1:] if pid.upper().startswith("P") else pid
            gaze_path = gaze_dir / f"p{int(num):02d}.parquet"
        try:
            if not img.is_file():
                raise FileNotFoundError(f"missing image {img}")
            segments = uio.read_json(seg_path)
            panels = uio.read_json(pan_path)
            gaze_all = pd.read_parquet(gaze_path)
            gaze = gaze_all[gaze_all["trial_id"].astype(str) == tid].copy()
            if gaze.empty:
                raise ValueError(f"no gaze rows for {pid}/{tid}")
            qc = inj_lookup.get((pid, tid))
            out_html = out_dir / f"{pid}_{tid}_{sc}.html"
            info = render_episode_html(
                participant_id=pid,
                trial_id=tid,
                star_condition=sc,
                image_path=img,
                segments=segments,
                panels=panels,
                gaze=gaze,
                injection_qc=qc,
                gate_cfg=gate_cfg,
                out_path=out_html,
                seed=0,
            )
            results.append(info)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{pid}/{tid}/{sc}: {e}")

    # Coverage checks
    star_on_trials = {r["trial_id"] for r in sample if r["star_condition"] == "star_on"}
    per_pid = pd.Series([r["participant_id"] for r in sample]).value_counts()
    summary = {
        "data_version": data_version,
        "out_dir": str(out_dir),
        "n_reports": len(results),
        "n_sample": len(sample),
        "n_participants": int(per_pid.shape[0]),
        "min_trials_per_participant": int(per_pid.min()) if len(per_pid) else 0,
        "star_on_trials_covered": sorted(star_on_trials),
        "eligible_star_coverage_ok": set(eligible).issubset(star_on_trials),
        "errors": errors,
        "ok": len(errors) == 0
        and len(results) == len(sample)
        and set(eligible).issubset(star_on_trials)
        and (int(per_pid.min()) >= int(gate_cfg.get("trials_per_participant") or 3) if len(per_pid) else False),
        "sign_off": "PENDING owner review in reports/DECISIONS.md — do not start P5 until signed off.",
    }
    uio.write_json(out_dir / "gate1_manifest.json", {"sample": sample, "results": results, "summary": summary})
    # Index page
    rows = "".join(
        f"<tr><td>{r['participant_id']}</td><td>{r['trial_id']}</td>"
        f"<td>{r['star_condition']}</td>"
        f"<td><a href='{Path(r['html']).name}'>{Path(r['html']).name}</a></td>"
        f"<td>{r['stats']['pct_inside_segment']:.1f}</td>"
        f"<td>{r['stats']['pct_inside_panel']:.1f}</td></tr>"
        for r in results
    )
    index = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Gate 1 index</title>
<style>body{{font-family:system-ui,sans-serif;margin:1.5rem}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:.3rem .5rem}} th{{background:#f4f4f4}}</style></head>
<body>
<h1>Visual Gate 1 — stratified sample</h1>
<p><strong>STOP:</strong> owner sign-off required in <code>reports/DECISIONS.md</code> before P5.
Do not self-certify.</p>
<pre>{json.dumps(summary, indent=2)}</pre>
<table>
<tr><th>Participant</th><th>Trial</th><th>Star</th><th>Report</th><th>% seg</th><th>% panel</th></tr>
{rows}
</table>
</body></html>
"""
    uio.write_text(out_dir / "index.html", index)
    return summary


def _run_smoke(repo_root: Path, out_dir: Path, gate_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build one HTML from fixtures + synthetic image/gaze (no _data required)."""
    fx = repo_root / "fixtures" / "trials" / "fx01_T99"
    segments = uio.read_json(fx / "segments.json")
    # Synthetic panels from segment extents
    panels = [
        {
            "aoi_id": "p_q",
            "aoi_type": "question",
            "panel_label": "question",
            "x_min": 0,
            "y_min": 0,
            "x_max": 500,
            "y_max": 200,
            "area": 100000,
        },
        {
            "aoi_id": "p_ui",
            "aoi_type": "general_ui",
            "panel_label": "ui",
            "x_min": 0,
            "y_min": 0,
            "x_max": 800,
            "y_max": 40,
            "area": 32000,
        },
        {
            "aoi_id": "p_star",
            "aoi_type": "star_chart",
            "panel_label": "star_chart",
            "x_min": 520,
            "y_min": 220,
            "x_max": 760,
            "y_max": 420,
            "area": 48000,
        },
    ]
    # Blank document image
    img_path = out_dir / "_smoke_doc.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (800, 600), color=(245, 245, 240)).save(img_path)

    rows = []
    for i, seg in enumerate(segments[:8]):
        g = seg["geometry"]
        cx, cy = g["x"], g["y"]
        for j in range(5):
            rows.append(
                {
                    "gaze_point_x_doc": cx + (j - 2) * 2,
                    "gaze_point_y_doc": cy + (j - 2) * 2,
                    "aoi_label": "Question" if seg["panel_label"] == "question" else "Mark_Scheme",
                    "eye_movement_type": "Fixation",
                    "eye_movement_type_index": i,
                    "gaze_event_duration": 100 + 10 * j,
                    "recording_timestamp": i * 1000 + j * 50,
                }
            )
    # Outside + star hits
    for j in range(10):
        rows.append(
            {
                "gaze_point_x_doc": 600 + j,
                "gaze_point_y_doc": 300 + j,
                "aoi_label": "Star_Chart",
                "eye_movement_type": "Fixation",
                "eye_movement_type_index": 99,
                "gaze_event_duration": 120,
                "recording_timestamp": 50_000 + j,
            }
        )
    gaze = pd.DataFrame(rows)
    qc = {
        "n_star_hits": 10,
        "n_star_relabel": 10,
        "star_hit_proportion": 0.1,
        "n_ui_label_updates": 0,
        "n_hit_aoi__answer_scroll_bar": 0,
        "n_hit_aoi__commentary_scroll_bar": 0,
        "n_hit_aoi__general_ui": 2,
    }
    info = render_episode_html(
        participant_id="P99",
        trial_id="T99",
        star_condition="star_on",
        image_path=img_path,
        segments=segments,
        panels=panels,
        gaze=gaze,
        injection_qc=qc,
        gate_cfg=gate_cfg,
        out_path=out_dir / "smoke_P99_T99_star_on.html",
        seed=0,
    )
    html = uio.read_text(Path(info["html"]))
    checks = {
        "has_plotly": "plotly" in html.lower(),
        "has_samples_button": "Samples" in html,
        "has_fixations_button": "Fixations" in html,
        "has_aoi_counts": "AOI_label counts" in html,
        "has_injection_qc": "n_star_hits" in html,
        "has_alignment_summary": "Alignment summary" in html,
        "has_star_panel": "star" in html.lower(),
    }
    summary = {
        "ok": all(checks.values()) and info["n_fixations"] > 0,
        "checks": checks,
        "html": info["html"],
        "stats": info["stats"],
        "smoke": True,
    }
    uio.write_json(out_dir / "smoke_summary.json", summary)
    return summary
