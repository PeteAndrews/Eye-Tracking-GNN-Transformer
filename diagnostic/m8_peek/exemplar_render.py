"""Step 3 — one-off document-space exemplar renderer (not gaze_overlay_check)."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from omegaconf import OmegaConf
from PIL import Image

from src.utils import io as uio


def _image_stem(trial_id: str, star_condition: str) -> str:
    if star_condition == "star_on":
        return f"{trial_id}S"
    if star_condition == "star_off":
        return f"{trial_id}NS"
    return trial_id


def _encode_jpeg(path: Path, *, quality: int = 70) -> tuple[str, int, int]:
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=int(quality), optimize=True)
    b64 = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}", w, h


def _panel_bucket(label: str) -> str:
    s = str(label).lower()
    if "outside" in s:
        return "outside"
    if s in ("question", "response", "mark_scheme", "commentary", "star_chart", "ui"):
        return s
    if "mark" in s:
        return "mark_scheme"
    if "star" in s:
        return "star_chart"
    return s


def find_exemplars(
    assign: pd.DataFrame,
    prototypes: list[int],
    *,
    min_len: int = 5,
    max_len: int = 80,
    per_prototype: int = 1,
    context_pad: int = 20,
) -> list[dict[str, Any]]:
    """Pick a seed hard-label run, then expand with context for multi-AOI replay.

    Ranking prefers: multi-panel seed windows, high hard fraction, high posterior.
    Stored t_start/t_end include context padding so renders are not single-AOI blobs.
    """
    exemplars: list[dict[str, Any]] = []
    used_pids: dict[int, set[str]] = {p: set() for p in prototypes}

    for proto in prototypes:
        post_col = f"post_{proto}"
        if post_col not in assign.columns:
            continue
        candidates: list[dict[str, Any]] = []
        for ep_i, g in assign.groupby("episode_idx", sort=True):
            g = g.sort_values("t").reset_index(drop=True)
            post = g[post_col].to_numpy(dtype=np.float64)
            hard = g["prototype"].to_numpy(dtype=int)
            panels = g["panel_label"].map(_panel_bucket).to_numpy()
            T = len(g)
            t0 = 0
            while t0 < T:
                if hard[t0] != proto:
                    t0 += 1
                    continue
                t1 = t0
                while t1 < T and hard[t1] == proto:
                    t1 += 1
                length = t1 - t0
                if min_len <= length <= max_len:
                    seed_panels = set(panels[t0:t1].tolist())
                    # Expand with context for display / multi-AOI visibility
                    a = max(0, t0 - context_pad)
                    b = min(T, t1 + context_pad)
                    ctx_panels = set(panels[a:b].tolist())
                    candidates.append(
                        {
                            "prototype": proto,
                            "episode_idx": int(ep_i),
                            "participant_id": str(g["participant_id"].iloc[0]),
                            "trial_id": str(g["trial_id"].iloc[0]),
                            "star_condition": str(g["star_condition"].iloc[0]),
                            # display window = seed + context
                            "t_start": int(g["t"].iloc[a]),
                            "t_end": int(g["t"].iloc[b - 1]),
                            "seed_t_start": int(g["t"].iloc[t0]),
                            "seed_t_end": int(g["t"].iloc[t1 - 1]),
                            "mean_posterior": float(post[t0:t1].mean()),
                            "hard_frac": 1.0,
                            "n_seed": length,
                            "n_display": int(b - a),
                            "n_panels_seed": len(seed_panels),
                            "n_panels_display": len(ctx_panels),
                            "panels_seed": sorted(seed_panels),
                            "panels_display": sorted(ctx_panels),
                        }
                    )
                t0 = max(t1, t0 + 1)
        # Prefer multi-panel display windows, then longer seeds, then posterior
        candidates.sort(
            key=lambda d: (
                -d["n_panels_display"],
                -d["n_panels_seed"],
                -d["n_seed"],
                -d["mean_posterior"],
            )
        )
        picked = 0
        for c in candidates:
            pid = c["participant_id"]
            if pid in used_pids[proto]:
                continue
            used_pids[proto].add(pid)
            exemplars.append(c)
            picked += 1
            if picked >= per_prototype:
                break
        if picked == 0 and candidates:
            exemplars.append(candidates[0])
    return exemplars


def render_exemplar_html(
    repo: Path,
    assign: pd.DataFrame,
    exemplar: dict[str, Any],
    *,
    out_path: Path,
) -> Path:
    """Full-episode posterior colouring + hard-membership + seed/context windows."""
    repo = Path(repo)
    data_cfg = OmegaConf.load(repo / "configs" / "data.yaml")
    img_root = repo / str(data_cfg.paths.document_images_dir)
    stem = _image_stem(exemplar["trial_id"], exemplar["star_condition"])
    img_path = None
    for ext in (".png", ".jpg", ".jpeg"):
        p = img_root / f"{stem}{ext}"
        if p.is_file():
            img_path = p
            break
    if img_path is None:
        raise FileNotFoundError(f"No document image for stem={stem} under {img_root}")

    uri, w, h = _encode_jpeg(img_path)
    proto = int(exemplar["prototype"])
    post_col = f"post_{proto}"
    ep = assign[assign["episode_idx"] == exemplar["episode_idx"]].sort_values("t")
    x = ep["x_doc"].to_numpy(dtype=float)
    y = ep["y_doc"].to_numpy(dtype=float)
    post = ep[post_col].to_numpy(dtype=float) if post_col in ep.columns else np.zeros(len(ep))
    hard = ep["prototype"].to_numpy(dtype=int)
    t = ep["t"].to_numpy(dtype=int)
    panels = ep["panel_label"].astype(str).to_numpy()

    seed_lo = int(exemplar.get("seed_t_start", exemplar["t_start"]))
    seed_hi = int(exemplar.get("seed_t_end", exemplar["t_end"]))
    ctx_lo = int(exemplar["t_start"])
    ctx_hi = int(exemplar["t_end"])
    in_seed = (t >= seed_lo) & (t <= seed_hi)
    in_ctx = (t >= ctx_lo) & (t <= ctx_hi)
    is_hard = hard == proto

    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=uri,
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex=w,
            sizey=h,
            sizing="stretch",
            opacity=1.0,
            layer="below",
        )
    )

    # Layer 1: all fixations coloured by this prototype's posterior (shows mass across AOIs)
    hover = [
        f"t={ti} panel={p} post={pr:.2f} hard={h == proto}"
        for ti, p, pr, h in zip(t, panels, post, hard)
    ]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=7,
                color=post,
                colorscale="Viridis",
                cmin=0,
                cmax=1,
                colorbar=dict(title=f"P(proto {proto})"),
                opacity=0.55,
                line=dict(width=0),
            ),
            name="all fixations (posterior)",
            text=hover,
            hoverinfo="text",
        )
    )

    # Layer 2: hard membership anywhere in the episode (white ring) — can span AOIs
    if is_hard.any():
        fig.add_trace(
            go.Scatter(
                x=x[is_hard],
                y=y[is_hard],
                mode="markers",
                marker=dict(
                    size=11,
                    color=post[is_hard],
                    colorscale="Viridis",
                    cmin=0,
                    cmax=1,
                    showscale=False,
                    line=dict(width=1.5, color="white"),
                    opacity=0.95,
                ),
                name=f"hard-assigned proto {proto}",
                text=[hover[i] for i in np.where(is_hard)[0]],
                hoverinfo="text",
            )
        )

    # Layer 3: seed window outline (magenta) — the contiguous hard run used to pick the clip
    if in_seed.any():
        fig.add_trace(
            go.Scatter(
                x=x[in_seed],
                y=y[in_seed],
                mode="markers+lines",
                marker=dict(
                    size=14,
                    color="magenta",
                    line=dict(width=1, color="white"),
                    opacity=0.9,
                ),
                line=dict(color="rgba(255,0,255,0.35)", width=1),
                name=f"seed hard-run t={seed_lo}–{seed_hi}",
                hoverinfo="skip",
            )
        )

    # Layer 4: context window (optional faint path)
    ctx_only = in_ctx & ~in_seed
    if ctx_only.any():
        fig.add_trace(
            go.Scatter(
                x=x[ctx_only],
                y=y[ctx_only],
                mode="markers",
                marker=dict(size=9, color="rgba(255,180,0,0.7)", line=dict(width=0)),
                name=f"context pad t={ctx_lo}–{ctx_hi}",
                hoverinfo="skip",
            )
        )

    panels_note = ",".join(exemplar.get("panels_display") or [])
    fig.update_layout(
        title=(
            f"DIAGNOSTIC peek — proto {proto} · {exemplar['participant_id']} · "
            f"{exemplar['trial_id']} ({exemplar['star_condition']}) · "
            f"seed post={exemplar['mean_posterior']:.2f} · panels[{panels_note}]"
        ),
        width=min(1100, w + 80),
        height=min(1400, int(h * (1100 / max(w, 1))) + 100),
        xaxis=dict(visible=False, range=[0, w]),
        yaxis=dict(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=70, b=10),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font=dict(color="#eee"),
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)
    return out_path


def run_exemplars(
    repo: Path,
    out_dir: Path,
    *,
    prototypes: Optional[list[int]] = None,
    per_prototype: int = 1,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    ex_dir = out_dir / "exemplars"
    ex_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous HTMLs so stale single-AOI clips don't linger
    for old in ex_dir.glob("proto*.html"):
        old.unlink()

    assign = pd.read_parquet(out_dir / "assignments.parquet")
    fp = uio.read_json(out_dir / "fingerprints.json")
    k = int(fp.get("chosen_k") or assign["prototype"].max() + 1)
    protos = prototypes if prototypes is not None else list(
        fp.get("exemplar_prototypes_all") or range(k)
    )
    protos = [int(p) for p in protos]

    found = find_exemplars(assign, protos, per_prototype=per_prototype)
    written = []
    for ex in found:
        name = (
            f"proto{ex['prototype']}_{ex['participant_id']}_"
            f"{ex['trial_id']}_t{ex['t_start']}-{ex['t_end']}.html"
        )
        path = ex_dir / name
        try:
            render_exemplar_html(repo, assign, ex, out_path=path)
            written.append({"exemplar": ex, "path": str(path)})
        except Exception as exc:  # noqa: BLE001
            written.append({"exemplar": ex, "error": str(exc)})

    summary = {
        "diagnostic": True,
        "label": "m8_diagnostic_peek_step3",
        "prototypes": protos,
        "exemplars": written,
        "note": (
            "Full-episode posterior colouring; magenta = seed hard-run; "
            "orange = context pad. Hard membership can span AOIs even when the "
            "seed run is panel-local (panel-dominated prototypes)."
        ),
    }
    uio.write_json(ex_dir / "exemplars.json", summary)
    index_lines = [
        "# Diagnostic exemplars (no names)",
        "",
        "Colour = P(prototype) over the **whole episode**. White ring = hard-assigned.",
        "Magenta = seed contiguous hard-run; orange = ±context pad (often multi-AOI).",
        "",
    ]
    for w in written:
        if "path" in w:
            p = Path(w["path"]).name
            ex = w["exemplar"]
            panels = ",".join(ex.get("panels_display") or [])
            index_lines.append(
                f"- [proto {ex['prototype']} · {ex['participant_id']} · "
                f"{ex['trial_id']} t={ex['t_start']}–{ex['t_end']} "
                f"panels={panels}]({p})"
            )
        else:
            index_lines.append(f"- ERROR: {w}")
    (ex_dir / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    return summary
