"""P3 — AOI hit injection (star-chart override + additive UI regions).

Operates on P1 pruned gaze at sample level in raw document px. Writes the
canonical gaze table used by all downstream stages (P4–P7).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils import io as uio

STAR_COL = "aoi__star_chart"
STAR_LABEL = "Star_Chart"


def _strict_inside(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
) -> np.ndarray:
    """Strict containment (boundary excluded)."""
    return (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)


def load_panel_index(metadata_dir: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Map (trial_id, star_condition) → panel region rows from P2 outputs."""
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in sorted(metadata_dir.glob("*__panels.json")):
        panels = uio.read_json(path)
        if not panels:
            continue
        key = (str(panels[0]["trial_id"]), str(panels[0]["star_condition"]))
        index[key] = panels
    return index


def _regions_of_type(panels: list[dict[str, Any]], aoi_type: str) -> list[dict[str, Any]]:
    return [p for p in panels if p.get("aoi_type") == aoi_type]


def _has_content_label(label: Any, content_labels: set[str]) -> bool:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return False
    return str(label).strip() in content_labels


def inject_episode(
    df: pd.DataFrame,
    panels: list[dict[str, Any]],
    *,
    star_condition: str,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inject AOI hits into one episode; return (frame, qc row).

    UI hits are additive (one-hots always; label only if no content AOI).
    Star-chart injection (star_on only) overrides label and zeros other one-hots.
    """
    out = df.copy()
    n = len(out)
    x = out["gaze_point_x_doc"].to_numpy(dtype=float)
    y = out["gaze_point_y_doc"].to_numpy(dtype=float)

    star_column = str(cfg.get("star_column", STAR_COL))
    star_label = str(cfg.get("star_label", STAR_LABEL))
    existing = list(cfg.get("existing_aoi_onehots") or [])
    ui_specs = list(cfg.get("ui_regions") or [])
    content_labels = {str(s) for s in (cfg.get("content_aoi_labels") or [])}

    ui_columns = [str(s["column"]) for s in ui_specs]
    all_inject_onehots = ui_columns + [star_column]

    for col in all_inject_onehots:
        out[col] = np.zeros(n, dtype=np.int8)

    # --- UI regions (all episodes) ---
    # Track best (smallest) UI region for label assignment
    best_area = np.full(n, np.inf, dtype=float)
    best_label = np.array([None] * n, dtype=object)

    for spec in ui_specs:
        aoi_type = str(spec["aoi_type"])
        col = str(spec["column"])
        label = str(spec["label"])
        hit = np.zeros(n, dtype=bool)
        for reg in _regions_of_type(panels, aoi_type):
            inside = _strict_inside(
                x, y, float(reg["x_min"]), float(reg["y_min"]), float(reg["x_max"]), float(reg["y_max"])
            )
            hit |= inside
            area = float(reg.get("area") or max(0.0, float(reg["x_max"]) - float(reg["x_min"])) * max(
                0.0, float(reg["y_max"]) - float(reg["y_min"])
            ))
            better = inside & (area < best_area)
            best_area[better] = area
            best_label[better] = label
        out[col] = hit.astype(np.int8)

    labels = out["aoi_label"].to_numpy(copy=True)
    # Additive label: never override content AOI hits; Outside/Advance/empty OK
    allow_ui_label = np.array(
        [not _has_content_label(lab, content_labels) for lab in labels],
        dtype=bool,
    )
    ui_label_mask = allow_ui_label & np.array([lab is not None for lab in best_label], dtype=bool)
    n_ui_relabel = int(ui_label_mask.sum())
    labels = labels.copy()
    labels[ui_label_mask] = best_label[ui_label_mask]

    # --- Star-chart injection (star_on only) ---
    star_hit = np.zeros(n, dtype=bool)
    n_star_relabel = 0
    if star_condition == "star_on":
        for reg in _regions_of_type(panels, "star_chart"):
            star_hit |= _strict_inside(
                x,
                y,
                float(reg["x_min"]),
                float(reg["y_min"]),
                float(reg["x_max"]),
                float(reg["y_max"]),
            )
        out[star_column] = star_hit.astype(np.int8)
        if star_hit.any():
            # Zero all other AOI one-hots (existing + UI)
            for col in existing + ui_columns:
                if col in out.columns:
                    vals = out[col].to_numpy(copy=True)
                    vals[star_hit] = 0
                    out[col] = vals
            before = labels.copy()
            labels[star_hit] = star_label
            n_star_relabel = int((before[star_hit] != star_label).sum())

    out["aoi_label"] = labels

    qc = {
        "n_samples": n,
        "star_condition": star_condition,
        "n_star_hits": int(star_hit.sum()),
        "n_star_relabel": n_star_relabel,
        "star_hit_proportion": float(star_hit.mean()) if n else 0.0,
        "n_ui_label_updates": n_ui_relabel,
    }
    for spec in ui_specs:
        col = str(spec["column"])
        qc[f"n_hit_{col}"] = int(out[col].sum())
    return out, qc


def run_p3(repo_root: Optional[Path] = None) -> dict[str, Any]:
    """Inject AOI hits for all participants; write gaze_canonical + QC."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    data_cfg = OmegaConf.load(repo_root / "configs" / "data.yaml")
    pre_cfg = OmegaConf.load(repo_root / "configs" / "preprocessing.yaml")
    data_version = str(data_cfg.data_version)
    inj_cfg = OmegaConf.to_container(pre_cfg.aoi_injection, resolve=True)
    assert isinstance(inj_cfg, dict)

    pruned_dir = repo_root / str(data_cfg.paths.processed_root) / data_version / "gaze_pruned"
    meta_dir = repo_root / str(data_cfg.paths.processed_root) / data_version / "metadata"
    reg_dir = repo_root / str(data_cfg.paths.processed_root) / data_version / "registry"
    out_dir = repo_root / str(data_cfg.paths.processed_root) / data_version / "gaze_canonical"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels_index = load_panel_index(meta_dir)
    star_tbl = pd.read_parquet(reg_dir / "star_conditions.parquet")
    star_map = {
        (str(r.participant_id), str(r.trial_id)): str(r.star_condition)
        for r in star_tbl.itertuples(index=False)
    }

    qc_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    n_participants = 0

    eligible = {str(t) for t in data_cfg.star_eligible_trials}

    for path in sorted(pruned_dir.glob("p*.parquet")):
        num = path.stem[1:] if path.stem.lower().startswith("p") else path.stem
        pid_key = f"P{int(num):02d}" if str(num).isdigit() else path.stem.upper()
        df = pd.read_parquet(path)
        parts = []
        for trial_id, ep in df.groupby("trial_id", sort=True):
            tid = str(trial_id)
            sc = star_map.get((pid_key, tid))
            if sc is None:
                sc_raw = ep["star_chart"].dropna().unique()
                if tid in eligible:
                    sc = "star_on" if (len(sc_raw) and int(sc_raw[0]) == 1) else "star_off"
                else:
                    sc = "not_eligible"
                errors.append(f"star_map miss {pid_key}/{tid}; inferred {sc}")

            panels = panels_index.get((tid, sc))
            if panels is None:
                errors.append(f"missing panels for {tid}/{sc}")
                parts.append(ep)
                continue

            injected, qc = inject_episode(ep, panels, star_condition=sc, cfg=inj_cfg)
            qc["participant_id"] = pid_key
            qc["trial_id"] = tid
            qc_rows.append(qc)
            parts.append(injected)

        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values(
            ["participant_id", "trial_id", "recording_timestamp"]
        ).reset_index(drop=True)
        out.to_parquet(out_dir / f"{path.stem}.parquet", index=False)
        n_participants += 1

    qc_df = pd.DataFrame(qc_rows)
    qc_path = out_dir / "injection_qc.parquet"
    if len(qc_df):
        qc_df.to_parquet(qc_path, index=False)
        uio.write_json(out_dir / "injection_qc.json", qc_df.to_dict(orient="records"))

    star_eps = qc_df[qc_df["star_condition"] == "star_on"] if len(qc_df) else qc_df
    summary = {
        "data_version": data_version,
        "out_dir": str(out_dir),
        "n_participants": n_participants,
        "n_episodes": len(qc_rows),
        "n_star_on_episodes": int(len(star_eps)),
        "star_hit_samples_total": int(star_eps["n_star_hits"].sum()) if len(star_eps) else 0,
        "star_relabel_total": int(star_eps["n_star_relabel"].sum()) if len(star_eps) else 0,
        "ui_hit_totals": (
            {
                c: int(qc_df[c].sum()) if c in qc_df.columns else 0
                for c in (
                    "n_hit_aoi__answer_scroll_bar",
                    "n_hit_aoi__commentary_scroll_bar",
                    "n_hit_aoi__general_ui",
                )
            }
            if len(qc_df)
            else {}
        ),
        "errors": errors,
        "ok": n_participants == int(data_cfg.expected.n_participants)
        and len(qc_rows) == int(data_cfg.expected.n_participants) * int(data_cfg.expected.n_trials)
        and len(star_eps) == int(data_cfg.expected.n_participants)
        * int(data_cfg.expected.star_on_per_participant)
        and not any("missing panels" in e for e in errors),
        "note": (
            "Scrollbar hit rates are indicative: regions are thin relative to gaze precision."
        ),
    }
    uio.write_json(out_dir / "p3_summary.json", summary)
    return summary
