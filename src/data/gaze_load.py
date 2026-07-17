"""P1 — gaze table pruning and tidying.

Loads Tobii sample-level TSVs, filters to Eye Tracker rows, renames to
snake_case, drops unused columns, and writes per-participant parquet plus
per-episode QC. DACSmm columns are summarised for P6 ε derivation before drop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils import io as uio

# ---------------------------------------------------------------------------
# Column contracts (preprocessing plan P1) — single source of truth
# ---------------------------------------------------------------------------

COL_SENSOR = "Sensor"
COL_PARTICIPANT = "Participant ID"
COL_TRIAL = "Trial"
COL_TRIAL_RAW = "Trial Raw"
COL_REC_TS = "Recording timestamp"
COL_STAR = "Star Chart"
COL_QTYPE = "Question type"

# Extracted for P6 ε before drop (not kept in pruned table)
DACSMM_COLS = [
    "Eye position left X (DACSmm)",
    "Eye position left Y (DACSmm)",
    "Eye position left Z (DACSmm)",
    "Eye position right X (DACSmm)",
    "Eye position right Y (DACSmm)",
    "Eye position right Z (DACSmm)",
    "Gaze point left X (DACSmm)",
    "Gaze point left Y (DACSmm)",
    "Gaze point right X (DACSmm)",
    "Gaze point right Y (DACSmm)",
]

PIXEL_GAZE_COLS = [
    "Gaze point X",
    "Gaze point Y",
    "Gaze point left X",
    "Gaze point left Y",
    "Gaze point right X",
    "Gaze point right Y",
]

KEEP_COLS = [
    COL_PARTICIPANT,
    COL_REC_TS,
    COL_TRIAL,
    COL_STAR,
    COL_QTYPE,
    "Eye movement type",
    "Eye movement type index",
    "Gaze event duration",
    "Validity left",
    "Validity right",
    "Pupil diameter left",
    "Pupil diameter right",
    "Pupil diameter filtered",
    "Fixation point X",
    "Fixation point Y",
    "Fixation point X (MCSnorm)",
    "Fixation point Y (MCSnorm)",
    "Gaze point X",
    "Gaze point Y",
    "Gaze point X (doc)",
    "Gaze point Y (doc)",
    "scroll_offset_y",
    "scroll_ratio",
    "gaze_region",
    "AOI_label",
    "AOI__Advance",
    "AOI__Commentary",
    "AOI__Green_Answer_Box",
    "AOI__Grey_Answer_Box",
    "AOI__Mark_Scheme",
    "AOI__Question",
    "AOI__Response",
    "correction_applied",
    "left_offset_px",
    "calibration_key",
]

# Dropped after Sensor filter / Trial Raw cross-check / ε extraction
DROP_AFTER_USE = [
    COL_SENSOR,
    COL_TRIAL_RAW,
    "scroll_correction_flag",
    "left_correction_flag",
    "Project name",
    "Ungrouped",
    "Event",
    "Event value",
    "Computer timestamp",
    "Eyetracker timestamp",
    "Gaze direction left X",
    "Gaze direction left Y",
    "Gaze direction left Z",
    "Gaze direction right X",
    "Gaze direction right Y",
    "Gaze direction right Z",
    "Gaze point left X (MCSnorm)",
    "Gaze point left Y (MCSnorm)",
    "Gaze point right X (MCSnorm)",
    "Gaze point right Y (MCSnorm)",
    "Gaze point X (MCSnorm)",
    "Gaze point Y (MCSnorm)",
    *DACSMM_COLS,
]

RENAME_MAP = {
    COL_PARTICIPANT: "participant_id",
    COL_REC_TS: "recording_timestamp",
    COL_TRIAL: "trial_id",
    COL_STAR: "star_chart",
    COL_QTYPE: "question_type",
    "Eye movement type": "eye_movement_type",
    "Eye movement type index": "eye_movement_type_index",
    "Gaze event duration": "gaze_event_duration",
    "Validity left": "validity_left",
    "Validity right": "validity_right",
    "Pupil diameter left": "pupil_diameter_left",
    "Pupil diameter right": "pupil_diameter_right",
    "Pupil diameter filtered": "pupil_diameter_filtered",
    "Fixation point X": "fixation_point_x",
    "Fixation point Y": "fixation_point_y",
    "Fixation point X (MCSnorm)": "fixation_point_x_mcsnorm",
    "Fixation point Y (MCSnorm)": "fixation_point_y_mcsnorm",
    "Gaze point X": "gaze_point_x",
    "Gaze point Y": "gaze_point_y",
    "Gaze point X (doc)": "gaze_point_x_doc",
    "Gaze point Y (doc)": "gaze_point_y_doc",
    "scroll_offset_y": "scroll_offset_y",
    "scroll_ratio": "scroll_ratio",
    "gaze_region": "gaze_region",
    "AOI_label": "aoi_label",
    "AOI__Advance": "aoi__advance",
    "AOI__Commentary": "aoi__commentary",
    "AOI__Green_Answer_Box": "aoi__green_answer_box",
    "AOI__Grey_Answer_Box": "aoi__grey_answer_box",
    "AOI__Mark_Scheme": "aoi__mark_scheme",
    "AOI__Question": "aoi__question",
    "AOI__Response": "aoi__response",
    "correction_applied": "correction_applied",
    "left_offset_px": "left_offset_px",
    "calibration_key": "calibration_key",
}


def expected_pruned_columns() -> list[str]:
    return [RENAME_MAP[c] for c in KEEP_COLS]


@dataclass
class EpisodeQC:
    participant_id: str
    trial_id: str
    n_rows: int
    n_correction_false: int
    n_trial_raw_disagree: int
    n_empty_trial_dropped: int
    timestamp_monotonic: bool
    sample_period_ms_median: float
    eye_z_median_mm: float
    mm_per_px_x: float
    mm_per_px_y: float


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", encoding="utf-8", low_memory=False)


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def extract_epsilon_inputs(df: pd.DataFrame) -> dict[str, float]:
    """Per-file summary for P6 ε derivation (DACSmm ↔ pixel regression)."""
    out = {
        "eye_z_median_mm": float("nan"),
        "mm_per_px_x": float("nan"),
        "mm_per_px_y": float("nan"),
        "n_pairs_xy": 0,
    }
    z_cols = ["Eye position left Z (DACSmm)", "Eye position right Z (DACSmm)"]
    z_vals = []
    for c in z_cols:
        if c in df.columns:
            z_vals.append(pd.to_numeric(df[c], errors="coerce"))
    if z_vals:
        z = pd.concat(z_vals, axis=0).dropna()
        if len(z):
            out["eye_z_median_mm"] = float(z.median())

    # Regress DACSmm gaze vs pixel gaze (left eye if present)
    pairs = [
        ("Gaze point left X (DACSmm)", "Gaze point left X"),
        ("Gaze point right X (DACSmm)", "Gaze point right X"),
    ]
    slopes_x = []
    for dacs, px in pairs:
        if dacs in df.columns and px in df.columns:
            a = pd.to_numeric(df[dacs], errors="coerce")
            b = pd.to_numeric(df[px], errors="coerce")
            mask = a.notna() & b.notna() & (b != 0)
            if mask.sum() >= 5:
                # mm per pixel ≈ median(|dacs|/|px|) as robust slope proxy near origin;
                # also fit least-squares through origin.
                b_m, a_m = b[mask].to_numpy(dtype=float), a[mask].to_numpy(dtype=float)
                denom = float(np.dot(b_m, b_m))
                if denom > 0:
                    slopes_x.append(float(np.dot(b_m, a_m) / denom))
                    out["n_pairs_xy"] += int(mask.sum())
    pairs_y = [
        ("Gaze point left Y (DACSmm)", "Gaze point left Y"),
        ("Gaze point right Y (DACSmm)", "Gaze point right Y"),
    ]
    slopes_y = []
    for dacs, px in pairs_y:
        if dacs in df.columns and px in df.columns:
            a = pd.to_numeric(df[dacs], errors="coerce")
            b = pd.to_numeric(df[px], errors="coerce")
            mask = a.notna() & b.notna()
            if mask.sum() >= 5:
                b_m, a_m = b[mask].to_numpy(dtype=float), a[mask].to_numpy(dtype=float)
                denom = float(np.dot(b_m, b_m))
                if denom > 0:
                    slopes_y.append(float(np.dot(b_m, a_m) / denom))

    if slopes_x:
        out["mm_per_px_x"] = float(np.median(slopes_x))
    if slopes_y:
        out["mm_per_px_y"] = float(np.median(slopes_y))
    return out


def prune_gaze_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, float]]:
    """Filter, QC, rename, and drop columns. Returns (pruned, episode_qc_rows, eps_inputs)."""
    n_in = len(df)
    df = df[df[COL_SENSOR] == "Eye Tracker"].copy()

    # Trial Raw cross-check before drop
    trial = df[COL_TRIAL].fillna("").astype(str).str.strip()
    trial_raw = (
        df[COL_TRIAL_RAW].fillna("").astype(str).str.strip()
        if COL_TRIAL_RAW in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    empty_trial = trial == ""
    n_empty = int(empty_trial.sum())
    disagree = (~empty_trial) & (trial_raw != "") & (trial != trial_raw)

    eps_inputs = extract_epsilon_inputs(df)

    # Drop empty-trial rows
    df = df.loc[~empty_trial].copy()
    trial = df[COL_TRIAL].fillna("").astype(str).str.strip().str.upper()
    df[COL_TRIAL] = trial

    if "correction_applied" in df.columns:
        corr_false = ~_as_bool_series(df["correction_applied"])
    else:
        corr_false = pd.Series(False, index=df.index)

    # Per-episode QC (before column drop)
    qc_rows: list[dict[str, Any]] = []
    pid_col = df[COL_PARTICIPANT].astype(str)
    for (pid, tid), grp in df.groupby([pid_col, COL_TRIAL], sort=False):
        ts = pd.to_numeric(grp[COL_REC_TS], errors="coerce")
        diffs = ts.diff().dropna()
        mono = bool(ts.is_monotonic_increasing) if len(ts) else True
        period = float(diffs.median()) if len(diffs) else float("nan")
        # trial-raw disagrees within this episode (from pre-filter mask aligned by index)
        n_dis = int(disagree.reindex(grp.index, fill_value=False).sum())
        n_cf = int(corr_false.reindex(grp.index, fill_value=False).sum())
        qc_rows.append(
            {
                "participant_id": str(pid),
                "trial_id": str(tid),
                "n_rows": int(len(grp)),
                "n_correction_false": n_cf,
                "n_trial_raw_disagree": n_dis,
                "n_empty_trial_dropped": n_empty,  # file-level; same for all episodes
                "timestamp_monotonic": mono,
                "sample_period_ms_median": period,
                "eye_z_median_mm": eps_inputs["eye_z_median_mm"],
                "mm_per_px_x": eps_inputs["mm_per_px_x"],
                "mm_per_px_y": eps_inputs["mm_per_px_y"],
                "n_rows_in_file": n_in,
            }
        )

    missing_keep = [c for c in KEEP_COLS if c not in df.columns]
    if missing_keep:
        raise KeyError(f"Missing required columns: {missing_keep}")

    pruned = df[KEEP_COLS].rename(columns=RENAME_MAP)
    pruned = pruned.sort_values(
        ["participant_id", "trial_id", "recording_timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    return pruned, qc_rows, eps_inputs


def prune_gaze_file(tsv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    raw = _read_tsv(tsv_path)
    pruned, qc_rows, eps = prune_gaze_dataframe(raw)
    return pruned, pd.DataFrame(qc_rows), eps


def run_p1(repo_root: Optional[Path] = None) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(repo_root / "configs" / "data.yaml")
    data_version = str(cfg.data_version)
    gaze_dir = repo_root / str(cfg.paths.gaze_dir)
    out_dir = repo_root / str(cfg.paths.processed_root) / data_version / "gaze_pruned"
    out_dir.mkdir(parents=True, exist_ok=True)

    tsvs = sorted(gaze_dir.glob("*.tsv"))
    all_qc: list[pd.DataFrame] = []
    eps_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for tsv in tsvs:
        try:
            pruned, qc, eps = prune_gaze_file(tsv)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{tsv.name}: {e}")
            continue
        pid = pruned["participant_id"].iloc[0] if len(pruned) else tsv.stem
        safe_pid = str(pid).strip().lower().replace(" ", "")
        out_path = out_dir / f"{safe_pid}.parquet"
        pruned.to_parquet(out_path, index=False)
        all_qc.append(qc)
        eps_rows.append({"source_file": tsv.name, "participant_id": str(pid), **eps})

    qc_all = pd.concat(all_qc, ignore_index=True) if all_qc else pd.DataFrame()
    if len(qc_all):
        qc_all.to_parquet(out_dir / "episode_qc.parquet", index=False)
        uio.write_csv(out_dir / "episode_qc.csv", qc_all.to_dict(orient="records"))
    eps_df = pd.DataFrame(eps_rows)
    if len(eps_df):
        eps_df.to_parquet(out_dir / "epsilon_inputs.parquet", index=False)

    summary = {
        "data_version": data_version,
        "out_dir": str(out_dir),
        "n_files": len(tsvs),
        "n_written": len(eps_rows),
        "n_episodes": int(len(qc_all)),
        "n_correction_false_total": int(qc_all["n_correction_false"].sum()) if len(qc_all) else 0,
        "errors": errors,
        "ok": len(errors) == 0 and len(eps_rows) == len(tsvs),
    }
    uio.write_json(out_dir / "p1_summary.json", summary)
    return summary
