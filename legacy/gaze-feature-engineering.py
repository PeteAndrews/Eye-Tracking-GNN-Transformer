#!/usr/bin/env python3
"""
Build fixation-level feature tables from sample-level Tobii TSV exports.

This script converts sample-level rows (~250Hz) into event-level rows, then outputs
only fixations with features suitable for continuous HMM modeling.

Usage examples
--------------

Build fixation events from a folder of AOI-processed TSVs:

```bash
python gaze-feature-engineering.py --input _data/phase3-aoi-processed --output _data/phase4-fixation-events
```

Drop fixations with too-low validity (any-eye valid rate < 0.5):

```bash
python gaze-feature-engineering.py --input _data/phase3-aoi-processed --output _data/phase4-fixation-events --min-valid-any 0.5
```

Strict mode: drop AOI-overlap events (aoi_active_count > 1) instead of keeping them:

```bash
python gaze-feature-engineering.py --input _data/phase3-aoi-processed --output _data/phase4-fixation-events --strict
```
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Constants / column helpers
# -----------------------------

COL_PARTICIPANT = "Participant ID"
COL_TRIAL = "Trial"
COL_SENSOR = "Sensor"
COL_STAR_CHART = "Star Chart"
COL_QUESTION_TYPE = "Question Type"
COL_QUESTION_TYPE_ALT = "Question type"

COL_REC_TS = "Recording timestamp"
COL_EYE_MOVE_TYPE = "Eye movement type"
COL_EYE_MOVE_INDEX = "Eye movement type index"
COL_EVENT_DUR = "Gaze event duration"

COL_VALID_L = "Validity left"
COL_VALID_R = "Validity right"

COL_PUPIL_F = "Pupil diameter filtered"
COL_PUPIL_L = "Pupil diameter left"
COL_PUPIL_R = "Pupil diameter right"

COL_AOI_LABEL = "AOI_label"

COL_FIX_X_MCS = "Fixation point X (MCSnorm)"
COL_FIX_Y_MCS = "Fixation point Y (MCSnorm)"

COL_GAZE_X_MCS = "Gaze point X (MCSnorm)"
COL_GAZE_Y_MCS = "Gaze point Y (MCSnorm)"
COL_GAZE_X_L_MCS = "Gaze point left X (MCSnorm)"
COL_GAZE_Y_L_MCS = "Gaze point left Y (MCSnorm)"
COL_GAZE_X_R_MCS = "Gaze point right X (MCSnorm)"
COL_GAZE_Y_R_MCS = "Gaze point right Y (MCSnorm)"

AOI_ONEHOT_PREFIX = "AOI__"


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _find_tsv_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.tsv") if p.is_file()])


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _first_present_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _safe_mode(series: pd.Series) -> Optional[str]:
    s = series.dropna()
    if len(s) == 0:
        return None
    # If multiple modes, take the first stable one
    try:
        m = s.mode(dropna=True)
        if len(m) > 0:
            return str(m.iloc[0])
    except Exception:
        pass
    return str(s.iloc[0])


def _is_nonempty(s: pd.Series) -> pd.Series:
    # Treat empty-string and whitespace as empty; NaN empty
    return s.notna() & (s.astype(str).str.strip() != "")


@dataclass(frozen=True)
class BuildArgs:
    input_dir: Path
    output_dir: Path
    min_valid_any: float
    strict: bool


class FixationFeatureBuilder:
    def __init__(self, *, min_valid_any: float = 0.0, strict: bool = False):
        self.min_valid_any = float(min_valid_any)
        self.strict = bool(strict)
        # Per-file stats populated during processing (for logging)
        self._last_aoi_overlap_count: int = 0
        self._last_prev_sacc_found_rate: float = float("nan")
        self._last_fix_multiple_aoi_count: int = 0

    # ---------
    # Public API
    # ---------
    def process_file(self, input_file: Path, output_dir: Path) -> None:
        df = self._read_tsv(input_file)
        df = self._filter_eye_tracker_rows(df)

        if len(df) == 0:
            print(f"\n{input_file.name}: no eye-tracker rows after filtering; skipping.")
            return

        sample_period_ms = self._compute_sample_period_ms(df)

        events = self._build_event_table(df, sample_period_ms=sample_period_ms, file_for_errors=input_file)
        fixations = self._attach_prev_saccade_features(events)
        fixations_out, removed_rate = self._apply_min_valid_any(fixations)

        out_path = output_dir / f"{input_file.stem}_fixations.tsv"
        _ensure_dir(out_path.parent)
        fixations_out.to_csv(out_path, sep="\t", index=False)

        # Logging (per file)
        n_events = len(events)
        n_fix = len(fixations)
        n_out = len(fixations_out)
        n_mismatch = int(events["duration_mismatch"].sum()) if "duration_mismatch" in events.columns else 0
        n_aoi_overlap = self._last_aoi_overlap_count
        prev_found_rate = self._last_prev_sacc_found_rate
        n_fix_multiple_aoi = self._last_fix_multiple_aoi_count

        print(f"\n{input_file.name}")
        print(f"  sample_period_ms: {sample_period_ms:.3f}")
        print(f"  number of events total (fix+sacc): {n_events}")
        print(f"  number of fixations output: {n_out} (raw fixations: {n_fix})")
        print(f"  count of duration_mismatch events: {n_mismatch}")
        print(f"  count of AOI overlaps (aoi_active_count > 1): {n_aoi_overlap}")
        print(f"  number of fixations with AOI_label_clean == 'MultipleAOI': {n_fix_multiple_aoi}")
        if not math.isnan(prev_found_rate):
            print(f"  % fixations with prev_sacc_found == 1: {prev_found_rate * 100:.2f}%")
            print(f"  % fixations with prev_sacc_found == 0: {(1.0 - prev_found_rate) * 100:.2f}%")
        if self.min_valid_any > 0:
            print(f"  % fixations removed by --min-valid-any: {removed_rate * 100:.2f}%")
        print(f"  output: {out_path}")

    # -----------------
    # Reading & filtering
    # -----------------
    def _read_tsv(self, path: Path) -> pd.DataFrame:
        # correctness-first; avoid dtype guessing pitfalls
        return pd.read_csv(path, sep="\t", low_memory=False)

    def _filter_eye_tracker_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        # Marker rows (START/STOP) may exist; exclude from event aggregation.
        # Do NOT enforce Eyetracker timestamp non-empty.
        if COL_SENSOR in df.columns:
            df = df[df[COL_SENSOR].astype(str).str.strip() == "Eye Tracker"]

        if COL_EYE_MOVE_TYPE not in df.columns or COL_EYE_MOVE_INDEX not in df.columns:
            raise ValueError(f"Missing required columns: {COL_EYE_MOVE_TYPE!r} and/or {COL_EYE_MOVE_INDEX!r}")

        df = df[_is_nonempty(df[COL_EYE_MOVE_TYPE]) & _is_nonempty(df[COL_EYE_MOVE_INDEX])]
        return df

    # -------------
    # Step 1
    # -------------
    def _compute_sample_period_ms(self, df: pd.DataFrame) -> float:
        if COL_REC_TS not in df.columns:
            raise ValueError(f"Missing required column: {COL_REC_TS!r}")

        ts = pd.to_numeric(df[COL_REC_TS], errors="coerce")
        diffs = ts.diff().dropna()
        diffs = diffs[diffs > 0]  # guard against trial resets or weird rows
        if len(diffs) == 0:
            return 4.0  # fallback
        return float(diffs.median())

    # ----------------------------
    # Step 2/3/4: event aggregation
    # ----------------------------
    def _build_event_table(self, df: pd.DataFrame, *, sample_period_ms: float, file_for_errors: Path) -> pd.DataFrame:
        # Sort so groupwise first/last correspond to event temporal ends.
        _coerce_numeric(df, [COL_REC_TS, COL_EVENT_DUR, COL_PUPIL_F, COL_PUPIL_L, COL_PUPIL_R,
                            COL_FIX_X_MCS, COL_FIX_Y_MCS,
                            COL_GAZE_X_MCS, COL_GAZE_Y_MCS,
                            COL_GAZE_X_L_MCS, COL_GAZE_Y_L_MCS,
                            COL_GAZE_X_R_MCS, COL_GAZE_Y_R_MCS])

        # Eye movement type index is sometimes numeric; keep it numeric for grouping if possible.
        df[COL_EYE_MOVE_INDEX] = pd.to_numeric(df[COL_EYE_MOVE_INDEX], errors="coerce").astype("Int64")

        missing_keys = [c for c in (COL_PARTICIPANT, COL_TRIAL, COL_EYE_MOVE_INDEX) if c not in df.columns]
        if missing_keys:
            raise ValueError(f"Missing required columns: {missing_keys}")

        df = df.sort_values([COL_PARTICIPANT, COL_TRIAL, COL_REC_TS], kind="mergesort")

        # Detect AOI one-hot columns dynamically
        aoi_onehots = [c for c in df.columns if c.startswith(AOI_ONEHOT_PREFIX)]

        # Create event_id per (Participant, Trial) as a run-length encoding over
        # (Eye movement type index, Eye movement type). This avoids grouping by index alone.
        df = df.sort_values([COL_PARTICIPANT, COL_TRIAL, COL_REC_TS], kind="mergesort")
        pt = [COL_PARTICIPANT, COL_TRIAL]
        idx_shift = df.groupby(pt, sort=False, dropna=False)[COL_EYE_MOVE_INDEX].shift(1)
        typ_shift = df.groupby(pt, sort=False, dropna=False)[COL_EYE_MOVE_TYPE].shift(1)
        new_event = (df[COL_EYE_MOVE_INDEX] != idx_shift) | (df[COL_EYE_MOVE_TYPE] != typ_shift)
        # Ensure no missing values (pd.NA can appear from nullable comparisons)
        new_event = new_event.astype("boolean").fillna(True)
        df["_new_event"] = new_event.astype("int8")
        df["event_id"] = df.groupby(pt, sort=False, dropna=False)["_new_event"].cumsum().astype("int64")

        group_cols = [COL_PARTICIPANT, COL_TRIAL, "event_id"]
        g = df.groupby(group_cols, dropna=False, sort=False)

        # Named aggregation (robust and readable)
        agg_kwargs = {
            "ts": (COL_REC_TS, "min"),
            "ts_max": (COL_REC_TS, "max"),
            "dur_event_ms": (COL_EVENT_DUR, "first"),
        }
        # Traceability: keep original event index + type, but do not use as grouping key.
        agg_kwargs[COL_EYE_MOVE_INDEX] = (COL_EYE_MOVE_INDEX, "first")
        agg_kwargs[COL_EYE_MOVE_TYPE] = (COL_EYE_MOVE_TYPE, "first")
        if COL_STAR_CHART in df.columns:
            agg_kwargs[COL_STAR_CHART] = (COL_STAR_CHART, "first")

        question_type_col = _first_present_column(df, [COL_QUESTION_TYPE, COL_QUESTION_TYPE_ALT])
        if question_type_col is not None:
            # Standardize exported column name while supporting both input variants.
            agg_kwargs[COL_QUESTION_TYPE] = (question_type_col, "first")

        # Validity rates: compute at sample-level then mean over event samples
        if COL_VALID_L in df.columns and COL_VALID_R in df.columns:
            left_valid = df[COL_VALID_L].astype(str) == "Valid"
            right_valid = df[COL_VALID_R].astype(str) == "Valid"
            df["_valid_any"] = (left_valid | right_valid).astype(float)
            df["_valid_both"] = (left_valid & right_valid).astype(float)
            agg_kwargs["valid_any_rate"] = ("_valid_any", "mean")
            agg_kwargs["valid_both_rate"] = ("_valid_both", "mean")

        # Pupil medians
        if COL_PUPIL_F in df.columns:
            agg_kwargs["pupil_med"] = (COL_PUPIL_F, "median")
        if COL_PUPIL_L in df.columns:
            agg_kwargs["pupil_med_l"] = (COL_PUPIL_L, "median")
        if COL_PUPIL_R in df.columns:
            agg_kwargs["pupil_med_r"] = (COL_PUPIL_R, "median")

        # Pupil missing rate (filtered)
        if COL_PUPIL_F in df.columns:
            df["_pupil_missing"] = df[COL_PUPIL_F].isna().astype(float)
            agg_kwargs["pupil_missing_rate"] = ("_pupil_missing", "mean")

        # Fixation point medians (MCSnorm) + gaze medians (MCSnorm)
        if COL_FIX_X_MCS in df.columns:
            agg_kwargs["fix_pt_x_med"] = (COL_FIX_X_MCS, "median")
        if COL_FIX_Y_MCS in df.columns:
            agg_kwargs["fix_pt_y_med"] = (COL_FIX_Y_MCS, "median")
        if COL_GAZE_X_MCS in df.columns:
            agg_kwargs["gaze_x_med"] = (COL_GAZE_X_MCS, "median")
        if COL_GAZE_Y_MCS in df.columns:
            agg_kwargs["gaze_y_med"] = (COL_GAZE_Y_MCS, "median")
        if COL_GAZE_X_L_MCS in df.columns:
            agg_kwargs["gaze_x_l_med"] = (COL_GAZE_X_L_MCS, "median")
        if COL_GAZE_Y_L_MCS in df.columns:
            agg_kwargs["gaze_y_l_med"] = (COL_GAZE_Y_L_MCS, "median")
        if COL_GAZE_X_R_MCS in df.columns:
            agg_kwargs["gaze_x_r_med"] = (COL_GAZE_X_R_MCS, "median")
        if COL_GAZE_Y_R_MCS in df.columns:
            agg_kwargs["gaze_y_r_med"] = (COL_GAZE_Y_R_MCS, "median")

        # Saccade geometry inputs: first/last binocular gaze points
        if COL_GAZE_X_MCS in df.columns:
            agg_kwargs["gaze_x0"] = (COL_GAZE_X_MCS, "first")
            agg_kwargs["gaze_x1"] = (COL_GAZE_X_MCS, "last")
        if COL_GAZE_Y_MCS in df.columns:
            agg_kwargs["gaze_y0"] = (COL_GAZE_Y_MCS, "first")
            agg_kwargs["gaze_y1"] = (COL_GAZE_Y_MCS, "last")

        # AOI onehots: max over samples (any-hit)
        for c in aoi_onehots:
            agg_kwargs[c] = (c, "max")

        # AOI label (mode)
        if COL_AOI_LABEL in df.columns:
            agg_kwargs[COL_AOI_LABEL] = (COL_AOI_LABEL, _safe_mode)

        core = g.agg(**agg_kwargs).reset_index()
        if COL_EYE_MOVE_TYPE not in core.columns or COL_EYE_MOVE_INDEX not in core.columns:
            raise RuntimeError("Event traceability fields missing after aggregation")

        core["dur_ts_ms"] = (core["ts_max"] - core["ts"]) + float(sample_period_ms)
        core["duration_mismatch"] = (core["dur_event_ms"] - core["dur_ts_ms"]).abs() > (2.0 * float(sample_period_ms))
        core["dur_ms"] = np.where(core["duration_mismatch"], core["dur_ts_ms"], core["dur_event_ms"])

        # If missing validity/pupil columns, ensure output fields exist
        if "valid_any_rate" not in core.columns:
            core["valid_any_rate"] = np.nan
        if "valid_both_rate" not in core.columns:
            core["valid_both_rate"] = np.nan
        if "pupil_missing_rate" not in core.columns:
            core["pupil_missing_rate"] = np.nan

        # AOI_label_clean and AOI__NoAOI
        if COL_AOI_LABEL in core.columns:
            aoi_label_raw = core[COL_AOI_LABEL].astype(str)
            core["AOI_label_clean"] = np.where(aoi_label_raw.str.strip().str.lower() == "outside", "NoAOI", aoi_label_raw)
        else:
            core["AOI_label_clean"] = "NoAOI"

        core["AOI__NoAOI"] = (core["AOI_label_clean"] == "NoAOI").astype(int)
        # Always ensure MultipleAOI flag exists for downstream schema
        core["AOI__MultipleAOI"] = 0

        # AOI overlap check: count active AOI__ columns excluding NoAOI
        aoi_cols_ex_no = [
            c
            for c in core.columns
            if c.startswith(AOI_ONEHOT_PREFIX) and c not in ("AOI__NoAOI", "AOI__MultipleAOI")
        ]
        if aoi_cols_ex_no:
            # Preserve original overlap information for logging/QA before enforcing exclusivity.
            core["aoi_active_count_original"] = core[aoi_cols_ex_no].fillna(0).astype(float).sum(axis=1)
            core["is_aoi_overlap_original"] = core["aoi_active_count_original"] > 1

            # The modeling-facing count is stored in aoi_active_count (may be overwritten below).
            core["aoi_active_count"] = core["aoi_active_count_original"]

            overlaps = core[core["is_aoi_overlap_original"]]
            self._last_aoi_overlap_count = int(overlaps.shape[0])
            if len(overlaps) > 0:
                print(
                    f"\nAOI overlaps detected in {file_for_errors.name}: {len(overlaps)} event(s) "
                    f"with aoi_active_count_original > 1"
                )
                is_ov = core["is_aoi_overlap_original"]
                if not self.strict:
                    # Keep but make mutually exclusive:
                    # - label MultipleAOI
                    # - set AOI__MultipleAOI=1
                    # - zero all other AOI one-hots and NoAOI
                    core.loc[is_ov, "AOI_label_clean"] = "MultipleAOI"
                    core.loc[is_ov, "AOI__MultipleAOI"] = 1
                    core.loc[is_ov, "AOI__NoAOI"] = 0
                    if aoi_cols_ex_no:
                        core.loc[is_ov, aoi_cols_ex_no] = 0
                    # After enforcing exclusivity, treat as a single active AOI for downstream modeling
                    core.loc[is_ov, "aoi_active_count"] = 1

                show = core[is_ov].head(5)
                show_cols = group_cols + [
                    COL_EYE_MOVE_TYPE,
                    "ts",
                    "aoi_active_count_original",
                    "aoi_active_count",
                    "AOI_label_clean",
                ]
                show_cols = [c for c in show_cols if c in show.columns]
                print(show[show_cols].to_string(index=False))

                if self.strict:
                    # Drop overlapped events entirely in strict mode
                    core = core[~is_ov].copy()
        else:
            core["aoi_active_count"] = 0
            self._last_aoi_overlap_count = 0

        # Fixation-specific features + fallback for missing fixation point fields
        core["fix_x"] = np.nan
        core["fix_y"] = np.nan
        core["fix_x_l"] = np.nan
        core["fix_y_l"] = np.nan
        core["fix_x_r"] = np.nan
        core["fix_y_r"] = np.nan

        is_fix = core[COL_EYE_MOVE_TYPE].astype(str) == "Fixation"
        if "fix_pt_x_med" in core.columns:
            core.loc[is_fix, "fix_x"] = core.loc[is_fix, "fix_pt_x_med"]
        if "fix_pt_y_med" in core.columns:
            core.loc[is_fix, "fix_y"] = core.loc[is_fix, "fix_pt_y_med"]

        # Optional gaze medians (left/right)
        if "gaze_x_l_med" in core.columns:
            core.loc[is_fix, "fix_x_l"] = core.loc[is_fix, "gaze_x_l_med"]
        if "gaze_y_l_med" in core.columns:
            core.loc[is_fix, "fix_y_l"] = core.loc[is_fix, "gaze_y_l_med"]
        if "gaze_x_r_med" in core.columns:
            core.loc[is_fix, "fix_x_r"] = core.loc[is_fix, "gaze_x_r_med"]
        if "gaze_y_r_med" in core.columns:
            core.loc[is_fix, "fix_y_r"] = core.loc[is_fix, "gaze_y_r_med"]

        # Fallback: if fixation point medians missing, use binocular gaze medians
        if "gaze_x_med" in core.columns and "gaze_y_med" in core.columns:
            need_fallback = is_fix & (pd.isna(core["fix_x"]) | pd.isna(core["fix_y"]))
            core.loc[need_fallback, "fix_x"] = core.loc[need_fallback, "gaze_x_med"]
            core.loc[need_fallback, "fix_y"] = core.loc[need_fallback, "gaze_y_med"]

        # Saccade-specific features
        for c in ["sacc_dx", "sacc_dy", "sacc_amp", "sacc_angle", "sacc_is_regression", "sacc_speed"]:
            core[c] = np.nan

        is_sacc = core[COL_EYE_MOVE_TYPE].astype(str) == "Saccade"
        x0 = core.get("gaze_x0", np.nan)
        y0 = core.get("gaze_y0", np.nan)
        x1 = core.get("gaze_x1", np.nan)
        y1 = core.get("gaze_y1", np.nan)

        dx = (x1 - x0).astype(float)
        dy = (y1 - y0).astype(float)
        amp = np.sqrt(dx ** 2 + dy ** 2)
        angle = np.arctan2(dy, dx)
        speed = amp / core["dur_ms"].astype(float)

        core.loc[is_sacc, "sacc_dx"] = dx[is_sacc]
        core.loc[is_sacc, "sacc_dy"] = dy[is_sacc]
        core.loc[is_sacc, "sacc_amp"] = amp[is_sacc]
        core.loc[is_sacc, "sacc_angle"] = angle[is_sacc]
        core.loc[is_sacc, "sacc_is_regression"] = (dx[is_sacc] < 0).astype(float)
        core.loc[is_sacc, "sacc_speed"] = speed[is_sacc]

        # Keep only needed columns downstream; but preserve enough for prev-saccade attachment
        return core

    # ------------------------
    # Step 5: attach prev sacc
    # ------------------------
    def _attach_prev_saccade_features(self, events: pd.DataFrame) -> pd.DataFrame:
        # Sort by ts within participant/trial
        pt_cols = [COL_PARTICIPANT, COL_TRIAL]
        events = events.sort_values([*pt_cols, "ts"], kind="mergesort").reset_index(drop=True)

        # Robust "previous saccade" features:
        # - Create saccade-only columns (feature on saccade rows, else NaN)
        # - Forward-fill within (participant, trial) to carry the most recent saccade so far
        # - Shift(1) within group to ensure "previous" (exclude a saccade on the same row)
        is_sacc = events[COL_EYE_MOVE_TYPE].astype(str) == "Saccade"

        sacc_map = {
            "dur_ms": "prev_sacc_dur_ms",
            "sacc_dx": "prev_sacc_dx",
            "sacc_dy": "prev_sacc_dy",
            "sacc_amp": "prev_sacc_amp",
            "sacc_angle": "prev_sacc_angle",
            "sacc_is_regression": "prev_sacc_is_regression",
            "sacc_speed": "prev_sacc_speed",
        }

        sacc_only_cols: List[str] = []
        for src_col in sacc_map.keys():
            only_col = f"_sacc_{src_col}_only"
            sacc_only_cols.append(only_col)
            if src_col in events.columns:
                events[only_col] = np.where(is_sacc, events[src_col].astype(float), np.nan)
            else:
                events[only_col] = np.nan

        # Forward-fill the saccade-only columns within each (Participant, Trial),
        # then shift by 1 to ensure we attach the most recent *preceding* saccade.
        events[sacc_only_cols] = events.groupby(pt_cols, sort=False, dropna=False)[sacc_only_cols].ffill()
        prev = events.groupby(pt_cols, sort=False, dropna=False)[sacc_only_cols].shift(1)

        for src_col, out_col in sacc_map.items():
            only_col = f"_sacc_{src_col}_only"
            events[out_col] = prev[only_col]

        events["prev_sacc_found"] = events["prev_sacc_dur_ms"].notna().astype(int)

        # Drop temporary helper columns before output
        events = events.drop(columns=sacc_only_cols, errors="ignore")

        # Output fixation rows only with schema
        fix = events[events[COL_EYE_MOVE_TYPE].astype(str) == "Fixation"].copy()

        fix["fix_dur_ms"] = fix["dur_ms"]

        # Ensure AOI columns exist (including NoAOI)
        required_aoi = [
            "AOI__Commentary",
            "AOI__Green_Score_Box",
            "AOI__Grey_Score_Box",
            "AOI__Mark_Scheme",
            "AOI__Next_Arrow",
            "AOI__Question",
            "AOI__Response",
            "AOI__NoAOI",
            "AOI__MultipleAOI",
        ]
        for c in required_aoi:
            if c not in fix.columns:
                fix[c] = 0

        # Build output columns in requested order
        out_cols = [
            COL_PARTICIPANT,
            COL_TRIAL,
            COL_STAR_CHART,
            COL_QUESTION_TYPE,
            "ts",
            "fix_dur_ms",
            "fix_x",
            "fix_y",
            "fix_x_l",
            "fix_y_l",
            "fix_x_r",
            "fix_y_r",
            "valid_any_rate",
            "valid_both_rate",
            "pupil_med",
            "pupil_med_l",
            "pupil_med_r",
            "pupil_missing_rate",
            "AOI_label_clean",
            *required_aoi,
            "aoi_active_count",
            "prev_sacc_found",
            "prev_sacc_dur_ms",
            "prev_sacc_dx",
            "prev_sacc_dy",
            "prev_sacc_amp",
            "prev_sacc_angle",
            "prev_sacc_is_regression",
            "prev_sacc_speed",
        ]
        for c in out_cols:
            if c not in fix.columns:
                fix[c] = np.nan

        # Per-file summary: % of fixations that found a prior saccade
        if len(fix) == 0:
            self._last_prev_sacc_found_rate = float("nan")
            self._last_fix_multiple_aoi_count = 0
        else:
            self._last_prev_sacc_found_rate = float(pd.to_numeric(fix["prev_sacc_found"], errors="coerce").fillna(0).mean())
            self._last_fix_multiple_aoi_count = int((fix["AOI_label_clean"].astype(str) == "MultipleAOI").sum())

        return fix[out_cols]

    def _apply_min_valid_any(self, fixations: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        if self.min_valid_any <= 0:
            return fixations, 0.0
        n0 = len(fixations)
        out = fixations[fixations["valid_any_rate"].astype(float) >= self.min_valid_any].copy()
        removed = 0.0 if n0 == 0 else (n0 - len(out)) / n0
        return out, removed


def _parse_args(argv: Optional[Sequence[str]] = None) -> BuildArgs:
    p = argparse.ArgumentParser(description="Build fixation-level event features from Tobii sample TSVs.")
    p.add_argument("--input", required=True, help="Folder containing TSVs; process all *.tsv recursively")
    p.add_argument("--output", required=True, help="Folder to write output TSVs")
    p.add_argument("--min-valid-any", type=float, default=0.0, help="Optionally drop fixations with valid_any_rate below this")
    p.add_argument("--strict", action="store_true", help="Error out when unexpected multi-AOI overlaps or inconsistent event typing occur")
    args = p.parse_args(argv)

    return BuildArgs(
        input_dir=_as_path(args.input),
        output_dir=_as_path(args.output),
        min_valid_any=float(args.min_valid_any),
        strict=bool(args.strict),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    tsv_files = _find_tsv_files(input_dir)
    # Safety: if output is inside input, don't recursively re-process generated files.
    try:
        out_resolved = output_dir.resolve()
        tsv_files = [p for p in tsv_files if not p.resolve().is_relative_to(out_resolved)]
    except Exception:
        # is_relative_to not available or resolve failed; fall back to string prefix check
        out_str = str(output_dir.resolve()) if output_dir.exists() else str(output_dir)
        tsv_files = [p for p in tsv_files if not str(p).startswith(out_str)]
    if not tsv_files:
        print(f"Found 0 TSV files under: {input_dir}")
        return 0

    print(f"Found {len(tsv_files)} TSV file(s) to process")
    _ensure_dir(output_dir)

    builder = FixationFeatureBuilder(min_valid_any=args.min_valid_any, strict=args.strict)
    for f in tsv_files:
        try:
            builder.process_file(f, output_dir)
        except Exception as e:
            print(f"\nERROR processing {f}: {e}", file=sys.stderr)
            if args.strict:
                raise

    print("\nProcessing complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

