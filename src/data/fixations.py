"""P6 fixation event construction (legacy aggregation port + doc/scroll/assign/loops).

Event segmentation mirrors ``legacy/gaze-feature-engineering.py``: run-length
IDs over (eye_movement_type_index, eye_movement_type) within (participant, trial),
duration reconciliation, validity/pupil passthrough, prev-saccade attachment.
Coordinates and saccade geometry use raw document px (then DOCnorm for features).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.epsilon import derive_epsilon
from src.data.gaze_assignment import assign_fixations
from src.data.loops import annotate_loops
from src.utils import io as uio


def compute_sample_period_ms(timestamps: np.ndarray) -> float:
    if len(timestamps) < 2:
        return 4.0  # ~250 Hz default
    d = np.diff(np.sort(timestamps.astype(float)))
    d = d[np.isfinite(d) & (d > 0)]
    if len(d) == 0:
        return 4.0
    return float(np.median(d))


def build_event_table(df: pd.DataFrame, *, sample_period_ms: float) -> pd.DataFrame:
    """Aggregate samples → events (legacy-compatible segmentation)."""
    need = [
        "participant_id",
        "trial_id",
        "recording_timestamp",
        "eye_movement_type",
        "eye_movement_type_index",
        "gaze_event_duration",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    work = df.copy()
    work["eye_movement_type_index"] = pd.to_numeric(
        work["eye_movement_type_index"], errors="coerce"
    ).astype("Int64")
    work = work.sort_values(
        ["participant_id", "trial_id", "recording_timestamp"], kind="mergesort"
    )
    pt = ["participant_id", "trial_id"]
    idx_shift = work.groupby(pt, sort=False, dropna=False)["eye_movement_type_index"].shift(1)
    typ_shift = work.groupby(pt, sort=False, dropna=False)["eye_movement_type"].shift(1)
    new_event = (work["eye_movement_type_index"] != idx_shift) | (
        work["eye_movement_type"] != typ_shift
    )
    new_event = new_event.astype("boolean").fillna(True)
    work["_new_event"] = new_event.astype("int8")
    work["event_id"] = work.groupby(pt, sort=False, dropna=False)["_new_event"].cumsum().astype(
        "int64"
    )

    # Validity
    if "validity_left" in work.columns and "validity_right" in work.columns:
        left_valid = work["validity_left"].astype(str) == "Valid"
        right_valid = work["validity_right"].astype(str) == "Valid"
        work["_valid_any"] = (left_valid | right_valid).astype(float)
        work["_valid_both"] = (left_valid & right_valid).astype(float)
    else:
        work["_valid_any"] = np.nan
        work["_valid_both"] = np.nan

    if "pupil_diameter_filtered" in work.columns:
        work["_pupil_missing"] = work["pupil_diameter_filtered"].isna().astype(float)

    # Scroll activity within event
    if "scroll_offset_y" in work.columns:
        work["_scroll"] = work["scroll_offset_y"].astype(float)

    g = work.groupby([*pt, "event_id"], dropna=False, sort=False)
    agg: dict[str, Any] = {
        "t_start_ms": ("recording_timestamp", "min"),
        "t_end_ms": ("recording_timestamp", "max"),
        "dur_event_ms": ("gaze_event_duration", "first"),
        "eye_movement_type_index": ("eye_movement_type_index", "first"),
        "eye_movement_type": ("eye_movement_type", "first"),
        "valid_any_rate": ("_valid_any", "mean"),
        "valid_both_rate": ("_valid_both", "mean"),
        "x_doc": ("gaze_point_x_doc", "median"),
        "y_doc": ("gaze_point_y_doc", "median"),
        "gaze_x0": ("gaze_point_x_doc", "first"),
        "gaze_y0": ("gaze_point_y_doc", "first"),
        "gaze_x1": ("gaze_point_x_doc", "last"),
        "gaze_y1": ("gaze_point_y_doc", "last"),
        "n_samples": ("recording_timestamp", "size"),
    }
    if "scroll_offset_y" in work.columns:
        agg["scroll_offset_y_med"] = ("scroll_offset_y", "median")
        agg["scroll_offset_y_first"] = ("scroll_offset_y", "first")
        agg["scroll_offset_y_last"] = ("scroll_offset_y", "last")
        agg["scroll_offset_y_min"] = ("scroll_offset_y", "min")
        agg["scroll_offset_y_max"] = ("scroll_offset_y", "max")
    if "viewport_doc_position" in work.columns:
        agg["viewport_doc_position"] = ("viewport_doc_position", "median")
    if "gaze_viewport_y" in work.columns:
        agg["gaze_viewport_y"] = ("gaze_viewport_y", "median")
    if "w_doc" in work.columns:
        agg["w_doc"] = ("w_doc", "first")
        agg["h_doc"] = ("h_doc", "first")
    if "pupil_diameter_filtered" in work.columns:
        agg["pupil_med"] = ("pupil_diameter_filtered", "median")
        agg["pupil_missing_rate"] = ("_pupil_missing", "mean")
    if "correction_applied" in work.columns:
        agg["correction_applied_any"] = (
            "correction_applied",
            lambda s: bool(pd.Series(s).fillna(False).astype(bool).any()),
        )
    if "star_chart" in work.columns:
        agg["star_chart"] = ("star_chart", "first")
    if "question_type" in work.columns:
        agg["question_type"] = ("question_type", "first")

    core = g.agg(**agg).reset_index()
    core["dur_ts_ms"] = (core["t_end_ms"] - core["t_start_ms"]) + float(sample_period_ms)
    core["duration_mismatch"] = (
        core["dur_event_ms"] - core["dur_ts_ms"]
    ).abs() > (2.0 * float(sample_period_ms))
    core["duration_ms"] = np.where(
        core["duration_mismatch"], core["dur_ts_ms"], core["dur_event_ms"]
    )

    # DOCnorm of median fixation position
    if "w_doc" in core.columns:
        core["x_docnorm"] = core["x_doc"] / core["w_doc"].astype(float)
        core["y_docnorm"] = core["y_doc"] / core["h_doc"].astype(float)
    else:
        core["x_docnorm"] = np.nan
        core["y_docnorm"] = np.nan

    # Saccade geometry in document px (then DOCnorm)
    is_sacc = core["eye_movement_type"].astype(str) == "Saccade"
    dx = (core["gaze_x1"] - core["gaze_x0"]).astype(float)
    dy = (core["gaze_y1"] - core["gaze_y0"]).astype(float)
    amp = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    speed = amp / core["duration_ms"].astype(float).replace(0, np.nan)
    core["sacc_dx"] = np.where(is_sacc, dx, np.nan)
    core["sacc_dy"] = np.where(is_sacc, dy, np.nan)
    core["sacc_amp"] = np.where(is_sacc, amp, np.nan)
    core["sacc_angle"] = np.where(is_sacc, angle, np.nan)
    core["sacc_is_regression"] = np.where(is_sacc, (dx < 0).astype(float), np.nan)
    core["sacc_speed"] = np.where(is_sacc, speed, np.nan)
    # DOCnorm saccade
    if "w_doc" in core.columns:
        core["sacc_dx_docnorm"] = np.where(is_sacc, dx / core["w_doc"], np.nan)
        core["sacc_dy_docnorm"] = np.where(is_sacc, dy / core["h_doc"], np.nan)
        core["sacc_amp_docnorm"] = np.where(
            is_sacc,
            np.sqrt((dx / core["w_doc"]) ** 2 + (dy / core["h_doc"]) ** 2),
            np.nan,
        )

    return core


def attach_prev_saccade(events: pd.DataFrame) -> pd.DataFrame:
    """Legacy prev_sacc_* attachment (ffill then shift)."""
    pt = ["participant_id", "trial_id"]
    events = events.sort_values([*pt, "t_start_ms"], kind="mergesort").reset_index(drop=True)
    is_sacc = events["eye_movement_type"].astype(str) == "Saccade"
    sacc_map = {
        "duration_ms": "prev_sacc_dur_ms",
        "sacc_dx": "prev_sacc_dx",
        "sacc_dy": "prev_sacc_dy",
        "sacc_amp": "prev_sacc_amp",
        "sacc_angle": "prev_sacc_angle",
        "sacc_is_regression": "prev_sacc_is_regression",
        "sacc_speed": "prev_sacc_speed",
        "sacc_amp_docnorm": "prev_sacc_amp_docnorm",
    }
    only_cols = []
    for src in sacc_map:
        col = f"_only_{src}"
        only_cols.append(col)
        if src in events.columns:
            events[col] = np.where(is_sacc, events[src].astype(float), np.nan)
        else:
            events[col] = np.nan
    events[only_cols] = events.groupby(pt, sort=False, dropna=False)[only_cols].ffill()
    prev = events.groupby(pt, sort=False, dropna=False)[only_cols].shift(1)
    for src, out_col in sacc_map.items():
        events[out_col] = prev[f"_only_{src}"]
    events["prev_sacc_found"] = events["prev_sacc_dur_ms"].notna().astype(int)
    events = events.drop(columns=only_cols, errors="ignore")
    return events


def add_scroll_features(fixations: pd.DataFrame) -> pd.DataFrame:
    """Per-fixation scroll features from event scroll stats + inter-fixation deltas."""
    pt = ["participant_id", "trial_id"]
    fix = fixations.sort_values([*pt, "t_start_ms"], kind="mergesort").copy()
    # Displacement since previous fixation
    prev_scroll = fix.groupby(pt, sort=False)["scroll_offset_y_med"].shift(1)
    disp = fix["scroll_offset_y_med"] - prev_scroll
    disp = disp.fillna(0.0)
    dt = (fix["t_start_ms"] - fix.groupby(pt, sort=False)["t_start_ms"].shift(1)).fillna(0.0)
    vel = np.where(dt > 0, disp / (dt / 1000.0), 0.0)

    direction = np.where(disp > 1e-6, "down", np.where(disp < -1e-6, "up", "none"))
    during = (fix["scroll_offset_y_max"] - fix["scroll_offset_y_min"]).abs() > 1e-6

    # Time since scroll onset/offset: approximate from within-episode scroll changes
    # Onset = last time scroll started changing before this fixation; offset = last settle.
    # Simple episode-level: track last onset/offset timestamps via scan.
    t_onset = []
    t_offset = []
    for _, g in fix.groupby(pt, sort=False):
        last_onset = g["t_start_ms"].iloc[0]
        last_offset = g["t_start_ms"].iloc[0]
        prev_s = None
        for row in g.itertuples(index=False):
            s = float(row.scroll_offset_y_med)
            if prev_s is not None and abs(s - prev_s) > 1e-6:
                last_onset = float(row.t_start_ms)
            if prev_s is not None and abs(s - prev_s) <= 1e-6:
                last_offset = float(row.t_start_ms)
            t = float(row.t_start_ms)
            t_onset.append(max(0.0, t - last_onset))
            t_offset.append(max(0.0, t - last_offset))
            prev_s = s

    fix["scroll_direction"] = direction
    fix["scroll_displacement_px"] = disp.to_numpy()
    fix["scroll_velocity_px_s"] = vel
    fix["scroll_during"] = during.to_numpy()
    fix["t_since_scroll_onset_ms"] = t_onset
    fix["t_since_scroll_offset_ms"] = t_offset
    if "viewport_doc_position" not in fix.columns:
        fix["viewport_doc_position"] = 0.0
    if "gaze_viewport_y" not in fix.columns:
        fix["gaze_viewport_y"] = np.nan
    fix["viewport_doc_position"] = fix["viewport_doc_position"].fillna(0.0).clip(0.0, 1.0)
    return fix


def events_to_fixation_records(
    events: pd.DataFrame,
    *,
    star_condition: str,
) -> list[dict[str, Any]]:
    events = attach_prev_saccade(events)
    fix = events[events["eye_movement_type"].astype(str) == "Fixation"].copy()
    if fix.empty:
        return []
    # Ensure scroll aggregate cols exist
    for c in (
        "scroll_offset_y_med",
        "scroll_offset_y_min",
        "scroll_offset_y_max",
        "scroll_offset_y_first",
        "scroll_offset_y_last",
    ):
        if c not in fix.columns:
            fix[c] = 0.0
    fix = add_scroll_features(fix)
    records = []
    for i, row in enumerate(fix.itertuples(index=False)):
        amp = float(row.prev_sacc_amp) if pd.notna(row.prev_sacc_amp) else 0.0
        ang = float(row.prev_sacc_angle) if pd.notna(row.prev_sacc_angle) else 0.0
        direction_deg = float(np.degrees(ang)) if pd.notna(row.prev_sacc_angle) else 0.0
        records.append(
            {
                "participant_id": str(row.participant_id),
                "trial_id": str(row.trial_id),
                "fixation_id": f"fix_{i:04d}",
                "t_start_ms": float(row.t_start_ms),
                "duration_ms": float(row.duration_ms),
                "x_doc": float(row.x_doc) if pd.notna(row.x_doc) else float("nan"),
                "y_doc": float(row.y_doc) if pd.notna(row.y_doc) else float("nan"),
                "x_docnorm": float(row.x_docnorm) if pd.notna(row.x_docnorm) else float("nan"),
                "y_docnorm": float(row.y_docnorm) if pd.notna(row.y_docnorm) else float("nan"),
                "star_condition": star_condition,
                "valid_any_rate": float(row.valid_any_rate)
                if pd.notna(row.valid_any_rate)
                else float("nan"),
                "valid_both_rate": float(row.valid_both_rate)
                if pd.notna(row.valid_both_rate)
                else float("nan"),
                "duration_mismatch": bool(row.duration_mismatch),
                "prev_sacc_found": int(row.prev_sacc_found),
                "prev_saccade": {
                    "amplitude": amp,
                    "direction_deg": direction_deg,
                },
                "scroll": {
                    "direction": str(row.scroll_direction),
                    "displacement_px": float(row.scroll_displacement_px),
                    "velocity_px_s": float(row.scroll_velocity_px_s),
                    "t_since_scroll_onset_ms": float(row.t_since_scroll_onset_ms),
                    "t_since_scroll_offset_ms": float(row.t_since_scroll_offset_ms),
                    "during_scroll": bool(row.scroll_during),
                    "viewport_doc_position": float(row.viewport_doc_position),
                    "gaze_viewport_y": float(row.gaze_viewport_y)
                    if pd.notna(row.gaze_viewport_y)
                    else 0.0,
                },
            }
        )
    return records


def build_episode_fixations(
    gaze: pd.DataFrame,
    segments: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    *,
    star_condition: str,
    epsilon: float,
    empty_space_map: dict[str, str],
    loop_templates: list[list[str]],
    max_loop_gap_events: int,
    min_valid_any: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Full P6 pipeline for one episode."""
    ts = gaze["recording_timestamp"].to_numpy(dtype=float)
    period = compute_sample_period_ms(ts)
    events = build_event_table(gaze, sample_period_ms=period)
    # Filter fixations by validity after event build (legacy applies on fixations)
    records = events_to_fixation_records(events, star_condition=star_condition)
    if min_valid_any > 0:
        records = [r for r in records if (r.get("valid_any_rate") or 0) >= min_valid_any]

    # Attach segment_role for loop refine
    seg_by_id = {s["segment_id"]: s for s in segments}
    assigned = assign_fixations(
        records,
        segments,
        panels,
        epsilon=epsilon,
        empty_space_map=empty_space_map,
    )
    for r in assigned:
        sid = r.get("segment_id")
        if sid and sid in seg_by_id:
            r["segment_role"] = seg_by_id[sid].get("segment_role")
            bools = seg_by_id[sid].get("bools") or {}
            r["is_level_descriptor"] = bool(bools.get("is_level_descriptor"))

    looped, loop_counts = annotate_loops(
        assigned,
        templates=loop_templates,
        max_loop_gap_events=max_loop_gap_events,
        star_condition=star_condition,
    )

    # QC
    n = len(looped)
    n_empty = sum(1 for r in looped if r.get("segment_id") is None)
    n_amb = sum(1 for r in looped if r.get("ambiguous"))
    n_edge = sum(1 for r in looped if r.get("edge_zone"))
    confs = [float(r.get("assignment_confidence") or 0) for r in looped]
    qc = {
        "n_fixations": n,
        "n_empty_space": n_empty,
        "pct_empty_space": 100.0 * n_empty / n if n else 0.0,
        "pct_ambiguous": 100.0 * n_amb / n if n else 0.0,
        "pct_edge_zone": 100.0 * n_edge / n if n else 0.0,
        "mean_confidence": float(np.mean(confs)) if confs else 0.0,
        "loop_counts": loop_counts,
        "sample_period_ms": period,
    }
    return looped, qc


def _image_stem(trial_id: str, star_condition: str) -> str:
    if star_condition == "star_on":
        return f"{trial_id}S"
    if star_condition == "star_off":
        return f"{trial_id}NS"
    return trial_id


def run_p6(repo_root: Optional[Path] = None, *, max_participants: Optional[int] = None) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    data_cfg = OmegaConf.load(repo_root / "configs" / "data.yaml")
    pre_cfg = OmegaConf.load(repo_root / "configs" / "preprocessing.yaml")
    data_version = str(data_cfg.data_version)

    # Derive ε first
    eps_summary = derive_epsilon(repo_root, write_config_comment=True)
    # Reload config after write
    pre_cfg = OmegaConf.load(repo_root / "configs" / "preprocessing.yaml")
    epsilon = float(pre_cfg.gaze_assignment.epsilon_doc_px)
    scales = list(pre_cfg.gaze_assignment.sensitivity_scales)
    empty_map = dict(pre_cfg.empty_space_category_map)
    templates = [list(t) for t in pre_cfg.loops.templates]
    max_gap = int(pre_cfg.loops.max_loop_gap_events)
    min_valid = float(pre_cfg.event_aggregation.min_valid_any)

    processed = repo_root / str(data_cfg.paths.processed_root) / data_version
    gaze_dir = processed / "gaze_coords"
    meta_dir = processed / "metadata"
    out_dir = processed / "fixations"
    out_dir.mkdir(parents=True, exist_ok=True)

    star_tbl = pd.read_parquet(processed / "registry" / "star_conditions.parquet")
    star_map = {
        (str(r.participant_id), str(r.trial_id)): str(r.star_condition)
        for r in star_tbl.itertuples(index=False)
    }

    qc_rows = []
    sens_rows = []
    corpus_loops: dict[str, int] = {}
    errors = []
    n_participants = 0
    n_episodes = 0

    paths = sorted(gaze_dir.glob("p*.parquet"))
    if max_participants is not None:
        paths = paths[: max_participants]

    for path in paths:
        num = path.stem[1:] if path.stem.lower().startswith("p") else path.stem
        pid = f"P{int(num):02d}" if str(num).isdigit() else path.stem.upper()
        df = pd.read_parquet(path)
        n_participants += 1
        for trial_id, ep in df.groupby("trial_id", sort=True):
            tid = str(trial_id)
            sc = star_map.get((pid, tid), "not_eligible")
            stem = _image_stem(tid, sc)
            try:
                segments = uio.read_json(meta_dir / f"{stem}__segments.json")
                panels = uio.read_json(meta_dir / f"{stem}__panels.json")
            except FileNotFoundError as e:
                errors.append(f"{pid}/{tid}: {e}")
                continue

            fixations, qc = build_episode_fixations(
                ep,
                segments,
                panels,
                star_condition=sc,
                epsilon=epsilon,
                empty_space_map=empty_map,
                loop_templates=templates,
                max_loop_gap_events=max_gap,
                min_valid_any=min_valid,
            )
            # ε sensitivity: re-assign only (same fixation geometry)
            base_ids = [f.get("segment_id") for f in fixations]
            sens = {"participant_id": pid, "trial_id": tid, "star_condition": sc}
            base_pts = [
                {"x_doc": f["x_doc"], "y_doc": f["y_doc"], "fixation_id": f["fixation_id"]}
                for f in fixations
            ]
            for scale in scales:
                alt = assign_fixations(
                    base_pts,
                    segments,
                    panels,
                    epsilon=epsilon * float(scale),
                    empty_space_map=empty_map,
                )
                alt_ids = [f.get("segment_id") for f in alt]
                m = min(len(base_ids), len(alt_ids))
                changed = sum(1 for i in range(m) if base_ids[i] != alt_ids[i])
                sens[f"pct_changed_x{scale}"] = 100.0 * changed / m if m else 0.0
            sens_rows.append(sens)

            for tid_key, c in (qc.get("loop_counts") or {}).items():
                corpus_loops[tid_key] = corpus_loops.get(tid_key, 0) + int(c)

            # Flatten for parquet
            flat_rows = []
            for f in fixations:
                flat = {k: v for k, v in f.items() if k not in ("scroll", "prev_saccade")}
                for sk, sv in (f.get("scroll") or {}).items():
                    flat[f"scroll_{sk}"] = sv
                for sk, sv in (f.get("prev_saccade") or {}).items():
                    flat[f"prev_saccade_{sk}"] = sv
                flat["data_version"] = data_version
                flat_rows.append(flat)

            ep_dir = out_dir / pid
            ep_dir.mkdir(parents=True, exist_ok=True)
            out_path = ep_dir / f"{tid}__{sc}.parquet"
            pd.DataFrame(flat_rows).to_parquet(out_path, index=False)

            # Validate a sample against schema (nested form)
            if flat_rows:
                sample = fixations[0]
                try:
                    nested = {
                        "participant_id": sample["participant_id"],
                        "trial_id": sample["trial_id"],
                        "fixation_id": sample["fixation_id"],
                        "t_start_ms": sample["t_start_ms"],
                        "duration_ms": sample["duration_ms"],
                        "segment_id": sample.get("segment_id"),
                        "empty_space_category": sample.get("empty_space_category"),
                        "panel_label": sample["panel_label"],
                        "assignment_confidence": sample["assignment_confidence"],
                        "scroll": sample["scroll"],
                        "prev_saccade": sample["prev_saccade"],
                        "ambiguous": bool(sample.get("ambiguous", False)),
                        "segment_id_alt": sample.get("segment_id_alt"),
                        "star_condition": sample.get("star_condition"),
                    }
                    uio.validate(nested, "fixation")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"schema {pid}/{tid}: {e}")

            qc_row = {
                "participant_id": pid,
                "trial_id": tid,
                "star_condition": sc,
                **{k: v for k, v in qc.items() if k != "loop_counts"},
            }
            qc_rows.append(qc_row)
            n_episodes += 1

    qc_df = pd.DataFrame(qc_rows)
    if len(qc_df):
        qc_df.to_parquet(out_dir / "episode_qc.parquet", index=False)
    sens_df = pd.DataFrame(sens_rows)
    if len(sens_df):
        sens_df.to_parquet(out_dir / "epsilon_sensitivity.parquet", index=False)

    min_tmpl = int(pre_cfg.loops.min_template_corpus_count)
    dropped = {k: v for k, v in corpus_loops.items() if v < min_tmpl}
    if dropped:
        uio.write_json(out_dir / "dropped_loop_templates.json", dropped)

    summary = {
        "data_version": data_version,
        "out_dir": str(out_dir),
        "n_participants": n_participants,
        "n_episodes": n_episodes,
        "epsilon_doc_px": epsilon,
        "epsilon_derivation": eps_summary,
        "corpus_loop_counts": corpus_loops,
        "dropped_loop_templates": dropped,
        "mean_pct_empty": float(qc_df["pct_empty_space"].mean()) if len(qc_df) else 0.0,
        "mean_pct_ambiguous": float(qc_df["pct_ambiguous"].mean()) if len(qc_df) else 0.0,
        "mean_confidence": float(qc_df["mean_confidence"].mean()) if len(qc_df) else 0.0,
        "sensitivity_mean_pct_changed": {
            f"x{s}": float(sens_df[f"pct_changed_x{s}"].mean()) if len(sens_df) else 0.0
            for s in scales
        },
        "errors": errors[:50],
        "n_errors": len(errors),
        "ok": n_episodes == int(data_cfg.expected.n_participants) * int(data_cfg.expected.n_trials)
        and len(errors) == 0,
    }
    uio.write_json(out_dir / "p6_summary.json", summary)
    return summary
