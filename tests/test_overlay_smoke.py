"""Smoke tests for P4 Visual Gate 1 overlay checker."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.viz.overlay_check import (
    aggregate_fixations,
    compute_alignment_stats,
    select_stratified_sample,
    run_gate1_batch,
)

ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_fixations() -> None:
    gaze = pd.DataFrame(
        {
            "eye_movement_type": ["Fixation", "Fixation", "Saccade", "Fixation"],
            "eye_movement_type_index": [1, 1, 2, 3],
            "gaze_point_x_doc": [10.0, 12.0, 50.0, 20.0],
            "gaze_point_y_doc": [10.0, 14.0, 50.0, 20.0],
            "gaze_event_duration": [100.0, 100.0, 30.0, 80.0],
            "recording_timestamp": [0, 10, 20, 30],
            "aoi_label": ["Question", "Question", "Outside", "Mark_Scheme"],
        }
    )
    fix = aggregate_fixations(gaze)
    assert len(fix) == 2
    assert fix.loc[fix.eye_movement_type_index == 1, "x"].iloc[0] == 11.0


def test_alignment_stats_hand_computed() -> None:
    segments = [
        {
            "geometry": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10},
            "panel_label": "question",
        }
    ]
    panels = [{"x_min": 0, "y_min": 0, "x_max": 20, "y_max": 20}]
    gaze = pd.DataFrame(
        {
            "gaze_point_x_doc": [5.0, 15.0, 50.0],
            "gaze_point_y_doc": [5.0, 15.0, 50.0],
            "aoi_label": ["Question", "Outside", "Outside"],
        }
    )
    stats = compute_alignment_stats(gaze, segments, panels, w_doc=40, h_doc=40)
    assert stats["n_valid_xy"] == 3
    assert abs(stats["pct_inside_segment"] - 100 / 3) < 0.1
    assert abs(stats["pct_inside_panel"] - 200 / 3) < 0.1
    assert abs(stats["pct_outside_document"] - 100 / 3) < 0.1


def test_select_stratified_covers_eligible_and_min_trials() -> None:
    rows = []
    eligible = ["T11", "T12", "T13", "T21", "T27", "T30"]
    for p in range(1, 7):
        pid = f"P{p:02d}"
        for i in range(1, 31):
            tid = f"T{i:02d}"
            if tid in eligible:
                sc = "star_on" if (p + i) % 2 == 0 else "star_off"
            else:
                sc = "not_eligible"
            rows.append({"participant_id": pid, "trial_id": tid, "star_condition": sc})
    # Guarantee each eligible trial has ≥1 star_on episode
    for i, tid in enumerate(eligible):
        rows.append(
            {"participant_id": f"P{i + 1:02d}", "trial_id": tid, "star_condition": "star_on"}
        )
    star_tbl = pd.DataFrame(rows).drop_duplicates(["participant_id", "trial_id"], keep="last")
    sample = select_stratified_sample(
        star_tbl, eligible=eligible, trials_per_participant=3, seed=0
    )
    star_on = {e["trial_id"] for e in sample if e["star_condition"] == "star_on"}
    assert set(eligible).issubset(star_on)
    counts = pd.Series([e["participant_id"] for e in sample]).value_counts()
    assert counts.min() >= 3
    assert counts.shape[0] == 6


def test_smoke_builds_nonempty_report(tmp_path: Path, monkeypatch) -> None:
    # Write smoke into tmp by monkeypatching out_dir via running with repo root
    summary = run_gate1_batch(ROOT, smoke=True)
    assert summary["ok"] is True
    assert summary["checks"]["has_plotly"]
    assert summary["checks"]["has_samples_button"]
    assert summary["checks"]["has_fixations_button"]
    assert summary["checks"]["has_aoi_counts"]
    assert summary["checks"]["has_injection_qc"]
    assert summary["checks"]["has_alignment_summary"]
    html_path = Path(summary["html"])
    assert html_path.is_file()
    text = html_path.read_text(encoding="utf-8")
    assert "Gate 1" in text
    assert len(text) > 500
