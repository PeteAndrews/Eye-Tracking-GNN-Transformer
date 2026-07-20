"""Tests for P6 fixation event aggregation (legacy-compatible segmentation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.aoi_injection import write_gaze_table
from src.data.fixations import attach_prev_saccade, build_event_table, compute_sample_period_ms


def _samples() -> pd.DataFrame:
    # Two fixations separated by a saccade; run-length on (index, type)
    rows = []
    t = 0.0
    # Fixation index 1 — 3 samples
    for _ in range(3):
        rows.append(
            {
                "participant_id": "P01",
                "trial_id": "T01",
                "recording_timestamp": t,
                "eye_movement_type": "Fixation",
                "eye_movement_type_index": 1,
                "gaze_event_duration": 120.0,
                "gaze_point_x_doc": 100.0,
                "gaze_point_y_doc": 200.0,
                "validity_left": "Valid",
                "validity_right": "Valid",
                "scroll_offset_y": 0.0,
                "w_doc": 1920.0,
                "h_doc": 1080.0,
                "viewport_doc_position": 0.0,
                "gaze_viewport_y": 0.2,
            }
        )
        t += 4.0
    # Saccade index 2 — 2 samples
    for i in range(2):
        rows.append(
            {
                "participant_id": "P01",
                "trial_id": "T01",
                "recording_timestamp": t,
                "eye_movement_type": "Saccade",
                "eye_movement_type_index": 2,
                "gaze_event_duration": 40.0,
                "gaze_point_x_doc": 100.0 + 50 * i,
                "gaze_point_y_doc": 200.0,
                "validity_left": "Valid",
                "validity_right": "Valid",
                "scroll_offset_y": 0.0,
                "w_doc": 1920.0,
                "h_doc": 1080.0,
                "viewport_doc_position": 0.0,
                "gaze_viewport_y": 0.2,
            }
        )
        t += 4.0
    # Fixation index 3
    for _ in range(2):
        rows.append(
            {
                "participant_id": "P01",
                "trial_id": "T01",
                "recording_timestamp": t,
                "eye_movement_type": "Fixation",
                "eye_movement_type_index": 3,
                "gaze_event_duration": 100.0,
                "gaze_point_x_doc": 150.0,
                "gaze_point_y_doc": 200.0,
                "validity_left": "Valid",
                "validity_right": "Valid",
                "scroll_offset_y": 10.0,
                "w_doc": 1920.0,
                "h_doc": 1080.0,
                "viewport_doc_position": 0.05,
                "gaze_viewport_y": 0.18,
            }
        )
        t += 4.0
    return pd.DataFrame(rows)


def test_sample_period() -> None:
    assert abs(compute_sample_period_ms(np.array([0.0, 4.0, 8.0, 12.0])) - 4.0) < 1e-9


def test_event_segmentation_three_events() -> None:
    ev = build_event_table(_samples(), sample_period_ms=4.0)
    assert len(ev) == 3
    types = list(ev["eye_movement_type"])
    assert types == ["Fixation", "Saccade", "Fixation"]


def test_duration_reconciliation_fields() -> None:
    ev = build_event_table(_samples(), sample_period_ms=4.0)
    assert "dur_ts_ms" in ev.columns
    assert "duration_mismatch" in ev.columns
    assert "duration_ms" in ev.columns


def test_prev_saccade_attached_to_second_fixation() -> None:
    ev = build_event_table(_samples(), sample_period_ms=4.0)
    ev = attach_prev_saccade(ev)
    fix = ev[ev.eye_movement_type == "Fixation"]
    assert int(fix.iloc[0]["prev_sacc_found"]) == 0
    assert int(fix.iloc[1]["prev_sacc_found"]) == 1
    assert fix.iloc[1]["prev_sacc_amp"] > 0


def test_legacy_comparability_event_count_on_type_index_runs() -> None:
    """Same run-length rule as legacy: count events from (index, type) changes."""
    df = _samples()
    # Manual legacy-style count
    n = 0
    prev = None
    for row in df.itertuples(index=False):
        key = (row.eye_movement_type_index, row.eye_movement_type)
        if key != prev:
            n += 1
            prev = key
    ev = build_event_table(df, sample_period_ms=4.0)
    assert len(ev) == n


def test_write_gaze_table_used_for_fixations_export(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "fixation_id": ["fix_0000", "fix_0001"],
            "trial_id": ["T01", "T01"],
            "x_doc": [10.0, 20.0],
        }
    )
    stem = tmp_path / "T01__not_eligible"
    write_gaze_table(stem, df)
    assert stem.with_suffix(".parquet").is_file()
    tsv = stem.with_suffix(".tsv")
    assert tsv.is_file()
    text = tsv.read_text(encoding="utf-8")
    assert "fixation_id" in text
    assert "\t" in text.splitlines()[0]
