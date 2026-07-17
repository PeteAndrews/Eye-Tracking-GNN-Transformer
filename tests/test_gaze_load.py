"""Tests for P1 gaze pruning (synthetic fixture TSV)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.gaze_load import (
    KEEP_COLS,
    RENAME_MAP,
    expected_pruned_columns,
    prune_gaze_dataframe,
)


def _fixture_tsv(tmp_path: Path) -> Path:
    """Minimal TSV covering keep/drop/rename and QC paths."""
    rows = []
    base = {
        "Participant ID": "P99",
        "Recording timestamp": 1000,
        "Computer timestamp": 1,
        "Sensor": "Eye Tracker",
        "Project name": "test",
        "Eyetracker timestamp": 1,
        "Trial Raw": "T01",
        "Trial": "T01",
        "Star Chart": "0",
        "Question type": "FIB",
        "Event": "",
        "Event value": "",
        "Gaze point X": 100.0,
        "Gaze point Y": 200.0,
        "Gaze point left X": 99.0,
        "Gaze point left Y": 201.0,
        "Gaze point right X": 101.0,
        "Gaze point right Y": 199.0,
        "Gaze direction left X": 0.0,
        "Gaze direction left Y": 0.0,
        "Gaze direction left Z": -1.0,
        "Gaze direction right X": 0.0,
        "Gaze direction right Y": 0.0,
        "Gaze direction right Z": -1.0,
        "Pupil diameter left": 3.0,
        "Pupil diameter right": 3.1,
        "Pupil diameter filtered": 3.05,
        "Validity left": "Valid",
        "Validity right": "Valid",
        "Eye position left X (DACSmm)": 100.0,
        "Eye position left Y (DACSmm)": 170.0,
        "Eye position left Z (DACSmm)": 700.0,
        "Eye position right X (DACSmm)": 170.0,
        "Eye position right Y (DACSmm)": 170.0,
        "Eye position right Z (DACSmm)": 700.0,
        "Gaze point left X (DACSmm)": 50.0,
        "Gaze point left Y (DACSmm)": 100.0,
        "Gaze point right X (DACSmm)": 51.0,
        "Gaze point right Y (DACSmm)": 99.0,
        "Gaze point X (MCSnorm)": 0.5,
        "Gaze point Y (MCSnorm)": 0.5,
        "Gaze point left X (MCSnorm)": 0.5,
        "Gaze point left Y (MCSnorm)": 0.5,
        "Gaze point right X (MCSnorm)": 0.5,
        "Gaze point right Y (MCSnorm)": 0.5,
        "Eye movement type": "Fixation",
        "Gaze event duration": 200,
        "Eye movement type index": 1,
        "Fixation point X": 100.0,
        "Fixation point Y": 200.0,
        "Fixation point X (MCSnorm)": 0.5,
        "Fixation point Y (MCSnorm)": 0.5,
        "Ungrouped": "",
        "AOI_label": "Question",
        "AOI__Advance": 0,
        "AOI__Commentary": 0,
        "AOI__Green_Answer_Box": 0,
        "AOI__Grey_Answer_Box": 0,
        "AOI__Mark_Scheme": 0,
        "AOI__Question": 1,
        "AOI__Response": 0,
        "scroll_offset_y": 0.0,
        "scroll_ratio": 0.0,
        "Gaze point X (doc)": 100.0,
        "Gaze point Y (doc)": 200.0,
        "gaze_region": "",
        "correction_applied": "False",
        "calibration_key": "T01_NS",
        "scroll_correction_flag": "missing",
        "left_offset_px": 0.0,
        "left_correction_flag": "",
    }
    # 5 eye-tracker rows + 1 non-eye + 1 empty trial + 1 trial-raw disagree
    for i in range(5):
        r = dict(base)
        r["Recording timestamp"] = 1000 + i * 4
        r["Gaze point left X"] = 100.0 + i
        r["Gaze point left X (DACSmm)"] = 0.5 * (100.0 + i)  # 0.5 mm/px
        r["Gaze point left Y"] = 200.0 + i
        r["Gaze point left Y (DACSmm)"] = 0.5 * (200.0 + i)
        r["correction_applied"] = "False" if i < 2 else "True"
        rows.append(r)
    other = dict(base)
    other["Sensor"] = "Mouse"
    other["Recording timestamp"] = 2000
    rows.append(other)
    empty = dict(base)
    empty["Trial"] = ""
    empty["Recording timestamp"] = 3000
    rows.append(empty)
    disagree = dict(base)
    disagree["Trial"] = "T02"
    disagree["Trial Raw"] = "T99"
    disagree["Recording timestamp"] = 4000
    disagree["Question type"] = "OW"
    rows.append(disagree)

    path = tmp_path / "p99.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, encoding="utf-8")
    return path


def test_prune_keep_drop_rename(tmp_path: Path) -> None:
    path = _fixture_tsv(tmp_path)
    raw = pd.read_csv(path, sep="\t", encoding="utf-8")
    pruned, qc, eps = prune_gaze_dataframe(raw)

    assert list(pruned.columns) == expected_pruned_columns()
    for c in KEEP_COLS:
        assert RENAME_MAP[c] in pruned.columns
    assert "scroll_correction_flag" not in pruned.columns
    assert "Sensor" not in pruned.columns
    assert "Eye position left Z (DACSmm)" not in pruned.columns

    # Eye Tracker only; empty trial dropped → 5 + 1 disagree = 6
    assert len(pruned) == 6
    assert set(pruned["trial_id"]) == {"T01", "T02"}

    # Sorted
    assert pruned["recording_timestamp"].is_monotonic_increasing

    # ε inputs extracted
    assert eps["eye_z_median_mm"] == pytest.approx(700.0)
    assert eps["mm_per_px_x"] == pytest.approx(0.5, rel=0.05)

    # QC: correction_false counted on T01
    t01 = [r for r in qc if r["trial_id"] == "T01"][0]
    assert t01["n_correction_false"] == 2
    t02 = [r for r in qc if r["trial_id"] == "T02"][0]
    assert t02["n_trial_raw_disagree"] == 1


def test_rename_map_covers_keep_cols() -> None:
    for c in KEEP_COLS:
        assert c in RENAME_MAP
