"""Smoke / unit tests for Visual Gate 2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.viz.gate2_overlay import flag_qc_episodes, run_gate2_batch

ROOT = Path(__file__).resolve().parents[1]


def test_flag_qc_episodes() -> None:
    qc = pd.DataFrame(
        [
            {
                "participant_id": "P01",
                "trial_id": "T01",
                "star_condition": "not_eligible",
                "pct_empty_space": 50.0,
                "pct_ambiguous": 10.0,
                "mean_confidence": 0.5,
            },
            {
                "participant_id": "P02",
                "trial_id": "T02",
                "star_condition": "not_eligible",
                "pct_empty_space": 5.0,
                "pct_ambiguous": 5.0,
                "mean_confidence": 0.1,
            },
            {
                "participant_id": "P03",
                "trial_id": "T03",
                "star_condition": "not_eligible",
                "pct_empty_space": 5.0,
                "pct_ambiguous": 5.0,
                "mean_confidence": 0.5,
            },
        ]
    )
    flagged = flag_qc_episodes(qc)
    keys = {(f["participant_id"], f["trial_id"]) for f in flagged}
    assert ("P01", "T01") in keys
    assert ("P02", "T02") in keys
    assert ("P03", "T03") not in keys


def test_gate2_smoke() -> None:
    summary = run_gate2_batch(ROOT, smoke=True)
    assert summary["ok"] is True
    assert summary["checks"]["has_assigned"]
    assert summary["checks"]["has_empty"]
    assert summary["checks"]["has_ambiguous"]
    assert summary["checks"]["has_qc"]
    assert summary["checks"]["has_edge_hist"]
    assert summary["checks"]["has_panel_compare"]
    html = Path(summary["html"])
    assert html.is_file()
    text = html.read_text(encoding="utf-8")
    assert "Gate 2" in text
    assert len(text) > 500
