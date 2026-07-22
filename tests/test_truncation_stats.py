"""Unit tests for truncation accounting guard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.train.truncation_stats import truncation_stats_from_qc


def test_truncation_stats_from_qc(tmp_path: Path) -> None:
    rows = [
        {"participant_id": "P01", "trial_id": "T01", "star_condition": "not_eligible", "n_fixations": 100},
        {"participant_id": "P01", "trial_id": "T02", "star_condition": "star_on", "n_fixations": 400},
        {"participant_id": "P02", "trial_id": "T01", "star_condition": "not_eligible", "n_fixations": 256},
    ]
    pq = tmp_path / "episode_qc.parquet"
    pd.DataFrame(rows).to_parquet(pq)

    full = truncation_stats_from_qc(pq, max_seq_len=256)
    assert full["n_episodes"] == 3
    assert full["n_episodes_truncated"] == 1
    assert full["n_fixations_total"] == 756
    assert full["n_fixations_kept"] == 100 + 256 + 256
    assert full["n_fixations_discarded"] == 144

    subset = truncation_stats_from_qc(
        pq,
        max_seq_len=256,
        keys=[("P01", "T02", "star_on")],
    )
    assert subset["n_episodes"] == 1
    assert subset["n_episodes_truncated"] == 1
    assert subset["n_fixations_discarded"] == 144
