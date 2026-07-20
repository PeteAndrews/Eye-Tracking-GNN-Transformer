"""Unit tests for M2 encoder selection (ranking metric + pair I/O). No HF downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.text.encoder_selection import (
    NEGATIVE_EASY,
    NEGATIVE_HARD,
    infer_negative_type,
    load_pair_table,
    ranking_accuracy,
    reviewed_triples,
    sync_negative_types,
    validate_pairs_for_promote,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "encoder_eval_pairs_fixture.csv"


def test_ranking_accuracy_perfect():
    a = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    r = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    assert ranking_accuracy(a, r, u) == 1.0


def test_ranking_accuracy_fails_when_unrelated_closer():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    r = np.array([[0.0, 1.0]], dtype=np.float32)
    u = np.array([[1.0, 0.0]], dtype=np.float32)
    assert ranking_accuracy(a, r, u) == 0.0


def test_load_fixture_pairs():
    df = load_pair_table(FIXTURE)
    assert len(df) == 6
    kept = reviewed_triples(df)
    assert len(kept) == 6
    assert set(kept["negative_type"]) == {NEGATIVE_HARD, NEGATIVE_EASY}
    assert (kept["negative_type"] == NEGATIVE_HARD).sum() == 3
    assert (kept["negative_type"] == NEGATIVE_EASY).sum() == 3


def test_same_trial_unrelated_is_hard_not_error():
    row = pd.Series(
        {
            "pair_id": "x",
            "anchor_trial": "T11",
            "unrelated_trial": "T11",
            "negative_type": NEGATIVE_HARD,
            "anchor_text": "a",
            "related_text": "b",
            "unrelated_text": "c",
            "reviewed": True,
        }
    )
    assert infer_negative_type(row) == NEGATIVE_HARD
    df = pd.DataFrame([row])
    assert validate_pairs_for_promote(df) == []


def test_sync_negative_type_after_owner_edit():
    # Owner converts easy → hard by editing unrelated_trial; label should follow trials
    df = pd.DataFrame(
        [
            {
                "pair_id": "y",
                "category": "response_mark_scheme",
                "negative_type": NEGATIVE_EASY,
                "anchor_trial": "T11",
                "unrelated_trial": "T11",
                "anchor_text": "a",
                "related_text": "b",
                "unrelated_text": "c",
                "reviewed": True,
            }
        ]
    )
    synced = sync_negative_types(df)
    assert synced.iloc[0]["negative_type"] == NEGATIVE_HARD


def test_promote_does_not_require_draft_mix():
    # All-hard reviewed set is acceptable (no mix quota)
    df = pd.DataFrame(
        [
            {
                "pair_id": f"p{i}",
                "category": "response_mark_scheme",
                "negative_type": NEGATIVE_HARD,
                "anchor_trial": "T11",
                "unrelated_trial": "T11",
                "anchor_text": f"anchor {i}",
                "related_text": f"related {i}",
                "unrelated_text": f"unrelated {i}",
                "reviewed": True,
            }
            for i in range(3)
        ]
    )
    assert validate_pairs_for_promote(df) == []


def test_rubric_and_level_excluded_from_content_ms():
    from src.text.encoder_selection import is_content_mark_scheme, is_rubric_instruction

    rubric = {
        "panel_label": "mark_scheme",
        "corrected_text": "any two from",
        "bools": {},
        "segment_role": "bullet_point",
    }
    level = {
        "panel_label": "mark_scheme",
        "corrected_text": "Level 2: clear explanation with some evidence.",
        "bools": {"is_level_descriptor": True},
        "segment_role": "level_descriptor",
    }
    content = {
        "panel_label": "mark_scheme",
        "corrected_text": "iodine solution tests for starch",
        "bools": {"is_mark_scheme_point": True},
        "segment_role": "bullet_point",
    }
    assert is_rubric_instruction(rubric)
    assert not is_content_mark_scheme(rubric)
    assert not is_content_mark_scheme(level)
    assert is_content_mark_scheme(content)


def test_validate_rejects_anchor_eq_related():
    df = pd.DataFrame(
        [
            {
                "pair_id": "bad",
                "category": "commentary_paraphrase",
                "negative_type": NEGATIVE_HARD,
                "anchor_trial": "T11",
                "unrelated_trial": "T11",
                "anchor_text": "same text here",
                "related_text": "same text here",
                "unrelated_text": "other",
                "reviewed": True,
            }
        ]
    )
    errs = validate_pairs_for_promote(df)
    assert any("anchor_text == related_text" in e for e in errs)


def test_bakeoff_blocked_without_reviewed(tmp_path: Path):
    from src.text.encoder_selection import run_bakeoff

    (tmp_path / "configs").mkdir()
    cfg = (ROOT / "configs" / "encoder_selection.yaml").read_text(encoding="utf-8")
    (tmp_path / "configs" / "encoder_selection.yaml").write_text(cfg, encoding="utf-8")
    summary = run_bakeoff(tmp_path, require_reviewed=True)
    assert summary["ok"] is False
    assert summary["blocked"] is True
