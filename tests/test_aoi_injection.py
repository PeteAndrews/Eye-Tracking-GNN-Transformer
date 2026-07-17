"""Unit tests for P3 AOI hit injection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from pathlib import Path

from src.data.aoi_injection import inject_episode, write_gaze_table

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def inj_cfg() -> dict:
    cfg = OmegaConf.load(ROOT / "configs" / "preprocessing.yaml")
    return OmegaConf.to_container(cfg.aoi_injection, resolve=True)


def _panels() -> list[dict]:
    return [
        {
            "aoi_id": "star",
            "aoi_type": "star_chart",
            "x_min": 100,
            "y_min": 100,
            "x_max": 200,
            "y_max": 200,
            "area": 10000,
        },
        {
            "aoi_id": "comm",
            "aoi_type": "commentary",
            "x_min": 50,
            "y_min": 50,
            "x_max": 250,
            "y_max": 250,
            "area": 40000,
        },
        {
            "aoi_id": "asb",
            "aoi_type": "answer_scroll_bar",
            "x_min": 10,
            "y_min": 10,
            "x_max": 20,
            "y_max": 100,
            "area": 900,
        },
        {
            "aoi_id": "gui",
            "aoi_type": "general_ui",
            "x_min": 0,
            "y_min": 0,
            "x_max": 50,
            "y_max": 50,
            "area": 2500,
        },
        # Nested: scroll bar inside general_ui — smaller wins for label
        {
            "aoi_id": "asb_nested",
            "aoi_type": "answer_scroll_bar",
            "x_min": 5,
            "y_min": 5,
            "x_max": 15,
            "y_max": 15,
            "area": 100,
        },
    ]


def _row(x, y, label="Outside", **kw) -> dict:
    base = {
        "gaze_point_x_doc": x,
        "gaze_point_y_doc": y,
        "aoi_label": label,
        "aoi__advance": 0,
        "aoi__commentary": 1 if label == "Commentary" else 0,
        "aoi__green_answer_box": 0,
        "aoi__grey_answer_box": 0,
        "aoi__mark_scheme": 0,
        "aoi__question": 0,
        "aoi__response": 0,
        "star_chart": kw.get("star_chart", 1),
        "participant_id": "P01",
        "trial_id": "T21",
        "recording_timestamp": 0,
    }
    base.update(kw)
    return base


def test_star_inside_overrides_commentary(inj_cfg: dict) -> None:
    df = pd.DataFrame([_row(150, 150, "Commentary")])
    out, qc = inject_episode(df, _panels(), star_condition="star_on", cfg=inj_cfg)
    assert out["aoi_label"].iloc[0] == "Star_Chart"
    assert int(out["aoi__star_chart"].iloc[0]) == 1
    assert int(out["aoi__commentary"].iloc[0]) == 0
    assert qc["n_star_hits"] == 1
    assert qc["n_star_relabel"] == 1


def test_star_boundary_excluded(inj_cfg: dict) -> None:
    # On the edge of star box — strict inside → no hit
    df = pd.DataFrame([_row(100, 150, "Commentary")])
    out, qc = inject_episode(df, _panels(), star_condition="star_on", cfg=inj_cfg)
    assert int(out["aoi__star_chart"].iloc[0]) == 0
    assert out["aoi_label"].iloc[0] == "Commentary"
    assert qc["n_star_hits"] == 0


def test_star_outside_untouched(inj_cfg: dict) -> None:
    df = pd.DataFrame([_row(300, 300, "Commentary")])
    out, _ = inject_episode(df, _panels(), star_condition="star_on", cfg=inj_cfg)
    assert int(out["aoi__star_chart"].iloc[0]) == 0
    assert out["aoi_label"].iloc[0] == "Commentary"
    assert int(out["aoi__commentary"].iloc[0]) == 1


def test_star_off_episode_no_star_rule(inj_cfg: dict) -> None:
    df = pd.DataFrame([_row(150, 150, "Commentary")])
    out, qc = inject_episode(df, _panels(), star_condition="star_off", cfg=inj_cfg)
    assert int(out["aoi__star_chart"].iloc[0]) == 0
    assert out["aoi_label"].iloc[0] == "Commentary"
    assert qc["n_star_hits"] == 0


def test_ui_hit_outside_gets_label(inj_cfg: dict) -> None:
    df = pd.DataFrame([_row(15, 50, "Outside")])
    out, qc = inject_episode(df, _panels(), star_condition="not_eligible", cfg=inj_cfg)
    assert int(out["aoi__answer_scroll_bar"].iloc[0]) == 1
    assert out["aoi_label"].iloc[0] == "Answer_Scroll_Bar"
    assert qc["n_hit_aoi__answer_scroll_bar"] == 1


def test_ui_never_overrides_content(inj_cfg: dict) -> None:
    df = pd.DataFrame([_row(15, 50, "Question")])
    out, _ = inject_episode(df, _panels(), star_condition="not_eligible", cfg=inj_cfg)
    assert int(out["aoi__answer_scroll_bar"].iloc[0]) == 1  # one-hot still set
    assert out["aoi_label"].iloc[0] == "Question"  # label not overwritten


def test_ui_smaller_region_wins_label(inj_cfg: dict) -> None:
    # Point inside nested scroll bar AND general_ui → smaller scroll bar label
    df = pd.DataFrame([_row(10, 10, "Outside")])
    out, _ = inject_episode(df, _panels(), star_condition="not_eligible", cfg=inj_cfg)
    assert int(out["aoi__answer_scroll_bar"].iloc[0]) == 1
    assert int(out["aoi__general_ui"].iloc[0]) == 1
    assert out["aoi_label"].iloc[0] == "Answer_Scroll_Bar"


def test_star_zeros_ui_onehots(inj_cfg: dict) -> None:
    # Point in star only — UI should stay 0; if somehow both, star zeros UI
    df = pd.DataFrame([_row(150, 150, "Outside")])
    out, _ = inject_episode(df, _panels(), star_condition="star_on", cfg=inj_cfg)
    assert out["aoi_label"].iloc[0] == "Star_Chart"
    assert int(out["aoi__star_chart"].iloc[0]) == 1


def test_all_new_columns_present(inj_cfg: dict) -> None:
    df = pd.DataFrame([_row(0, 0, "Outside")])
    out, _ = inject_episode(df, _panels(), star_condition="not_eligible", cfg=inj_cfg)
    for col in (
        "aoi__star_chart",
        "aoi__answer_scroll_bar",
        "aoi__commentary_scroll_bar",
        "aoi__general_ui",
    ):
        assert col in out.columns


def test_write_gaze_table_writes_parquet_and_tsv(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "participant_id": ["P01", "P01"],
            "trial_id": ["T01", "T01"],
            "aoi_label": ["Outside", "Question"],
            "aoi__star_chart": [0, 0],
        }
    )
    stem = tmp_path / "p01"
    write_gaze_table(stem, df)
    assert stem.with_suffix(".parquet").is_file()
    tsv = stem.with_suffix(".tsv")
    assert tsv.is_file()
    text = tsv.read_text(encoding="utf-8")
    assert "aoi__star_chart" in text
    assert "\t" in text.splitlines()[0]
    reloaded = pd.read_csv(tsv, sep="\t", encoding="utf-8")
    assert list(reloaded.columns) == list(df.columns)
    assert len(reloaded) == 2
