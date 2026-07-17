"""Unit tests for P2 metadata compilation (geometry, panel map, fallbacks)."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.data.segments import (
    compile_variant,
    derive_segment_role,
    map_panel_label,
    resolve_aoi_spatially,
    union_boxes,
)
from src.utils import io as uio

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def panel_map() -> dict[str, str]:
    cfg = OmegaConf.load(ROOT / "configs" / "preprocessing.yaml")
    return dict(cfg.canonical_panel_map)


def test_union_boxes_geometry() -> None:
    boxes = [
        {"box_id": "b1", "x_min": 10, "y_min": 20, "x_max": 30, "y_max": 40},
        {"box_id": "b2", "x_min": 25, "y_min": 35, "x_max": 50, "y_max": 60},
    ]
    u = union_boxes(boxes)
    assert u["x_min"] == 10
    assert u["y_min"] == 20
    assert u["x_max"] == 50
    assert u["y_max"] == 60
    assert u["w"] == 40
    assert u["h"] == 40
    assert u["x"] == 30
    assert u["y"] == 40


def test_union_boxes_requires_input() -> None:
    with pytest.raises(ValueError):
        union_boxes([])


def test_map_panel_label_content(panel_map: dict[str, str]) -> None:
    assert map_panel_label("question", panel_map) == "question"
    assert map_panel_label("mark_scheme_answers", panel_map) == "mark_scheme"
    assert map_panel_label("mark_scheme_extra_information", panel_map) == "mark_scheme"
    assert map_panel_label("level_descriptor", panel_map) == "mark_scheme"
    assert map_panel_label("commentary", panel_map) == "commentary"
    assert map_panel_label("star_chart", panel_map) == "star_chart"


def test_map_panel_label_ui_collapses_to_schema_enum(panel_map: dict[str, str]) -> None:
    assert map_panel_label("general_ui", panel_map) == "ui"
    assert map_panel_label("answer_scroll_bar", panel_map) == "ui"
    assert map_panel_label("commentary_scroll_bar", panel_map) == "ui"


def test_derive_segment_role_from_sub_aoi() -> None:
    assert derive_segment_role("mark_scheme_answers", "bullet") == "answers"
    assert derive_segment_role("mark_scheme_extra_information", "sentence") == "extra_information"
    assert derive_segment_role("level_descriptor", "sentence") == "level_descriptor"
    assert derive_segment_role("question", "sentence") == "sentence"


def test_resolve_aoi_spatially_smaller_wins() -> None:
    panels = [
        {
            "aoi_id": "big",
            "aoi_type": "commentary",
            "x_min": 0,
            "y_min": 0,
            "x_max": 100,
            "y_max": 100,
        },
        {
            "aoi_id": "small",
            "aoi_type": "star_chart",
            "x_min": 40,
            "y_min": 40,
            "x_max": 60,
            "y_max": 60,
        },
    ]
    hit = resolve_aoi_spatially({"x": 50, "y": 50}, panels)
    assert hit is not None
    assert hit["aoi_id"] == "small"
    assert resolve_aoi_spatially({"x": 200, "y": 200}, panels) is None


def _raw_minimal() -> dict:
    return {
        "text_boxes": [
            {"box_id": "b1", "x_min": 10, "y_min": 10, "x_max": 40, "y_max": 30, "line_number": 1},
            {"box_id": "b2", "x_min": 10, "y_min": 50, "x_max": 40, "y_max": 70, "line_number": 2},
            {"box_id": "b3", "x_min": 50, "y_min": 50, "x_max": 80, "y_max": 70, "line_number": 2},
            {"box_id": "orphan", "x_min": 0, "y_min": 0, "x_max": 5, "y_max": 5, "line_number": 0},
        ],
        "aoi_annotations": [
            {
                "aoi_id": "aoi_q",
                "aoi_type": "question",
                "x_min": 0,
                "y_min": 0,
                "x_max": 100,
                "y_max": 100,
            },
            {
                "aoi_id": "aoi_ms",
                "aoi_type": "mark_scheme_answers",
                "x_min": 0,
                "y_min": 200,
                "x_max": 100,
                "y_max": 300,
            },
        ],
        "segments": [],
        "star_chart_annotations": [{"should": "be_ignored"}],
    }


def test_compile_geometry_and_unclaimed(panel_map: dict[str, str]) -> None:
    raw = _raw_minimal()
    raw["segments"] = [
        {
            "segment_id": "s1",
            "aoi_id": "aoi_q",
            "aoi_type": "question",
            "box_ids": ["b1"],
            "corrected_text": "Q text",
            "segment_type": "sentence",
            "segment_role": "stem",
            "segment_order": 0,
            "is_command_word": False,
            "bold_text": False,
            "italic_text": False,
        }
    ]
    segs, panels, qc = compile_variant(
        raw,
        trial_id="T99",
        star_condition="not_eligible",
        question_id="T99",
        panel_map=panel_map,
    )
    assert qc["ok"]
    assert len(segs) == 1
    assert segs[0]["geometry"]["n_boxes"] == 1
    assert segs[0]["geometry"]["w"] == 30
    assert "orphan" in qc["unclaimed_boxes"]
    assert len(panels) == 2
    uio.validate(segs[0], "segment")


def test_fallback_segment_role(panel_map: dict[str, str]) -> None:
    raw = _raw_minimal()
    raw["segments"] = [
        {
            "segment_id": "s_ms",
            "aoi_id": "aoi_ms",
            "aoi_type": "mark_scheme_answers",
            "box_ids": ["b2"],
            "corrected_text": "Answer bullet",
            "segment_type": "bullet",
            "segment_role": "",
            "segment_order": 0,
            "level_band": "",
            "mark_point_id": "",
            "star_id": "",
            "is_command_word": False,
            "bold_text": True,
            "italic_text": False,
        }
    ]
    segs, _, qc = compile_variant(
        raw,
        trial_id="T99",
        star_condition="not_eligible",
        question_id="T99",
        panel_map=panel_map,
    )
    assert segs[0]["segment_role"] == "answers"
    assert "segment_role_derived" in segs[0]["fallbacks_applied"]
    assert segs[0]["level_band"] is None
    assert segs[0]["formatting"]["bold"] is True
    assert segs[0]["panel_label"] == "mark_scheme"
    assert qc["fallback_counts"]["segment_role_derived"] == 1
    uio.validate(segs[0], "segment")


def test_fallback_spatial_aoi_id(panel_map: dict[str, str]) -> None:
    raw = _raw_minimal()
    raw["segments"] = [
        {
            "segment_id": "s_spa",
            "aoi_id": "",
            "aoi_type": "",
            "box_ids": ["b1"],
            "corrected_text": "Inside question panel",
            "segment_type": "sentence",
            "segment_role": "stem",
            "segment_order": 0,
            "is_command_word": False,
            "bold_text": False,
            "italic_text": False,
        }
    ]
    segs, _, qc = compile_variant(
        raw,
        trial_id="T99",
        star_condition="not_eligible",
        question_id="T99",
        panel_map=panel_map,
    )
    assert segs[0]["aoi_id"] == "aoi_q"
    assert segs[0]["panel_label"] == "question"
    assert "spatial_aoi_id" in segs[0]["fallbacks_applied"]
    assert qc["fallback_counts"]["spatial_aoi_id"] == 1


def test_fallback_segment_order_tiebreak(panel_map: dict[str, str]) -> None:
    raw = _raw_minimal()
    raw["segments"] = [
        {
            "segment_id": "lower",
            "aoi_id": "aoi_q",
            "aoi_type": "question",
            "box_ids": ["b2"],
            "corrected_text": "Lower",
            "segment_type": "sentence",
            "segment_role": "a",
            "segment_order": 5,
            "is_command_word": False,
            "bold_text": False,
            "italic_text": False,
        },
        {
            "segment_id": "upper",
            "aoi_id": "aoi_q",
            "aoi_type": "question",
            "box_ids": ["b1"],
            "corrected_text": "Upper",
            "segment_type": "sentence",
            "segment_role": "b",
            "segment_order": 5,
            "is_command_word": False,
            "bold_text": False,
            "italic_text": False,
        },
        {
            "segment_id": "right",
            "aoi_id": "aoi_q",
            "aoi_type": "question",
            "box_ids": ["b3"],
            "corrected_text": "Right same row",
            "segment_type": "sentence",
            "segment_role": "c",
            "segment_order": 5,
            "is_command_word": False,
            "bold_text": False,
            "italic_text": False,
        },
    ]
    segs, _, qc = compile_variant(
        raw,
        trial_id="T99",
        star_condition="not_eligible",
        question_id="T99",
        panel_map=panel_map,
    )
    by_id = {s["segment_id"]: s for s in segs}
    # top-to-bottom then left-to-right: upper (y=10), then lower (y=50,x=10), then right (y=50,x=50)
    assert by_id["upper"]["segment_order"] < by_id["lower"]["segment_order"]
    assert by_id["lower"]["segment_order"] < by_id["right"]["segment_order"]
    assert qc["fallback_counts"]["segment_order_tiebreak"] == 3
    for s in segs:
        assert "segment_order_tiebreak" in s["fallbacks_applied"]
        uio.validate(s, "segment")


def test_star_chart_annotations_ignored(panel_map: dict[str, str]) -> None:
    raw = _raw_minimal()
    raw["segments"] = [
        {
            "segment_id": "s1",
            "aoi_id": "aoi_q",
            "aoi_type": "question",
            "box_ids": ["b1"],
            "corrected_text": "Q",
            "segment_type": "sentence",
            "segment_role": "stem",
            "segment_order": 0,
            "is_command_word": False,
            "bold_text": False,
            "italic_text": False,
        }
    ]
    segs, panels, _ = compile_variant(
        raw,
        trial_id="T99",
        star_condition="not_eligible",
        question_id="T99",
        panel_map=panel_map,
    )
    assert len(segs) == 1
    assert all(p["aoi_type"] != "should" for p in panels)


def test_multiclaimed_box_errors(panel_map: dict[str, str]) -> None:
    raw = _raw_minimal()
    raw["segments"] = [
        {
            "segment_id": "s1",
            "aoi_id": "aoi_q",
            "aoi_type": "question",
            "box_ids": ["b1"],
            "corrected_text": "A",
            "segment_type": "sentence",
            "segment_role": "a",
            "segment_order": 0,
            "is_command_word": False,
            "bold_text": False,
            "italic_text": False,
        },
        {
            "segment_id": "s2",
            "aoi_id": "aoi_q",
            "aoi_type": "question",
            "box_ids": ["b1"],
            "corrected_text": "B",
            "segment_type": "sentence",
            "segment_role": "b",
            "segment_order": 1,
            "is_command_word": False,
            "bold_text": False,
            "italic_text": False,
        },
    ]
    _, _, qc = compile_variant(
        raw,
        trial_id="T99",
        star_condition="not_eligible",
        question_id="T99",
        panel_map=panel_map,
    )
    assert not qc["ok"]
    assert any("claimed by" in e for e in qc["errors"])
