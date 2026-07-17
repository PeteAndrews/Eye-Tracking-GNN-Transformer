"""Tests for P6 gaze→segment assignment."""

from __future__ import annotations

from src.data.gaze_assignment import assign_point

EPS = 10.0
EMPTY_MAP = {
    "question": "question_background",
    "commentary": "commentary_background",
    "star_chart": "star_chart_background",
    "answer_scroll_bar": "answer_scroll_bar",
    "general_ui": "ui_general",
    "ui_general": "ui_general",
}

SEGMENTS = [
    {
        "segment_id": "a",
        "panel_label": "question",
        "geometry": {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100},
    },
    {
        "segment_id": "b",
        "panel_label": "response",
        "geometry": {"x_min": 80, "y_min": 80, "x_max": 180, "y_max": 180},
    },
]

PANELS = [
    {
        "aoi_id": "pq",
        "aoi_type": "question",
        "panel_label": "question",
        "empty_space_key": "question",
        "x_min": 0,
        "y_min": 0,
        "x_max": 200,
        "y_max": 200,
        "area": 40000,
    },
    {
        "aoi_id": "ps",
        "aoi_type": "star_chart",
        "panel_label": "star_chart",
        "empty_space_key": "star_chart",
        "x_min": 50,
        "y_min": 50,
        "x_max": 90,
        "y_max": 90,
        "area": 1600,
    },
]


def test_interior_high_confidence() -> None:
    r = assign_point(50, 50, SEGMENTS[:1], PANELS, epsilon=EPS, empty_space_map=EMPTY_MAP)
    assert r["segment_id"] == "a"
    assert r["assignment_confidence"] == 1.0
    assert r["ambiguous"] is False
    assert r["empty_space_category"] is None


def test_edge_zone_decays_confidence() -> None:
    # 5 px from left edge inside box; ε=10 → conf=0.5
    r = assign_point(5, 50, SEGMENTS[:1], PANELS, epsilon=EPS, empty_space_map=EMPTY_MAP)
    assert r["segment_id"] == "a"
    assert r["edge_zone"] is True
    assert abs(r["assignment_confidence"] - 0.5) < 1e-6


def test_overlap_ambiguous() -> None:
    # Inside both overlapping boxes
    r = assign_point(90, 90, SEGMENTS, PANELS, epsilon=EPS, empty_space_map=EMPTY_MAP)
    assert r["ambiguous"] is True
    assert r["segment_id"] in {"a", "b"}
    assert r["segment_id_alt"] in {"a", "b"}
    assert r["segment_id"] != r["segment_id_alt"]


def test_outside_within_epsilon() -> None:
    # Just left of box a: x=-5, within ε=10
    r = assign_point(-5, 50, SEGMENTS[:1], PANELS, epsilon=EPS, empty_space_map=EMPTY_MAP)
    assert r["segment_id"] == "a"
    assert abs(r["assignment_confidence"] - 0.5) < 1e-6
    assert r["edge_zone"] is True


def test_empty_space_smaller_panel_wins() -> None:
    # Point in star nested in question, beyond segment boxes
    r = assign_point(70, 70, [], PANELS, epsilon=EPS, empty_space_map=EMPTY_MAP)
    assert r["segment_id"] is None
    assert r["empty_space_category"] == "star_chart_background"
    assert r["panel_label"] == "star_chart"
    assert r["assignment_confidence"] == 0.0


def test_outside_document() -> None:
    r = assign_point(1000, 1000, [], PANELS, epsilon=EPS, empty_space_map=EMPTY_MAP)
    assert r["empty_space_category"] == "outside_document"
    assert r["panel_label"] == "outside_document"
