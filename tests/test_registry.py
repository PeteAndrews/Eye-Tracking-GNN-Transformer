"""Unit tests for P0 identity parsing and registry helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.registry import (
    check_variant_consistency,
    parse_filename_identity,
    _strip_star_content,
)


def test_parse_non_eligible() -> None:
    ident = parse_filename_identity("T01-complete.json")
    assert ident.trial_id == "T01"
    assert ident.star_condition == "not_eligible"
    assert ident.stem == "T01"


def test_parse_star_on() -> None:
    ident = parse_filename_identity("T21S-complete.json")
    assert ident.trial_id == "T21"
    assert ident.star_condition == "star_on"
    assert ident.stem == "T21S"


def test_parse_star_off() -> None:
    ident = parse_filename_identity("T21NS.png")
    assert ident.trial_id == "T21"
    assert ident.star_condition == "star_off"
    assert ident.stem == "T21NS"


def test_parse_rejection() -> None:
    with pytest.raises(ValueError):
        parse_filename_identity("complete-T01.json")
    with pytest.raises(ValueError):
        parse_filename_identity("trial21.json")


def test_strip_star_content_removes_star_only() -> None:
    data = {
        "aoi_annotations": [
            {"aoi_id": "a1", "aoi_type": "response", "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
            {"aoi_id": "a2", "aoi_type": "star_chart", "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1},
        ],
        "text_boxes": [
            {"box_id": "b1", "parent_region": "a1"},
            {"box_id": "b2", "parent_region": "a2"},
            {"box_id": "b3", "parent_region": "a1"},
        ],
        "segments": [
            {"segment_id": "s1", "segment_type": "sentence", "aoi_type": "response", "box_ids": ["b1"]},
            {"segment_id": "s2", "segment_type": "star_concept", "aoi_type": "star_chart", "box_ids": ["b2"]},
            {
                "segment_id": "s3",
                "segment_type": "commentary_guidance",
                "aoi_type": "star_chart",
                "box_ids": ["b3"],
            },
        ],
    }
    stripped = _strip_star_content(data)
    assert len(stripped["aoi_id_types"]) == 1
    assert stripped["aoi_id_types"][0]["aoi_type"] == "response"
    assert [s["segment_id"] for s in stripped["segments"]] == ["s1"]
    assert {b["box_id"] for b in stripped["text_boxes"]} == {"b1"}


def test_variant_consistency_identical(tmp_path: Path) -> None:
    core = {
        "aoi_annotations": [
            {"aoi_id": "a1", "aoi_type": "response", "x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        ],
        "text_boxes": [{"box_id": "b1", "parent_region": "a1", "x_min": 0}],
        "segments": [
            {
                "segment_id": "s1",
                "segment_type": "sentence",
                "aoi_type": "response",
                "aoi_id": "a1",
                "segment_order": 0,
                "corrected_text": "hello",
                "box_ids": ["b1"],
            }
        ],
        "star_chart_annotations": [],
    }
    s = {
        **core,
        "aoi_annotations": core["aoi_annotations"]
        + [{"aoi_id": "star", "aoi_type": "star_chart", "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}],
        "segments": core["segments"]
        + [
            {
                "segment_id": "st",
                "segment_type": "star_concept",
                "aoi_type": "star_chart",
                "box_ids": [],
            }
        ],
    }
    ns = dict(core)
    (tmp_path / "T11S-complete.json").write_text(json.dumps(s), encoding="utf-8")
    (tmp_path / "T11NS-complete.json").write_text(json.dumps(ns), encoding="utf-8")
    report = check_variant_consistency(tmp_path, ["T11"])
    assert len(report) == 1
    assert report[0]["ok"] is True
    assert report[0]["hard_fail"] is False


def test_variant_consistency_detects_diff(tmp_path: Path) -> None:
    """AOI id/type mismatch is a hard failure."""
    s = {
        "aoi_annotations": [
            {"aoi_id": "a1", "aoi_type": "response", "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}
        ],
        "text_boxes": [],
        "segments": [
            {
                "segment_id": "s1",
                "segment_type": "sentence",
                "aoi_type": "response",
                "corrected_text": "A",
                "segment_order": 0,
            }
        ],
    }
    ns = {
        "aoi_annotations": [
            {"aoi_id": "a2", "aoi_type": "response", "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}
        ],
        "text_boxes": [],
        "segments": [
            {
                "segment_id": "s1",
                "segment_type": "sentence",
                "aoi_type": "response",
                "corrected_text": "A",
                "segment_order": 0,
            }
        ],
    }
    (tmp_path / "T12S.json").write_text(json.dumps(s), encoding="utf-8")
    (tmp_path / "T12NS.json").write_text(json.dumps(ns), encoding="utf-8")
    report = check_variant_consistency(tmp_path, ["T12"])
    assert report[0]["ok"] is False
    assert report[0]["hard_fail"] is True
    assert "aoi_id_types" in report[0]["hard_diffs"]


def test_variant_segment_asymmetry_is_triaged(tmp_path: Path) -> None:
    s = {
        "aoi_annotations": [
            {"aoi_id": "a1", "aoi_type": "response", "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}
        ],
        "text_boxes": [],
        "segments": [
            {
                "segment_id": "s1",
                "segment_type": "sentence",
                "aoi_type": "response",
                "corrected_text": "A",
                "segment_order": 0,
            },
            {
                "segment_id": "s_extra",
                "segment_type": "commentary_guidance",
                "aoi_type": "commentary",
                "corrected_text": "star instructions",
                "segment_order": 1,
            },
        ],
    }
    ns = {
        "aoi_annotations": [
            {"aoi_id": "a1", "aoi_type": "response", "x_min": 0, "y_min": 0, "x_max": 1, "y_max": 1}
        ],
        "text_boxes": [],
        "segments": [
            {
                "segment_id": "s1",
                "segment_type": "sentence",
                "aoi_type": "response",
                "corrected_text": "A",
                "segment_order": 0,
            }
        ],
    }
    (tmp_path / "T21S.json").write_text(json.dumps(s), encoding="utf-8")
    (tmp_path / "T21NS.json").write_text(json.dumps(ns), encoding="utf-8")
    report = check_variant_consistency(tmp_path, ["T21"])
    assert report[0]["ok"] is True
    assert report[0]["hard_fail"] is False
    assert "segments" in report[0]["triage_diffs"]


def test_variant_consistency_soft_geometry_drift(tmp_path: Path) -> None:
    s = {
        "aoi_annotations": [
            {"aoi_id": "a1", "aoi_type": "response", "x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}
        ],
        "text_boxes": [{"box_id": "b1", "parent_region": "a1", "y_min": 0}],
        "segments": [
            {
                "segment_id": "s1",
                "segment_type": "sentence",
                "aoi_type": "response",
                "corrected_text": "same",
                "segment_order": 0,
                "box_ids": ["b1"],
            }
        ],
    }
    ns = {
        "aoi_annotations": [
            {"aoi_id": "a1", "aoi_type": "response", "x_min": 0, "y_min": 2, "x_max": 10, "y_max": 12}
        ],
        "text_boxes": [{"box_id": "b1", "parent_region": "a1", "y_min": 2}],
        "segments": [
            {
                "segment_id": "s1",
                "segment_type": "sentence",
                "aoi_type": "response",
                "corrected_text": "same",
                "segment_order": 0,
                "box_ids": ["b1"],
            }
        ],
    }
    (tmp_path / "T13S.json").write_text(json.dumps(s), encoding="utf-8")
    (tmp_path / "T13NS.json").write_text(json.dumps(ns), encoding="utf-8")
    report = check_variant_consistency(tmp_path, ["T13"])
    assert report[0]["ok"] is True
    assert report[0]["hard_fail"] is False
    assert "aoi_geometry" in report[0]["soft_diffs"]
