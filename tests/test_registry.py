"""Unit tests for P0 identity parsing and NS↔S correspondence."""

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


def _aoi(aid, atype, y=0):
    return {
        "aoi_id": aid,
        "aoi_type": atype,
        "x_min": 0,
        "y_min": y,
        "x_max": 10,
        "y_max": y + 10,
    }


def _seg(sid, aoi_type, text, order, aoi_id="a1"):
    return {
        "segment_id": sid,
        "segment_type": "sentence",
        "aoi_type": aoi_type,
        "aoi_id": aoi_id,
        "corrected_text": text,
        "segment_order": order,
        "box_ids": [],
    }


def test_variant_correspondence_ok_with_geometry_drift(tmp_path: Path) -> None:
    s = {
        "aoi_annotations": [_aoi("a1", "response", 0), _aoi("star", "star_chart", 50)],
        "text_boxes": [],
        "segments": [
            _seg("s1", "response", "hello", 0),
            _seg("st", "star_chart", "star text", 0, aoi_id="star"),
        ],
    }
    s["segments"][1]["segment_type"] = "star_concept"
    ns = {
        "aoi_annotations": [_aoi("a1", "response", 2)],  # geometry drift
        "text_boxes": [],
        "segments": [_seg("s1b", "response", "hello", 0)],
    }
    (tmp_path / "T11S-complete.json").write_text(json.dumps(s), encoding="utf-8")
    (tmp_path / "T11NS-complete.json").write_text(json.dumps(ns), encoding="utf-8")
    report = check_variant_consistency(tmp_path, ["T11"])
    assert report[0]["ok"] is True
    assert report[0]["hard_fail"] is False


def test_variant_aoi_type_mismatch_hard(tmp_path: Path) -> None:
    s = {
        "aoi_annotations": [_aoi("a1", "response")],
        "text_boxes": [],
        "segments": [_seg("s1", "response", "A", 0)],
    }
    ns = {
        "aoi_annotations": [_aoi("a1", "question")],
        "text_boxes": [],
        "segments": [_seg("s1", "question", "A", 0)],
    }
    (tmp_path / "T12S.json").write_text(json.dumps(s), encoding="utf-8")
    (tmp_path / "T12NS.json").write_text(json.dumps(ns), encoding="utf-8")
    report = check_variant_consistency(tmp_path, ["T12"])
    assert report[0]["ok"] is False
    assert report[0]["aoi_type_multiset_ok"] is False


def test_star_conditional_s_only_allowlisted(tmp_path: Path) -> None:
    s = {
        "aoi_annotations": [_aoi("a1", "response"), _aoi("a2", "commentary")],
        "text_boxes": [],
        "segments": [
            _seg("s1", "response", "shared", 0),
            _seg("s2", "commentary", "In each case the requirement(s) to enter a level are indicated", 0, "a2"),
        ],
    }
    ns = {
        "aoi_annotations": [_aoi("a1", "response"), _aoi("a2", "commentary")],
        "text_boxes": [],
        "segments": [_seg("s1", "response", "shared", 0)],
    }
    (tmp_path / "T21S.json").write_text(json.dumps(s), encoding="utf-8")
    (tmp_path / "T21NS.json").write_text(json.dumps(ns), encoding="utf-8")
    report = check_variant_consistency(tmp_path, ["T21"])
    assert report[0]["ok"] is True
    assert report[0]["star_conditional_s_only"]
    assert report[0]["star_conditional_s_only"][0]["is_star_conditional"] is True


def test_unexpected_s_only_fails(tmp_path: Path) -> None:
    s = {
        "aoi_annotations": [_aoi("a1", "response")],
        "text_boxes": [],
        "segments": [
            _seg("s1", "response", "shared", 0),
            _seg("s2", "response", "totally unique response text", 1),
        ],
    }
    ns = {
        "aoi_annotations": [_aoi("a1", "response")],
        "text_boxes": [],
        "segments": [_seg("s1", "response", "shared", 0)],
    }
    (tmp_path / "T13S.json").write_text(json.dumps(s), encoding="utf-8")
    (tmp_path / "T13NS.json").write_text(json.dumps(ns), encoding="utf-8")
    report = check_variant_consistency(tmp_path, ["T13"])
    assert report[0]["ok"] is False
    assert report[0]["unexpected_s_only"]
