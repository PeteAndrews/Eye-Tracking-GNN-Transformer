"""Schema and fixture loading tests for M0."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import io as uio

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
TRIALS = FIXTURES / "trials"


@pytest.fixture(scope="module")
def schema_names() -> list[str]:
    return ["segment", "fixation", "star_conditions"]


def test_schemas_exist(schema_names: list[str]) -> None:
    for name in schema_names:
        path = uio.schema_path(name)
        assert path.is_file()
        schema = uio.load_schema(name)
        assert schema.get("title")
        assert schema.get("type") == "object"


def test_validators_construct(schema_names: list[str]) -> None:
    for name in schema_names:
        validator = uio.get_validator(name)
        assert validator is not None


def test_fixture_trials_exist() -> None:
    trials = sorted(p.name for p in TRIALS.iterdir() if p.is_dir())
    assert trials == ["fx01_T99", "fx02_T98_star_on"]


@pytest.mark.parametrize("trial_dir", ["fx01_T99", "fx02_T98_star_on"])
def test_segments_validate(trial_dir: str) -> None:
    segs = uio.read_json(TRIALS / trial_dir / "segments.json")
    assert isinstance(segs, list)
    assert 8 <= len(segs) <= 12
    for seg in segs:
        uio.validate(seg, "segment")
    ids = [s["segment_id"] for s in segs]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("trial_dir", ["fx01_T99", "fx02_T98_star_on"])
def test_fixations_validate(trial_dir: str) -> None:
    fixations = uio.read_json(TRIALS / trial_dir / "fixations.json")
    assert isinstance(fixations, list)
    assert len(fixations) == 40
    empty = 0
    for fix in fixations:
        uio.validate(fix, "fixation")
        if fix["segment_id"] is None:
            empty += 1
            assert fix["empty_space_category"] is not None
        else:
            assert fix["empty_space_category"] is None
    assert empty >= 1


def test_star_conditions_validate() -> None:
    table = uio.read_json(FIXTURES / "star_conditions.json")
    uio.validate(table, "star_conditions")
    conditions = {a["trial_id"]: a["star_condition"] for a in table["assignments"]}
    assert conditions["T99"] == "not_eligible"
    assert conditions["T98"] == "star_on"


def test_fx01_multi_relation_pair_documented() -> None:
    edges = uio.read_json(TRIALS / "fx01_T99" / "expected_edges.json")
    pairs = edges["multi_relation_pairs"]
    assert len(pairs) >= 1
    pair = pairs[0]
    assert set(pair["relations"]) >= {"SPATIAL_NEIGHBOUR", "SEMANTIC_CANDIDATE"}
    assert [pair["source"], pair["target"]] in edges["SPATIAL_NEIGHBOUR"]
    assert [pair["source"], pair["target"]] in edges["SEMANTIC_CANDIDATE"]


def test_fx01_expected_edge_types_present() -> None:
    edges = uio.read_json(TRIALS / "fx01_T99" / "expected_edges.json")
    for key in (
        "NEXT_SEGMENT",
        "PREVIOUS_SEGMENT",
        "BELONGS_TO",
        "SPATIAL_NEIGHBOUR",
        "SEMANTIC_CANDIDATE",
    ):
        assert key in edges
        assert len(edges[key]) >= 1


def test_invalid_segment_rejected() -> None:
    bad = {
        "segment_id": "x",
        "trial_id": "NOT_A_TRIAL",
        "question_id": "q",
        "panel_label": "response",
        "corrected_text": "hi",
        "segment_type": "sentence",
        "segment_role": "response_text",
        "level_band": None,
        "mark_point_id": None,
        "star_id": None,
        "bools": {},
        "formatting": {"bold": False, "italic": False, "formatted_prop": 0.0},
        "geometry": {"x": 0, "y": 0, "w": 1, "h": 1, "n_boxes": 1, "n_lines": 1},
        "segment_order": 0,
    }
    assert not uio.is_valid(bad, "segment")


def test_utf8_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "café.json"
    data = {"note": "curly quote “and” en–dash"}
    uio.write_json(path, data)
    assert uio.read_json(path) == data
