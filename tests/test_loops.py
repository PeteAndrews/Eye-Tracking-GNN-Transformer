"""Tests for P6 loop detector."""

from __future__ import annotations

from src.data.loops import annotate_loops, annotate_visits_and_returns


TEMPLATES = [
    ["response", "mark_scheme", "response"],
    ["question", "response", "question"],
    ["response", "star_chart", "response"],
]


def _fix(panel: str, t: float, sid: str | None = None) -> dict:
    return {
        "panel_label": panel,
        "t_start_ms": t,
        "segment_id": sid or f"{panel}_{t}",
        "empty_space_category": None,
    }


def test_visit_and_return_gap() -> None:
    seq = [
        _fix("response", 0, "r1"),
        _fix("mark_scheme", 100, "m1"),
        _fix("response", 200, "r1"),
    ]
    out = annotate_visits_and_returns(seq, max_loop_gap_events=20)
    assert out[0]["visit_count"] == 1
    assert out[0]["is_return"] is False
    assert out[2]["visit_count"] == 2
    assert out[2]["is_return"] is True
    assert out[2]["return_gap_events"] == 2
    assert out[2]["short_loop_return"] is True


def test_return_beyond_max_gap_not_short() -> None:
    seq = [_fix("response", i * 10, "r1") for i in range(25)]
    # revisit after inserting many others
    seq = [_fix("response", 0, "r1")]
    for i in range(1, 22):
        seq.append(_fix("mark_scheme", i * 10, f"m{i}"))
    seq.append(_fix("response", 220, "r1"))
    out = annotate_visits_and_returns(seq, max_loop_gap_events=20)
    assert out[-1]["is_return"] is True
    assert out[-1]["return_gap_events"] == 22
    assert out[-1]["short_loop_return"] is False


def test_detect_response_mark_scheme_loop() -> None:
    seq = [
        _fix("response", 0),
        _fix("mark_scheme", 100),
        _fix("response", 200),
    ]
    out, counts = annotate_loops(
        seq, templates=TEMPLATES, max_loop_gap_events=20, star_condition="not_eligible"
    )
    tid = "response→mark_scheme→response"
    assert counts[tid] >= 1
    assert out[0]["loop_role"] == "origin"
    assert out[1]["loop_role"] == "pivot"
    assert out[2]["loop_role"] == "closure"
    assert tid in out[0]["loop_template_id"]


def test_star_template_only_on_star_on() -> None:
    seq = [
        _fix("response", 0),
        _fix("star_chart", 100),
        _fix("response", 200),
    ]
    _, c_off = annotate_loops(
        seq, templates=TEMPLATES, max_loop_gap_events=20, star_condition="star_off"
    )
    _, c_on = annotate_loops(
        seq, templates=TEMPLATES, max_loop_gap_events=20, star_condition="star_on"
    )
    tid = "response→star_chart→response"
    assert c_off.get(tid, 0) == 0
    assert c_on.get(tid, 0) >= 1


def test_overlapping_templates() -> None:
    seq = [
        _fix("question", 0),
        _fix("response", 50),
        _fix("mark_scheme", 100),
        _fix("response", 150),
        _fix("question", 200),
    ]
    out, counts = annotate_loops(
        seq, templates=TEMPLATES, max_loop_gap_events=20, star_condition="not_eligible"
    )
    assert counts["response→mark_scheme→response"] >= 1
    assert counts["question→response→question"] >= 1
    # response at idx 1 may participate in both
    assert "→" in out[1]["loop_template_id"] or out[1]["loop_role"] != "none"
