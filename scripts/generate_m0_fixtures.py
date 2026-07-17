"""Generate M0 synthetic fixture trials (run once during M0; fixtures are committed).

Creates two trials (~10 segments, ~40 fixations each) that exercise:
- NEXT_SEGMENT / PREVIOUS_SEGMENT (ordered within panels)
- BELONGS_TO (segments → panel nodes)
- SPATIAL_NEIGHBOUR (nearby same-panel boxes)
- SEMANTIC_CANDIDATE (cross-panel related text)
- a multi-relation pair: SPATIAL_NEIGHBOUR ∧ SEMANTIC_CANDIDATE
  (response seg_r2 and mark_scheme seg_ms1 are both near in a shared
  cross-panel neighbourhood definition for later graph tests, and share
  high semantic overlap in text — expected_edges.json records both)
- empty-space fixations
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures"


def _bools(**kwargs):
    base = {
        "command_word": False,
        "domain_term": False,
        "is_bullet_point": False,
        "is_level_descriptor": False,
        "is_mark_scheme_point": False,
        "is_commentary": False,
        "is_star_chart": False,
        "requires_calculation": False,
        "contains_data_reference": False,
        "contains_allow_instruction": False,
        "contains_reject_instruction": False,
        "contains_comparison": False,
    }
    base.update(kwargs)
    return base


def _fmt(bold=False, italic=False, prop=0.0):
    return {"bold": bold, "italic": italic, "formatted_prop": prop}


def _geom(x, y, w, h, n_boxes=1, n_lines=1):
    """x,y are top-left of the box; schema stores centre + AABB corners."""
    return {
        "x": x + w / 2.0,
        "y": y + h / 2.0,
        "w": w,
        "h": h,
        "x_min": float(x),
        "y_min": float(y),
        "x_max": float(x + w),
        "y_max": float(y + h),
        "n_boxes": n_boxes,
        "n_lines": n_lines,
    }


def _seg(sid, trial, qid, panel, text, stype, role, order, geom, **kw):
    return {
        "segment_id": sid,
        "trial_id": trial,
        "question_id": qid,
        "panel_label": panel,
        "corrected_text": text,
        "segment_type": stype,
        "segment_role": role,
        "level_band": kw.pop("level_band", None),
        "mark_point_id": kw.pop("mark_point_id", None),
        "star_id": kw.pop("star_id", None),
        "bools": kw.pop("bools", _bools()),
        "formatting": kw.pop("formatting", _fmt()),
        "geometry": geom,
        "segment_order": order,
    }


def _scroll(direction="none", disp=0.0, vel=0.0, onset=0, offset=0, during=False, vp=0.0, gy=0.5):
    return {
        "direction": direction,
        "displacement_px": disp,
        "velocity_px_s": vel,
        "t_since_scroll_onset_ms": onset,
        "t_since_scroll_offset_ms": offset,
        "during_scroll": during,
        "viewport_doc_position": vp,
        "gaze_viewport_y": gy,
    }


def _sacc(amp=0.0, deg=0.0):
    return {"amplitude": amp, "direction_deg": deg}


def _fix(
    pid,
    trial,
    fid,
    t0,
    dur,
    panel,
    conf,
    *,
    segment_id=None,
    empty=None,
    scroll=None,
    sacc=None,
    ambiguous=False,
    alt=None,
    star_condition="not_eligible",
):
    return {
        "participant_id": pid,
        "trial_id": trial,
        "fixation_id": fid,
        "t_start_ms": t0,
        "duration_ms": dur,
        "segment_id": segment_id,
        "empty_space_category": empty,
        "panel_label": panel,
        "assignment_confidence": conf,
        "scroll": scroll or _scroll(),
        "prev_saccade": sacc or _sacc(),
        "ambiguous": ambiguous,
        "segment_id_alt": alt,
        "star_condition": star_condition,
    }


def build_fx01():
    """Non-star trial T99 — exercises all non-star edge types + empty space."""
    trial, qid = "T99", "Q_FIXTURE_01"
    segs = [
        _seg("seg_q1", trial, qid, "question", "Explain the process of photosynthesis.", "sentence", "question_stem", 0, _geom(40, 40, 400, 40)),
        _seg("seg_q2", trial, qid, "question", "Include the role of chlorophyll.", "sentence", "question_stem", 1, _geom(40, 90, 400, 40), bools=_bools(command_word=True)),
        _seg("seg_r1", trial, qid, "response", "Plants convert light energy into chemical energy.", "sentence", "response_text", 0, _geom(40, 200, 420, 50)),
        # Multi-relation partner A: spatially near ms1 in y-overlap zone for cross-panel spatial tests;
        # text overlaps "chlorophyll absorbs light" with seg_ms1 → SEMANTIC_CANDIDATE.
        _seg("seg_r2", trial, qid, "response", "Chlorophyll absorbs light to drive the reaction.", "sentence", "response_text", 1, _geom(40, 260, 420, 50), bools=_bools(domain_term=True)),
        _seg("seg_r3", trial, qid, "response", "Oxygen is released as a by-product.", "sentence", "response_text", 2, _geom(40, 320, 420, 50)),
        _seg(
            "seg_ms1",
            trial,
            qid,
            "mark_scheme",
            "Chlorophyll absorbs light energy.",
            "mark_scheme_point",
            "answers",
            0,
            _geom(500, 200, 380, 40),
            mark_point_id="mp_01",
            bools=_bools(is_bullet_point=True, is_mark_scheme_point=True, domain_term=True),
        ),
        _seg(
            "seg_ms2",
            trial,
            qid,
            "mark_scheme",
            "Chemical energy stored in glucose.",
            "mark_scheme_point",
            "answers",
            1,
            _geom(500, 250, 380, 40),
            mark_point_id="mp_02",
            bools=_bools(is_bullet_point=True, is_mark_scheme_point=True),
        ),
        _seg(
            "seg_ld1",
            trial,
            qid,
            "mark_scheme",
            "Level 2: clear explanation of energy conversion.",
            "level_descriptor",
            "level_descriptor",
            2,
            _geom(500, 320, 380, 50),
            level_band="L2",
            bools=_bools(is_level_descriptor=True),
        ),
        _seg(
            "seg_c1",
            trial,
            qid,
            "commentary",
            "Accept equivalent wording for energy conversion.",
            "commentary_guidance",
            "commentary",
            0,
            _geom(40, 450, 840, 40),
            bools=_bools(is_commentary=True, contains_allow_instruction=True),
        ),
        _seg(
            "seg_c2",
            trial,
            qid,
            "commentary",
            "Do not credit vague references to sunlight alone.",
            "commentary_guidance",
            "commentary",
            1,
            _geom(40, 500, 840, 40),
            bools=_bools(is_commentary=True, contains_reject_instruction=True),
        ),
    ]

    # ~40 fixations: reading path + returns + empty-space
    path = [
        ("seg_q1", "question", 1.0),
        ("seg_q2", "question", 1.0),
        ("seg_r1", "response", 1.0),
        ("seg_r2", "response", 0.95),
        ("seg_ms1", "mark_scheme", 1.0),
        ("seg_r2", "response", 0.9),  # return / loop closure
        ("seg_ms2", "mark_scheme", 1.0),
        ("seg_r3", "response", 1.0),
        ("seg_ld1", "mark_scheme", 0.85),
        ("seg_c1", "commentary", 1.0),
        ("seg_r1", "response", 0.9),
        ("seg_c2", "commentary", 1.0),
        ("seg_ms1", "mark_scheme", 1.0),
        ("seg_r2", "response", 1.0),
        ("seg_q1", "question", 0.8),
    ]
    fixations = []
    t = 0.0
    for i in range(40):
        if i in (7, 22, 35):
            # empty-space: panel background or outside
            if i == 7:
                fixations.append(
                    _fix("P_FX", trial, f"fix_{i:03d}", t, 120, "response", 0.0, empty="response_background")
                )
            elif i == 22:
                fixations.append(
                    _fix("P_FX", trial, f"fix_{i:03d}", t, 100, "mark_scheme", 0.0, empty="mark_scheme_background")
                )
            else:
                fixations.append(
                    _fix("P_FX", trial, f"fix_{i:03d}", t, 80, "outside_document", 0.0, empty="outside_document")
                )
        else:
            sid, panel, conf = path[i % len(path)]
            amb = i == 18
            alt = "seg_ms2" if amb else None
            conf_i = 0.55 if amb else conf
            scroll = _scroll(direction="down", disp=12.0, vel=40.0, during=True, vp=0.2, gy=0.4) if i == 10 else _scroll()
            sacc = _sacc(amp=80.0 + (i % 5) * 10, deg=90.0 if i % 2 == 0 else -90.0)
            fixations.append(
                _fix(
                    "P_FX",
                    trial,
                    f"fix_{i:03d}",
                    t,
                    150 + (i % 7) * 10,
                    panel,
                    conf_i,
                    segment_id=sid,
                    scroll=scroll,
                    sacc=sacc,
                    ambiguous=amb,
                    alt=alt,
                )
            )
        t += 180.0

    expected_edges = {
        "NEXT_SEGMENT": [
            ["seg_q1", "seg_q2"],
            ["seg_r1", "seg_r2"],
            ["seg_r2", "seg_r3"],
            ["seg_ms1", "seg_ms2"],
            ["seg_ms2", "seg_ld1"],
            ["seg_c1", "seg_c2"],
        ],
        "PREVIOUS_SEGMENT": [
            ["seg_q2", "seg_q1"],
            ["seg_r2", "seg_r1"],
            ["seg_r3", "seg_r2"],
            ["seg_ms2", "seg_ms1"],
            ["seg_ld1", "seg_ms2"],
            ["seg_c2", "seg_c1"],
        ],
        "BELONGS_TO": [
            ["seg_q1", "panel_question"],
            ["seg_q2", "panel_question"],
            ["seg_r1", "panel_response"],
            ["seg_r2", "panel_response"],
            ["seg_r3", "panel_response"],
            ["seg_ms1", "panel_mark_scheme"],
            ["seg_ms2", "panel_mark_scheme"],
            ["seg_ld1", "panel_mark_scheme"],
            ["seg_c1", "panel_commentary"],
            ["seg_c2", "panel_commentary"],
        ],
        "SPATIAL_NEIGHBOUR": [
            ["seg_r1", "seg_r2"],
            ["seg_r2", "seg_r3"],
            ["seg_ms1", "seg_ms2"],
            ["seg_ms2", "seg_ld1"],
            # Multi-relation pair also listed under SEMANTIC_CANDIDATE:
            ["seg_r2", "seg_ms1"],
        ],
        "SEMANTIC_CANDIDATE": [
            ["seg_r2", "seg_ms1"],  # multi-relation with SPATIAL_NEIGHBOUR
            ["seg_r1", "seg_ms2"],
            ["seg_r3", "seg_c1"],
        ],
        "multi_relation_pairs": [
            {
                "source": "seg_r2",
                "target": "seg_ms1",
                "relations": ["SPATIAL_NEIGHBOUR", "SEMANTIC_CANDIDATE"],
            }
        ],
    }
    return segs, fixations, expected_edges


def build_fx02():
    """Star-on trial T98 — includes star_chart segment + star loop path."""
    trial, qid = "T98", "Q_FIXTURE_02"
    segs = [
        _seg("seg_q1", trial, qid, "question", "Describe cellular respiration stages.", "sentence", "question_stem", 0, _geom(40, 40, 400, 40)),
        _seg("seg_r1", trial, qid, "response", "Glycolysis breaks down glucose in the cytoplasm.", "sentence", "response_text", 0, _geom(40, 200, 420, 50)),
        _seg("seg_r2", trial, qid, "response", "The Krebs cycle releases carbon dioxide.", "sentence", "response_text", 1, _geom(40, 260, 420, 50)),
        _seg(
            "seg_ms1",
            trial,
            qid,
            "mark_scheme",
            "Glycolysis occurs in the cytoplasm.",
            "mark_scheme_point",
            "answers",
            0,
            _geom(500, 200, 380, 40),
            mark_point_id="mp_01",
            bools=_bools(is_bullet_point=True, is_mark_scheme_point=True),
        ),
        _seg(
            "seg_ms2",
            trial,
            qid,
            "mark_scheme",
            "Krebs cycle produces CO2.",
            "mark_scheme_point",
            "answers",
            1,
            _geom(500, 250, 380, 40),
            mark_point_id="mp_02",
            bools=_bools(is_bullet_point=True, is_mark_scheme_point=True),
        ),
        _seg(
            "seg_c1",
            trial,
            qid,
            "commentary",
            "Credit correct organelle references.",
            "commentary_guidance",
            "commentary",
            0,
            _geom(40, 450, 840, 40),
            bools=_bools(is_commentary=True),
        ),
        _seg(
            "seg_st1",
            trial,
            qid,
            "star_chart",
            "Energy transfer",
            "star_concept",
            "star_concept",
            0,
            _geom(900, 200, 160, 40),
            star_id="star_01",
            bools=_bools(is_star_chart=True),
        ),
        _seg(
            "seg_st2",
            trial,
            qid,
            "star_chart",
            "Gas exchange",
            "star_concept",
            "star_concept",
            1,
            _geom(900, 250, 160, 40),
            star_id="star_02",
            bools=_bools(is_star_chart=True),
        ),
        _seg("seg_q2", trial, qid, "question", "Link each stage to its products.", "sentence", "question_stem", 1, _geom(40, 90, 400, 40)),
        _seg(
            "seg_ld1",
            trial,
            qid,
            "mark_scheme",
            "Level 3: stages linked to products accurately.",
            "level_descriptor",
            "level_descriptor",
            2,
            _geom(500, 320, 380, 50),
            level_band="L3",
            bools=_bools(is_level_descriptor=True),
        ),
    ]

    path = [
        ("seg_q1", "question", 1.0),
        ("seg_r1", "response", 1.0),
        ("seg_ms1", "mark_scheme", 1.0),
        ("seg_r1", "response", 0.9),
        ("seg_st1", "star_chart", 1.0),
        ("seg_r2", "response", 1.0),
        ("seg_st2", "star_chart", 0.95),
        ("seg_r2", "response", 0.9),
        ("seg_ms2", "mark_scheme", 1.0),
        ("seg_c1", "commentary", 1.0),
        ("seg_ld1", "mark_scheme", 0.85),
        ("seg_q2", "question", 1.0),
    ]
    fixations = []
    t = 0.0
    for i in range(40):
        if i in (5, 28):
            empty = "star_chart_background" if i == 5 else "commentary_background"
            panel = "star_chart" if i == 5 else "commentary"
            fixations.append(
                _fix(
                    "P_FX",
                    trial,
                    f"fix_{i:03d}",
                    t,
                    110,
                    panel,
                    0.0,
                    empty=empty,
                    star_condition="star_on",
                )
            )
        else:
            sid, panel, conf = path[i % len(path)]
            fixations.append(
                _fix(
                    "P_FX",
                    trial,
                    f"fix_{i:03d}",
                    t,
                    140 + (i % 6) * 8,
                    panel,
                    conf,
                    segment_id=sid,
                    sacc=_sacc(amp=50 + i % 40, deg=(i * 37) % 360),
                    star_condition="star_on",
                )
            )
        t += 170.0

    expected_edges = {
        "NEXT_SEGMENT": [
            ["seg_q1", "seg_q2"],
            ["seg_r1", "seg_r2"],
            ["seg_ms1", "seg_ms2"],
            ["seg_ms2", "seg_ld1"],
            ["seg_st1", "seg_st2"],
        ],
        "SEMANTIC_CANDIDATE": [
            ["seg_r1", "seg_ms1"],
            ["seg_r2", "seg_ms2"],
            ["seg_r1", "seg_st1"],
            ["seg_r2", "seg_st2"],
        ],
        "multi_relation_pairs": [
            {
                "source": "seg_r1",
                "target": "seg_ms1",
                "relations": ["SPATIAL_NEIGHBOUR", "SEMANTIC_CANDIDATE"],
            }
        ],
    }
    return segs, fixations, expected_edges


def main():
    star_table = {
        "schema_version": "v0_m0",
        "eligible_trials": ["T98"],
        "assignments": [
            {"participant_id": "P_FX", "trial_id": "T99", "star_condition": "not_eligible"},
            {"participant_id": "P_FX", "trial_id": "T98", "star_condition": "star_on"},
        ],
    }

    for name, builder in (("fx01_T99", build_fx01), ("fx02_T98_star_on", build_fx02)):
        segs, fixations, edges = builder()
        d = OUT / "trials" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "segments.json").write_text(json.dumps(segs, indent=2) + "\n", encoding="utf-8")
        (d / "fixations.json").write_text(json.dumps(fixations, indent=2) + "\n", encoding="utf-8")
        (d / "expected_edges.json").write_text(json.dumps(edges, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {len(segs)} segments, {len(fixations)} fixations")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "star_conditions.json").write_text(json.dumps(star_table, indent=2) + "\n", encoding="utf-8")
    print("wrote fixtures/star_conditions.json")


if __name__ == "__main__":
    main()
