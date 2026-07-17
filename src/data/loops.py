"""P6 visit/return + loop template detector (single source for D2 / biases).

Deterministic, config-driven. Operates on an ordered fixation list that already
has ``panel_label`` (and optionally ``segment_role`` for level-descriptor refine).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional


def _panel_state(fix: dict[str, Any], *, refine_level_descriptor: bool) -> str:
    panel = str(fix.get("panel_label") or "")
    if refine_level_descriptor and panel == "mark_scheme":
        role = str(fix.get("segment_role") or "")
        if role == "level_descriptor" or fix.get("is_level_descriptor"):
            return "mark_scheme_level_descriptor"
    return panel


def annotate_visits_and_returns(
    fixations: list[dict[str, Any]],
    *,
    max_loop_gap_events: int = 20,
) -> list[dict[str, Any]]:
    """Add visit_count, time_since_prev_visit_ms, is_return, return_gap_events/ms."""
    last_idx: dict[str, int] = {}
    last_t: dict[str, float] = {}
    visit_count: dict[str, int] = defaultdict(int)
    out = []
    for i, f in enumerate(fixations):
        key = f.get("segment_id") or f"empty:{f.get('empty_space_category')}:{f.get('panel_label')}"
        key = str(key)
        visit_count[key] += 1
        prev_i = last_idx.get(key)
        prev_t = last_t.get(key)
        t = float(f.get("t_start_ms") or 0.0)
        if prev_i is None:
            is_return = False
            gap_events = None
            gap_ms = None
            time_since = None
        else:
            gap_events = i - prev_i
            gap_ms = t - float(prev_t)
            time_since = gap_ms
            is_return = gap_events >= 1
        row = {
            **f,
            "visit_count": int(visit_count[key]),
            "time_since_prev_visit_ms": time_since,
            "is_return": bool(is_return),
            "return_gap_events": gap_events,
            "return_gap_ms": gap_ms,
            "short_loop_return": bool(
                is_return and gap_events is not None and gap_events <= max_loop_gap_events
            ),
        }
        out.append(row)
        last_idx[key] = i
        last_t[key] = t
    return out


def detect_loops(
    fixations: list[dict[str, Any]],
    templates: list[list[str]],
    *,
    max_loop_gap_events: int = 20,
    star_condition: str = "not_eligible",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Annotate loop_role / loop_template_ids / loop_origin_index.

    Templates are panel-level A→B→A sequences. Star templates containing
    ``star_chart`` are evaluated only when star_condition == star_on.
    Overlapping matches are all recorded (multi-hot template ids).
    """
    n = len(fixations)
    roles = ["none"] * n
    origins = [None] * n
    # template_id -> list of fix indices that participate
    template_hits: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    # (origin, pivot, closure) triples per template

    states = [_panel_state(f, refine_level_descriptor=True) for f in fixations]

    for tmpl in templates:
        if len(tmpl) != 3:
            continue
        a, b, a2 = tmpl
        assert a == a2 or True  # A→B→A form; third should equal first
        tid = "→".join(tmpl)
        if "star_chart" in tmpl and star_condition != "star_on":
            continue
        for i in range(n):
            if states[i] != a:
                continue
            # find pivot B within gap
            for j in range(i + 1, min(n, i + 1 + max_loop_gap_events + 1)):
                if states[j] != b:
                    continue
                for k in range(j + 1, min(n, j + 1 + max_loop_gap_events + 1)):
                    if states[k] != a:
                        continue
                    if (k - i) > max_loop_gap_events:
                        break
                    template_hits[tid].append((i, j, k))
                    break
                else:
                    continue
                break

    # Apply annotations (later matches can overwrite role with priority closure>pivot>origin
    # but we keep first origin index; multi-hot templates as pipe-joined ids)
    tmpl_ids: list[set[str]] = [set() for _ in range(n)]
    counts: dict[str, int] = { "→".join(t): 0 for t in templates }

    for tid, triples in template_hits.items():
        counts[tid] = len(triples)
        for i, j, k in triples:
            tmpl_ids[i].add(tid)
            tmpl_ids[j].add(tid)
            tmpl_ids[k].add(tid)
            if roles[i] == "none":
                roles[i] = "origin"
                origins[i] = i
            if roles[j] in ("none", "origin"):
                roles[j] = "pivot"
                origins[j] = i
            roles[k] = "closure"
            origins[k] = i

    out = []
    for idx, f in enumerate(fixations):
        ids = sorted(tmpl_ids[idx])
        out.append(
            {
                **f,
                "loop_role": roles[idx],
                "loop_template_id": "|".join(ids) if ids else "",
                "loop_origin_index": origins[idx],
            }
        )
    return out, counts


def annotate_loops(
    fixations: list[dict[str, Any]],
    *,
    templates: list[list[str]],
    max_loop_gap_events: int = 20,
    star_condition: str = "not_eligible",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    visited = annotate_visits_and_returns(
        fixations, max_loop_gap_events=max_loop_gap_events
    )
    return detect_loops(
        visited,
        templates,
        max_loop_gap_events=max_loop_gap_events,
        star_condition=star_condition,
    )
