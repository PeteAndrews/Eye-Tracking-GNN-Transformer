"""P6 gaze→segment assignment (raw document px).

Confidence is a deterministic function of geometry (see module docstring / PLAN P6.5):

1. Strictly inside exactly one box → confidence decays from 1.0 in the interior
   toward 0 at the box edge zone (distance-to-edge / ε, clipped to [0,1]).
2. Inside multiple boxes, or within ε of ≥2 boxes → nearest centre-weighted
   candidate; ambiguous=True; runner-up in segment_id_alt; confidence scaled by
   best/(best+runner) margin.
3. Outside all boxes but within ε of one → that segment; confidence = 1 - d/ε.
4. Beyond ε of all → empty-space via panel regions (smaller area wins); confidence 0.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# Centre-weighted distance: Euclidean to box centre, with a small penalty for
# large boxes so nested/overlapping picks favour the tighter region.
def _centre_dist(x: float, y: float, box: dict[str, float]) -> float:
    cx = 0.5 * (box["x_min"] + box["x_max"])
    cy = 0.5 * (box["y_min"] + box["y_max"])
    return float(np.hypot(x - cx, y - cy))


def _strict_inside(x: float, y: float, box: dict[str, float]) -> bool:
    return box["x_min"] < x < box["x_max"] and box["y_min"] < y < box["y_max"]


def _distance_to_box(x: float, y: float, box: dict[str, float]) -> float:
    """0 if inside (including boundary); else Euclidean distance to closest point on AABB."""
    dx = max(box["x_min"] - x, 0.0, x - box["x_max"])
    dy = max(box["y_min"] - y, 0.0, y - box["y_max"])
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return float(np.hypot(dx, dy))


def _distance_to_edge_inside(x: float, y: float, box: dict[str, float]) -> float:
    """Min distance to boundary when inside; 0 on/outside boundary."""
    if not (box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"]):
        return 0.0
    return float(
        min(x - box["x_min"], box["x_max"] - x, y - box["y_min"], box["y_max"] - y)
    )


def assign_point(
    x: float,
    y: float,
    segments: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    *,
    epsilon: float,
    empty_space_map: dict[str, str],
) -> dict[str, Any]:
    """Assign one fixation (x,y) in document px."""
    if not np.isfinite(x) or not np.isfinite(y):
        return {
            "segment_id": None,
            "segment_id_alt": None,
            "empty_space_category": "outside_document",
            "panel_label": "outside_document",
            "assignment_confidence": 0.0,
            "ambiguous": False,
            "edge_zone": False,
        }

    boxes = []
    for s in segments:
        g = s.get("geometry") or {}
        boxes.append(
            {
                "segment_id": s["segment_id"],
                "panel_label": s.get("panel_label") or "ui",
                "x_min": float(g["x_min"]),
                "y_min": float(g["y_min"]),
                "x_max": float(g["x_max"]),
                "y_max": float(g["y_max"]),
            }
        )

    inside = [b for b in boxes if _strict_inside(x, y, b)]
    near = []
    for b in boxes:
        d = _distance_to_box(x, y, b)
        if d <= epsilon:
            near.append((d, _centre_dist(x, y, b), b))

    # Case 1: exactly one strict interior
    if len(inside) == 1:
        b = inside[0]
        d_edge = _distance_to_edge_inside(x, y, b)
        edge_zone = d_edge < epsilon
        conf = 1.0 if not edge_zone else float(max(0.0, min(1.0, d_edge / epsilon)))
        return {
            "segment_id": b["segment_id"],
            "segment_id_alt": None,
            "empty_space_category": None,
            "panel_label": b["panel_label"],
            "assignment_confidence": conf,
            "ambiguous": False,
            "edge_zone": edge_zone,
        }

    # Case 2: multiple interiors OR ≥2 within ε
    if len(inside) >= 2 or (len(inside) == 0 and len(near) >= 2):
        candidates = inside if len(inside) >= 2 else [t[2] for t in near]
        ranked = sorted(candidates, key=lambda b: _centre_dist(x, y, b))
        best, runner = ranked[0], ranked[1]
        d0 = _centre_dist(x, y, best)
        d1 = _centre_dist(x, y, runner)
        margin = d1 / (d0 + d1 + 1e-9)
        conf = float(max(0.0, min(1.0, margin)))
        return {
            "segment_id": best["segment_id"],
            "segment_id_alt": runner["segment_id"],
            "empty_space_category": None,
            "panel_label": best["panel_label"],
            "assignment_confidence": conf,
            "ambiguous": True,
            "edge_zone": False,
        }

    # Case 3: outside all but within ε of exactly one (or one interior already handled)
    if len(inside) == 0 and len(near) == 1:
        d, _, b = near[0]
        conf = float(max(0.0, min(1.0, 1.0 - d / epsilon)))
        return {
            "segment_id": b["segment_id"],
            "segment_id_alt": None,
            "empty_space_category": None,
            "panel_label": b["panel_label"],
            "assignment_confidence": conf,
            "ambiguous": False,
            "edge_zone": True,
        }

    # Also: one interior already returned; if inside==0 and near==0 → empty space
    # If inside==0 and somehow near handled... fall through

    # Case 4: empty-space via panels (smaller area wins)
    return _empty_space(x, y, panels, empty_space_map)


def _empty_space(
    x: float,
    y: float,
    panels: list[dict[str, Any]],
    empty_space_map: dict[str, str],
) -> dict[str, Any]:
    hits = []
    for p in panels:
        if float(p["x_min"]) <= x <= float(p["x_max"]) and float(p["y_min"]) <= y <= float(p["y_max"]):
            area = float(
                p.get("area")
                or max(0.0, float(p["x_max"]) - float(p["x_min"]))
                * max(0.0, float(p["y_max"]) - float(p["y_min"]))
            )
            hits.append((area, p))
    if not hits:
        return {
            "segment_id": None,
            "segment_id_alt": None,
            "empty_space_category": "outside_document",
            "panel_label": "outside_document",
            "assignment_confidence": 0.0,
            "ambiguous": False,
            "edge_zone": False,
        }
    hits.sort(key=lambda t: t[0])
    p = hits[0][1]
    key = str(p.get("empty_space_key") or p.get("aoi_type") or p.get("panel_label") or "")
    cat = empty_space_map.get(key) or empty_space_map.get(str(p.get("panel_label"))) or "outside_document"
    # panel_label for schema: content backgrounds → parent panel; UI cats stay fine-grained
    panel_for_schema = {
        "question_background": "question",
        "response_background": "response",
        "mark_scheme_background": "mark_scheme",
        "commentary_background": "commentary",
        "star_chart_background": "star_chart",
        "answer_scroll_bar": "answer_scroll_bar",
        "commentary_scroll_bar": "commentary_scroll_bar",
        "ui_general": "ui_general",
        "outside_document": "outside_document",
    }.get(cat, "outside_document")
    return {
        "segment_id": None,
        "segment_id_alt": None,
        "empty_space_category": cat,
        "panel_label": panel_for_schema,
        "assignment_confidence": 0.0,
        "ambiguous": False,
        "edge_zone": False,
    }


def assign_fixations(
    fixations: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    *,
    epsilon: float,
    empty_space_map: dict[str, str],
) -> list[dict[str, Any]]:
    out = []
    for f in fixations:
        a = assign_point(
            float(f["x_doc"]),
            float(f["y_doc"]),
            segments,
            panels,
            epsilon=epsilon,
            empty_space_map=empty_space_map,
        )
        merged = {**f, **a}
        out.append(merged)
    return out
