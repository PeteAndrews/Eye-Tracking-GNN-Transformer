"""NS↔S node correspondence (M3-C1): panel + normalised text + relative order."""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence


def normalise_text(text: str) -> str:
    t = str(text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def relative_order_key(segments: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Per-panel rank by segment_order → dense 0..n-1."""
    by_panel: dict[str, list[dict[str, Any]]] = {}
    for s in segments:
        by_panel.setdefault(str(s["panel_label"]), []).append(s)
    out: dict[str, int] = {}
    for _p, group in by_panel.items():
        ordered = sorted(group, key=lambda x: int(x.get("segment_order") or 0))
        for i, s in enumerate(ordered):
            out[s["segment_id"]] = i
    return out


def correspondence_key(seg: dict[str, Any], rel_order: dict[str, int]) -> tuple[str, str, int]:
    return (
        str(seg["panel_label"]),
        normalise_text(str(seg.get("corrected_text") or "")),
        int(rel_order[seg["segment_id"]]),
    )


def match_ns_s(
    ns_segments: Sequence[dict[str, Any]],
    s_segments: Sequence[dict[str, Any]],
    *,
    star_conditional_ids: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Match NS non-star segments 1:1 onto S; allowlisted S-only star-conditionals excluded."""
    star_conditional_ids = star_conditional_ids or {
        s["segment_id"]
        for s in s_segments
        if s.get("is_star_conditional")
    }
    ns_rel = relative_order_key(ns_segments)
    s_rel = relative_order_key(
        [s for s in s_segments if s["segment_id"] not in star_conditional_ids]
    )

    s_index: dict[tuple[str, str, int], str] = {}
    for s in s_segments:
        if s["segment_id"] in star_conditional_ids:
            continue
        s_index[correspondence_key(s, s_rel)] = s["segment_id"]

    matched: list[dict[str, str]] = []
    missing: list[str] = []
    for ns in ns_segments:
        key = correspondence_key(ns, ns_rel)
        sid = s_index.get(key)
        if sid is None:
            missing.append(ns["segment_id"])
        else:
            matched.append({"ns": ns["segment_id"], "s": sid})

    return {
        "ok": len(missing) == 0,
        "n_matched": len(matched),
        "n_missing": len(missing),
        "missing_ns_ids": missing,
        "matched": matched,
        "n_star_conditional_excluded": len(star_conditional_ids),
    }
