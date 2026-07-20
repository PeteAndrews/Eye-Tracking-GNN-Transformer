"""Flag star-conditional segments and resolve document dimensions for graph build."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def flag_star_conditional(
    segments: Sequence[dict[str, Any]],
    patterns: Iterable[str],
) -> list[dict[str, Any]]:
    """Return copies with is_star_conditional=true when text matches allowlist."""
    pats = [_norm(p) for p in patterns if str(p).strip()]
    out: list[dict[str, Any]] = []
    for s in segments:
        s2 = dict(s)
        text = _norm(str(s.get("corrected_text") or ""))
        s2["is_star_conditional"] = bool(pats) and any(p in text for p in pats)
        out.append(s2)
    return out


def dims_for_stem(
    dim_rows: Sequence[dict[str, Any]],
    *,
    trial_id: str,
    star_condition: str,
    stem: str,
) -> tuple[float, float]:
    for r in dim_rows:
        if str(r.get("stem")) == stem:
            return float(r["W_doc"]), float(r["H_doc"])
    for r in dim_rows:
        if str(r.get("trial_id")) == trial_id and str(r.get("star_condition")) == star_condition:
            return float(r["W_doc"]), float(r["H_doc"])
    return 1920.0, 1080.0
