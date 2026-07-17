"""P2 — metadata compilation: segments, panel regions, fallbacks.

Loads audited trial-variant JSONs, derives segment geometry from text-box
unions, maps canonical panels, applies P2.7 fallbacks, and emits schema-
conforming segment + panel-region tables. ``star_chart_annotations`` is ignored.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from omegaconf import OmegaConf

from src.data.registry import list_metadata_files, parse_filename_identity
from src.utils import io as uio

# Sub-AOI → segment_role (P2.7)
SUB_AOI_ROLE = {
    "mark_scheme_answers": "answers",
    "mark_scheme_extra_information": "extra_information",
    "level_descriptor": "level_descriptor",
}

# Fine-grained UI panel labels → schema enum ``ui`` for segment nodes
UI_PANEL_LABELS = {"ui", "ui_general", "answer_scroll_bar", "commentary_scroll_bar"}

BOOL_FIELD_MAP = {
    "command_word": "is_command_word",
    "domain_term": "is_domain_specific_word",
    "is_bullet_point": "is_bullet_point",
    "is_level_descriptor": "is_level_descriptor",
    "is_mark_scheme_point": "is_mark_scheme_point",
    "is_commentary": "is_commentary_text",
    "is_star_chart": "is_star_chart",
    "requires_calculation": "requires_calculation",
    "contains_data_reference": "contains_data_reference",
    "contains_allow_instruction": "contains_allow_instruction",
    "contains_reject_instruction": "contains_reject_instruction",
    "contains_comparison": "contains_comparison",
}


def _nonempty(v: Any) -> bool:
    return v is not None and str(v).strip() != ""


def _as_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no", ""}:
        return False
    return default


def _null_if_empty(v: Any) -> Any:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def union_boxes(boxes: list[dict[str, Any]]) -> dict[str, float]:
    """Axis-aligned union of boxes with x_min/y_min/x_max/y_max."""
    if not boxes:
        raise ValueError("union_boxes requires at least one box")
    x_min = min(float(b["x_min"]) for b in boxes)
    y_min = min(float(b["y_min"]) for b in boxes)
    x_max = max(float(b["x_max"]) for b in boxes)
    y_max = max(float(b["y_max"]) for b in boxes)
    w = x_max - x_min
    h = y_max - y_min
    return {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "w": w,
        "h": h,
        "x": (x_min + x_max) / 2.0,
        "y": (y_min + y_max) / 2.0,
    }


def map_panel_label(aoi_type: str, panel_map: dict[str, str]) -> str:
    """Map metadata aoi_type → schema panel_label (UI collapsed to ``ui``)."""
    raw = panel_map.get(aoi_type, aoi_type)
    if raw in UI_PANEL_LABELS:
        return "ui"
    return raw


def derive_segment_role(aoi_type: str, segment_type: str) -> str:
    """P2.7: sub-AOI type where present, else segment_type."""
    if aoi_type in SUB_AOI_ROLE:
        return SUB_AOI_ROLE[aoi_type]
    return segment_type or ""


def resolve_aoi_spatially(
    geom: dict[str, float],
    panels: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Containing panel region; smaller area wins on overlap (P2.7)."""
    cx, cy = geom["x"], geom["y"]
    candidates = []
    for p in panels:
        if p["x_min"] <= cx <= p["x_max"] and p["y_min"] <= cy <= p["y_max"]:
            area = max(0.0, p["x_max"] - p["x_min"]) * max(0.0, p["y_max"] - p["y_min"])
            candidates.append((area, p))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[0][1]


def build_panel_regions(
    aois: list[dict[str, Any]],
    *,
    trial_id: str,
    star_condition: str,
    panel_map: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for a in aois:
        aoi_type = a.get("aoi_type") or ""
        rows.append(
            {
                "aoi_id": a.get("aoi_id"),
                "aoi_type": aoi_type,
                "panel_label": map_panel_label(aoi_type, panel_map),
                "empty_space_key": panel_map.get(aoi_type, aoi_type),
                "trial_id": trial_id,
                "star_condition": star_condition,
                "x_min": float(a["x_min"]),
                "y_min": float(a["y_min"]),
                "x_max": float(a["x_max"]),
                "y_max": float(a["y_max"]),
                "area": max(0.0, float(a["x_max"]) - float(a["x_min"]))
                * max(0.0, float(a["y_max"]) - float(a["y_min"])),
            }
        )
    return rows


def compile_variant(
    raw: dict[str, Any],
    *,
    trial_id: str,
    star_condition: str,
    question_id: str,
    panel_map: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compile one trial variant → (segments, panel_regions, qc)."""
    boxes_by_id = {b["box_id"]: b for b in raw.get("text_boxes", []) if b.get("box_id")}
    panels = build_panel_regions(
        raw.get("aoi_annotations", []),
        trial_id=trial_id,
        star_condition=star_condition,
        panel_map=panel_map,
    )
    aois_by_id = {p["aoi_id"]: p for p in panels if p.get("aoi_id")}

    claimed: dict[str, list[str]] = defaultdict(list)
    segments_out: list[dict[str, Any]] = []
    fallback_counts: Counter = Counter()
    errors: list[str] = []

    for s in raw.get("segments", []):
        sid = s.get("segment_id") or "?"
        fallbacks: list[str] = []
        bids = list(s.get("box_ids") or [])
        resolved_boxes = []
        for bid in bids:
            if bid not in boxes_by_id:
                errors.append(f"{sid}: unresolved box_id {bid}")
            else:
                resolved_boxes.append(boxes_by_id[bid])
                claimed[bid].append(sid)
        if not resolved_boxes:
            errors.append(f"{sid}: no resolvable boxes")
            continue

        geom_u = union_boxes(resolved_boxes)
        line_nums = {
            b.get("line_number")
            for b in resolved_boxes
            if b.get("line_number") is not None
        }
        n_lines = len(line_nums) if line_nums else len(resolved_boxes)

        aoi_id = _null_if_empty(s.get("aoi_id"))
        aoi_type = s.get("aoi_type") or ""
        if not aoi_id:
            hit = resolve_aoi_spatially(geom_u, panels)
            if hit is None:
                errors.append(f"{sid}: empty aoi_id and spatial resolve failed")
            else:
                aoi_id = hit["aoi_id"]
                aoi_type = hit["aoi_type"]
                fallbacks.append("spatial_aoi_id")
                fallback_counts["spatial_aoi_id"] += 1
        elif aoi_id in aois_by_id and not _nonempty(aoi_type):
            aoi_type = aois_by_id[aoi_id]["aoi_type"]

        panel_label = map_panel_label(aoi_type, panel_map)

        role_raw = s.get("segment_role")
        if _nonempty(role_raw):
            segment_role = str(role_raw).strip()
        else:
            segment_role = derive_segment_role(aoi_type, s.get("segment_type") or "")
            fallbacks.append("segment_role_derived")
            fallback_counts["segment_role_derived"] += 1

        text = s.get("corrected_text")
        from_ocr = False
        if not _nonempty(text):
            if _nonempty(s.get("ocr_text")):
                text = s.get("ocr_text")
                from_ocr = True
                fallbacks.append("corrected_text_from_ocr")
                fallback_counts["corrected_text_from_ocr"] += 1
            else:
                errors.append(f"{sid}: no text")
                text = ""

        order = s.get("segment_order")
        try:
            segment_order = int(order)
        except (TypeError, ValueError):
            errors.append(f"{sid}: bad segment_order {order!r}")
            segment_order = 0

        bools = {
            out_k: _as_bool(s.get(in_k))
            for out_k, in_k in BOOL_FIELD_MAP.items()
        }
        bold = _as_bool(s.get("bold_text"))
        italic = _as_bool(s.get("italic_text"))
        formatting = {
            "bold": bold,
            "italic": italic,
            "formatted_prop": float(1.0 if (bold or italic) else 0.0),
        }

        ocr_conf = s.get("ocr_confidence")
        try:
            ocr_conf_f = float(ocr_conf) if ocr_conf is not None and str(ocr_conf).strip() != "" else None
        except (TypeError, ValueError):
            ocr_conf_f = None

        segments_out.append(
            {
                "segment_id": sid,
                "trial_id": trial_id,
                "question_id": question_id,
                "star_condition": star_condition,
                "panel_label": panel_label,
                "aoi_type": aoi_type or None,
                "aoi_id": aoi_id,
                "corrected_text": str(text),
                "corrected_text_from_ocr_fallback": from_ocr,
                "segment_type": s.get("segment_type") or "unknown",
                "segment_role": segment_role,
                "level_band": _null_if_empty(s.get("level_band")),
                "mark_point_id": _null_if_empty(s.get("mark_point_id")),
                "star_id": _null_if_empty(s.get("star_id")),
                "bools": bools,
                "formatting": formatting,
                "geometry": {
                    "x": geom_u["x"],
                    "y": geom_u["y"],
                    "w": geom_u["w"],
                    "h": geom_u["h"],
                    "x_min": geom_u["x_min"],
                    "y_min": geom_u["y_min"],
                    "x_max": geom_u["x_max"],
                    "y_max": geom_u["y_max"],
                    "n_boxes": len(resolved_boxes),
                    "n_lines": int(n_lines),
                },
                "box_ids": bids,
                "segment_order": segment_order,
                "ocr_confidence": ocr_conf_f,
                "aoi_manual": s.get("aoi_manual") if isinstance(s.get("aoi_manual"), bool) else None,
                "aoi_ambiguous": s.get("aoi_ambiguous") if isinstance(s.get("aoi_ambiguous"), bool) else None,
                "fallbacks_applied": fallbacks,
            }
        )

    # Duplicate segment_order within panel → geometric tie-break (P2.7)
    by_panel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seg in segments_out:
        by_panel[seg["panel_label"]].append(seg)
    for panel, group in by_panel.items():
        orders = [g["segment_order"] for g in group]
        if len(orders) == len(set(orders)):
            continue
        tied_orders = {o for o in orders if orders.count(o) > 1}
        group.sort(
            key=lambda g: (
                g["segment_order"],
                g["geometry"]["y_min"],
                g["geometry"]["x_min"],
                g["segment_id"],
            )
        )
        for i, g in enumerate(group):
            if g["segment_order"] in tied_orders:
                g["fallbacks_applied"] = list(g["fallbacks_applied"]) + [
                    "segment_order_tiebreak"
                ]
                fallback_counts["segment_order_tiebreak"] += 1
            g["segment_order"] = i

    unclaimed = [bid for bid in boxes_by_id if bid not in claimed]
    multiclaimed = {bid: sids for bid, sids in claimed.items() if len(sids) > 1}
    for bid, sids in multiclaimed.items():
        errors.append(f"box {bid} claimed by {sids}")

    qc = {
        "trial_id": trial_id,
        "star_condition": star_condition,
        "n_segments": len(segments_out),
        "n_panels": len(panels),
        "n_unclaimed_boxes": len(unclaimed),
        "unclaimed_boxes": unclaimed,
        "fallback_counts": dict(fallback_counts),
        "errors": errors,
        "ok": len(errors) == 0,
    }
    return segments_out, panels, qc


def _write_segments_parquet(path: Path, segs: list[dict[str, Any]]) -> None:
    """Flatten nested segment fields for a tabular parquet companion."""
    rows = []
    for s in segs:
        g = s["geometry"]
        f = s["formatting"]
        row = {
            "segment_id": s["segment_id"],
            "trial_id": s["trial_id"],
            "question_id": s["question_id"],
            "star_condition": s.get("star_condition"),
            "panel_label": s["panel_label"],
            "aoi_type": s.get("aoi_type"),
            "aoi_id": s.get("aoi_id"),
            "corrected_text": s["corrected_text"],
            "segment_type": s["segment_type"],
            "segment_role": s["segment_role"],
            "level_band": s.get("level_band"),
            "mark_point_id": s.get("mark_point_id"),
            "star_id": s.get("star_id"),
            "segment_order": s["segment_order"],
            "geom_x": g["x"],
            "geom_y": g["y"],
            "geom_w": g["w"],
            "geom_h": g["h"],
            "geom_x_min": g["x_min"],
            "geom_y_min": g["y_min"],
            "geom_x_max": g["x_max"],
            "geom_y_max": g["y_max"],
            "n_boxes": g["n_boxes"],
            "n_lines": g["n_lines"],
            "bold": f["bold"],
            "italic": f["italic"],
            "formatted_prop": f["formatted_prop"],
            "ocr_confidence": s.get("ocr_confidence"),
            "fallbacks_applied": "|".join(s.get("fallbacks_applied") or []),
        }
        for bk, bv in (s.get("bools") or {}).items():
            row[f"bool_{bk}"] = bv
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def run_p2(repo_root: Optional[Path] = None, *, require_audit: bool = True) -> dict[str, Any]:
    """Compile all metadata variants. Optionally require a clean P2.6 audit first."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    data_cfg = OmegaConf.load(repo_root / "configs" / "data.yaml")
    pre_cfg = OmegaConf.load(repo_root / "configs" / "preprocessing.yaml")
    data_version = str(data_cfg.data_version)
    metadata_dir = repo_root / str(data_cfg.paths.metadata_dir)
    panel_map = dict(pre_cfg.canonical_panel_map)

    if require_audit:
        import subprocess
        import sys

        audit_script = repo_root / "scripts" / "run_audit.py"
        rc = subprocess.call([sys.executable, str(audit_script)], cwd=str(repo_root))
        if rc != 0:
            return {
                "ok": False,
                "errors": [f"P2.6 audit failed with exit code {rc}"],
                "data_version": data_version,
            }

    # question_id from P0 registry
    qpath = (
        repo_root
        / str(data_cfg.paths.processed_root)
        / data_version
        / "registry"
        / "question_types.json"
    )
    qid_by_trial: dict[str, str] = {}
    if qpath.is_file():
        qdoc = uio.read_json(qpath)
        qid_by_trial = dict(qdoc.get("question_id_by_trial") or {})

    out_dir = repo_root / str(data_cfg.paths.processed_root) / data_version / "metadata"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_qc: list[dict[str, Any]] = []
    fallback_totals: Counter = Counter()
    errors: list[str] = []
    n_variants = 0

    for path in list_metadata_files(metadata_dir):
        ident = parse_filename_identity(path.name)
        raw = uio.read_json(path)
        qid = qid_by_trial.get(ident.trial_id, ident.trial_id)
        segs, panels, qc = compile_variant(
            raw,
            trial_id=ident.trial_id,
            star_condition=ident.star_condition,
            question_id=qid,
            panel_map=panel_map,
        )
        for seg in segs:
            try:
                uio.validate(seg, "segment")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{path.name} {seg.get('segment_id')}: schema {e}")
                qc["ok"] = False

        stem = ident.stem
        uio.write_json(out_dir / f"{stem}__segments.json", segs)
        uio.write_json(out_dir / f"{stem}__panels.json", panels)
        # Panels are flat; segments stay JSON-canonical (nested schema fields).
        pd.DataFrame(panels).to_parquet(out_dir / f"{stem}__panels.parquet", index=False)
        _write_segments_parquet(out_dir / f"{stem}__segments.parquet", segs)

        qc["file"] = path.name
        qc["stem"] = stem
        all_qc.append(qc)
        for k, v in (qc.get("fallback_counts") or {}).items():
            fallback_totals[k] += v
        if not qc["ok"]:
            errors.extend(f"{path.name}: {e}" for e in qc["errors"])
        n_variants += 1

    summary = {
        "data_version": data_version,
        "out_dir": str(out_dir),
        "n_variants": n_variants,
        "fallback_totals": dict(fallback_totals),
        "n_unclaimed_boxes_total": sum(q.get("n_unclaimed_boxes", 0) for q in all_qc),
        "errors": errors,
        "ok": len(errors) == 0 and n_variants == int(data_cfg.expected.n_metadata_variants),
    }
    uio.write_json(out_dir / "p2_summary.json", summary)
    uio.write_json(out_dir / "p2_qc.json", all_qc)
    return summary
