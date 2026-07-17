"""P0 registries and identity.

Filename → (trial_id, star_condition) parsing; document-dimension registry;
star-condition assignment table; question_type validation from gaze TSVs;
S/NS metadata variant-consistency check.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
from PIL import Image
from omegaconf import OmegaConf

from src.utils import io as uio

FILENAME_RE = re.compile(r"^(T\d+)(S|NS)?", re.IGNORECASE)

# Metadata filename stems look like T21S-complete; strip the -complete suffix.
STEM_RE = re.compile(r"^(T\d+(?:S|NS)?)", re.IGNORECASE)

STAR_ELIGIBLE_DEFAULT = ("T11", "T12", "T13", "T21", "T27", "T30")


@dataclass(frozen=True)
class TrialIdentity:
    trial_id: str
    star_condition: str  # star_on | star_off | not_eligible
    stem: str  # e.g. T21S, T01


class RegistryError(Exception):
    """Raised when a P0 validation hard-fails."""


def parse_filename_identity(name: str) -> TrialIdentity:
    """Parse a metadata/image filename into trial identity.

    Patterns (preprocessing plan P0.1):
      T[n]   → trial_id=T[n], star_condition=not_eligible
      T[n]S  → trial_id=T[n], star_condition=star_on
      T[n]NS → trial_id=T[n], star_condition=star_off
    """
    stem = Path(name).stem
    m_stem = STEM_RE.match(stem)
    if not m_stem:
        raise ValueError(f"Unparseable filename stem: {name!r}")
    key = m_stem.group(1).upper()
    m = FILENAME_RE.match(key)
    if not m:
        raise ValueError(f"Unparseable trial key: {key!r} (from {name!r})")
    trial_id = m.group(1).upper()
    suffix = (m.group(2) or "").upper()
    star_condition = {"S": "star_on", "NS": "star_off", "": "not_eligible"}[suffix]
    return TrialIdentity(trial_id=trial_id, star_condition=star_condition, stem=key)


def load_data_config(repo_root: Path) -> Any:
    cfg_path = repo_root / "configs" / "data.yaml"
    return OmegaConf.load(cfg_path)


def _repo_path(repo_root: Path, rel: str) -> Path:
    return (repo_root / rel).resolve()


def list_metadata_files(metadata_dir: Path) -> list[Path]:
    files = sorted(metadata_dir.glob("*.json"))
    return [f for f in files if "save" not in f.name.lower()]


def list_document_images(image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)


def build_document_dimension_registry(image_dir: Path) -> list[dict[str, Any]]:
    """Read each document-space image once → (trial_id, star_condition, W, H)."""
    rows: list[dict[str, Any]] = []
    for img_path in list_document_images(image_dir):
        ident = parse_filename_identity(img_path.name)
        with Image.open(img_path) as im:
            w, h = im.size
        rows.append(
            {
                "trial_id": ident.trial_id,
                "star_condition": ident.star_condition,
                "stem": ident.stem,
                "filename": img_path.name,
                "W_doc": int(w),
                "H_doc": int(h),
            }
        )
    return rows


def build_trial_file_index(
    metadata_dir: Path,
    image_dir: Path,
    eligible: Iterable[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Match metadata ↔ images; return trial-variant rows and validation errors."""
    eligible_set = {e.upper() for e in eligible}
    errors: list[str] = []

    meta_by_stem: dict[str, Path] = {}
    for p in list_metadata_files(metadata_dir):
        try:
            ident = parse_filename_identity(p.name)
        except ValueError as e:
            errors.append(str(e))
            continue
        meta_by_stem[ident.stem] = p

    img_by_stem: dict[str, Path] = {}
    for p in list_document_images(image_dir):
        try:
            ident = parse_filename_identity(p.name)
        except ValueError as e:
            errors.append(str(e))
            continue
        img_by_stem[ident.stem] = p

    all_stems = sorted(set(meta_by_stem) | set(img_by_stem))
    rows: list[dict[str, Any]] = []
    for stem in all_stems:
        ident = parse_filename_identity(stem)
        meta = meta_by_stem.get(stem)
        img = img_by_stem.get(stem)
        if meta is None:
            errors.append(f"Missing metadata for stem {stem}")
        if img is None:
            errors.append(f"Missing document image for stem {stem}")
        if ident.trial_id in eligible_set and ident.star_condition == "not_eligible":
            errors.append(
                f"Eligible trial {ident.trial_id} has non-variant stem {stem}; "
                "expected S/NS variants only"
            )
        if ident.trial_id not in eligible_set and ident.star_condition != "not_eligible":
            errors.append(
                f"Non-eligible trial {ident.trial_id} has star variant stem {stem}"
            )
        rows.append(
            {
                "stem": stem,
                "trial_id": ident.trial_id,
                "star_condition": ident.star_condition,
                "metadata_file": meta.name if meta else None,
                "image_file": img.name if img else None,
                "is_star_eligible": ident.trial_id in eligible_set,
            }
        )

    trial_ids = {r["trial_id"] for r in rows}
    if len(trial_ids) != 30:
        errors.append(f"Expected 30 trials, found {len(trial_ids)}: {sorted(trial_ids)}")
    if len(rows) != 36:
        errors.append(f"Expected 36 trial variants (stems), found {len(rows)}")

    # Each eligible trial must have both star_on and star_off
    by_trial: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_trial[r["trial_id"]].add(r["star_condition"])
    for tid in sorted(eligible_set):
        conds = by_trial.get(tid, set())
        if conds != {"star_on", "star_off"}:
            errors.append(
                f"Eligible trial {tid} must have star_on+star_off variants; got {conds}"
            )

    return rows, errors


def _read_gaze_identity_columns(tsv_path: Path) -> pd.DataFrame:
    """Load only the columns needed for P0 identity / star / question_type."""
    usecols = [
        "Participant ID",
        "Trial",
        "Star Chart",
        "Question type",
        "Sensor",
    ]
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        encoding="utf-8",
        usecols=usecols,
        dtype={
            "Participant ID": "string",
            "Trial": "string",
            "Star Chart": "string",
            "Question type": "string",
            "Sensor": "string",
        },
        low_memory=False,
    )
    df = df[df["Sensor"] == "Eye Tracker"].copy()
    df["Trial"] = df["Trial"].fillna("").astype(str).str.strip()
    df = df[df["Trial"] != ""].copy()
    return df


def build_star_and_question_tables(
    gaze_dir: Path,
    eligible: Iterable[str],
    star_on_per_participant: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str], list[str]]:
    """Build star-condition assignments and per-trial question_type / question_id.

    question_type comes from the gaze ``Question type`` column, validated as
    constant per trial across all participants. question_id is the stable trial
    identity (``trial_id``), since each trial is one question with one response.
    """
    eligible_set = {e.upper() for e in eligible}
    errors: list[str] = []

    # (participant, trial) → set of star values / question types seen
    star_values: dict[tuple[str, str], set[str]] = defaultdict(set)
    qtype_by_trial_participant: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    participants: set[str] = set()

    tsvs = sorted(gaze_dir.glob("*.tsv"))
    if not tsvs:
        errors.append(f"No gaze TSVs found in {gaze_dir}")
        return [], {}, {}, errors

    for tsv in tsvs:
        df = _read_gaze_identity_columns(tsv)
        for (pid, trial), grp in df.groupby(["Participant ID", "Trial"], sort=False):
            pid_s = str(pid).strip()
            trial_s = str(trial).strip().upper()
            participants.add(pid_s)
            stars = {str(x).strip() for x in grp["Star Chart"].dropna().unique()}
            star_values[(pid_s, trial_s)] |= stars
            qtypes = {str(x).strip() for x in grp["Question type"].dropna().unique() if str(x).strip()}
            qtype_by_trial_participant[trial_s][pid_s] |= qtypes

    # Resolve star condition per episode
    assignments: list[dict[str, Any]] = []
    for (pid, trial), stars in sorted(star_values.items()):
        if len(stars) != 1:
            errors.append(
                f"Star Chart not constant for ({pid}, {trial}): {sorted(stars)}"
            )
            continue
        raw = next(iter(stars))
        # Export uses 0/1 (sometimes bool-ish strings)
        if raw in {"1", "1.0", "True", "true"}:
            cond = "star_on"
        elif raw in {"0", "0.0", "False", "false"}:
            cond = "star_off" if trial in eligible_set else "not_eligible"
        else:
            errors.append(f"Unrecognised Star Chart value {raw!r} for ({pid}, {trial})")
            continue

        if trial not in eligible_set and cond == "star_on":
            errors.append(f"star_on outside eligible trials: ({pid}, {trial})")
        if trial in eligible_set and cond == "not_eligible":
            # eligible trials should be star_on or star_off, never not_eligible
            errors.append(
                f"Eligible trial marked not_eligible: ({pid}, {trial}) raw={raw!r}"
            )

        if trial in eligible_set and cond == "star_off":
            pass  # ok
        elif trial not in eligible_set:
            cond = "not_eligible"

        assignments.append(
            {
                "participant_id": pid,
                "trial_id": trial,
                "star_condition": cond,
            }
        )

    # Exactly 3 star_on per participant among eligible
    by_pid: dict[str, list[str]] = defaultdict(list)
    for a in assignments:
        if a["star_condition"] == "star_on":
            by_pid[a["participant_id"]].append(a["trial_id"])
    for pid in sorted(participants):
        ons = sorted(by_pid.get(pid, []))
        if len(ons) != star_on_per_participant:
            errors.append(
                f"Participant {pid}: expected {star_on_per_participant} star_on "
                f"among eligible, got {len(ons)}: {ons}"
            )
        for t in ons:
            if t not in eligible_set:
                errors.append(f"Participant {pid}: star_on on non-eligible trial {t}")

    # question_type constant per trial across participants
    question_type: dict[str, str] = {}
    question_id: dict[str, str] = {}
    for trial, per_pid in sorted(qtype_by_trial_participant.items()):
        all_types: set[str] = set()
        for types in per_pid.values():
            all_types |= types
        if len(all_types) != 1:
            errors.append(
                f"Question type not constant for trial {trial}: {sorted(all_types)} "
                f"(per-participant={ {p: sorted(v) for p, v in per_pid.items()} })"
            )
            continue
        qt = next(iter(all_types))
        question_type[trial] = qt
        # Stable question identity = trial_id (one student response per question)
        question_id[trial] = trial

    return assignments, question_type, question_id, errors


def _is_star_segment(seg: dict[str, Any]) -> bool:
    """True if a segment belongs to the star overlay (not the non-star base)."""
    if seg.get("segment_type") == "star_concept":
        return True
    if seg.get("aoi_type") == "star_chart":
        return True
    if seg.get("is_star_chart") is True:
        return True
    return False


def _strip_star_content(data: dict[str, Any]) -> dict[str, Any]:
    """Non-star content for S/NS variant comparison.

    Removes star-chart AOIs, star_concept / star-panel segments, and text boxes
    referenced only by those segments or parented to a star AOI. AOI geometry is
    compared separately from identity (ids/types), because star_on layouts can
    shift panel boxes by a few pixels.
    """
    star_aoi_ids = {
        a.get("aoi_id")
        for a in data.get("aoi_annotations", [])
        if a.get("aoi_type") == "star_chart" and a.get("aoi_id")
    }
    aois_id_type = [
        {"aoi_id": a.get("aoi_id"), "aoi_type": a.get("aoi_type")}
        for a in data.get("aoi_annotations", [])
        if a.get("aoi_type") != "star_chart"
    ]
    aois_geometry = [
        {
            "aoi_id": a.get("aoi_id"),
            "x_min": a.get("x_min"),
            "y_min": a.get("y_min"),
            "x_max": a.get("x_max"),
            "y_max": a.get("y_max"),
        }
        for a in data.get("aoi_annotations", [])
        if a.get("aoi_type") != "star_chart"
    ]
    segs = [s for s in data.get("segments", []) if not _is_star_segment(s)]
    star_box_ids: set[str] = set()
    for s in data.get("segments", []):
        if _is_star_segment(s):
            for b in s.get("box_ids") or []:
                star_box_ids.add(b)
    boxes = [
        b
        for b in data.get("text_boxes", [])
        if b.get("box_id") not in star_box_ids
        and b.get("parent_region") not in star_aoi_ids
    ]
    # Segment compare on identity fields (not raw geometry via boxes — geometry
    # is recovered later from box unions and may shift with layout).
    seg_identity = [
        {
            "segment_id": s.get("segment_id"),
            "segment_type": s.get("segment_type"),
            "aoi_type": s.get("aoi_type"),
            "aoi_id": s.get("aoi_id"),
            "segment_order": s.get("segment_order"),
            "corrected_text": s.get("corrected_text"),
            "mark_point_id": s.get("mark_point_id"),
            "star_id": s.get("star_id"),
            "level_band": s.get("level_band"),
        }
        for s in segs
    ]
    return {
        "aoi_id_types": _canonical_json(sorted(aois_id_type, key=lambda x: str(x["aoi_id"]))),
        "aoi_geometry": _canonical_json(sorted(aois_geometry, key=lambda x: str(x["aoi_id"]))),
        "text_boxes": _canonical_json(boxes),
        "segments": _canonical_json(seg_identity),
    }


def _canonical_json(obj: Any) -> Any:
    """Sort keys / normalise for equality comparison."""
    return json.loads(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def _norm_text(text: Any) -> str:
    if text is None:
        return ""
    s = str(text)
    # Strip curly/smart quotes and similar so allowlist patterns like
    # "enter a level" still match "enter" / "enter".
    for ch in (
        "\u2018",
        "\u2019",
        "\u201c",
        "\u201d",
        "\u00b4",
        "`",
        "'",
        '"',
    ):
        s = s.replace(ch, "")
    return " ".join(s.split()).casefold()


def _aoi_type_multiset(data: dict[str, Any]) -> dict[str, int]:
    from collections import Counter

    types = [
        a.get("aoi_type")
        for a in data.get("aoi_annotations", [])
        if a.get("aoi_type") and a.get("aoi_type") != "star_chart"
    ]
    return dict(Counter(types))


def _is_star_conditional_text(text: str, patterns: Iterable[str]) -> bool:
    t = _norm_text(text)
    return any(_norm_text(p) in t for p in patterns)


def check_variant_consistency(
    metadata_dir: Path,
    eligible: Iterable[str],
    *,
    star_conditional_patterns: Optional[Iterable[str]] = None,
    panel_map: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """NS↔S non-star correspondence check (DECISIONS.md M3-C1 / amended #11).

    Per-variant construction: geometry and AOI ids may differ. Every NS non-star
    segment must map 1:1 to an S segment on (canonical panel, normalised text,
    relative order within panel). S-only segments whose text matches the
    star-conditional allowlist are recorded (not failures). AOI *type* multisets
    should match; id/geometry drift is soft.
    """
    from collections import Counter, defaultdict

    patterns = list(
        star_conditional_patterns
        or [
            "enter a level",
            "enter' a level",
            "put stars on the left",
            "put a star on the left",
            "put a star on the right",
            "using the system of stars",
            "system of stars described",
        ]
    )
    pmap = panel_map or {
        "question": "question",
        "response": "response",
        "mark_scheme": "mark_scheme",
        "mark_scheme_answers": "mark_scheme",
        "mark_scheme_extra_information": "mark_scheme",
        "level_descriptor": "mark_scheme",
        "commentary": "commentary",
        "star_chart": "star_chart",
        "general_ui": "ui_general",
        "answer_scroll_bar": "answer_scroll_bar",
        "commentary_scroll_bar": "commentary_scroll_bar",
    }

    def panel_of(seg: dict[str, Any]) -> str:
        at = seg.get("aoi_type") or ""
        return pmap.get(at, at or "unknown")

    def nonstar_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for s in data.get("segments", []):
            if _is_star_segment(s):
                continue
            rows.append(
                {
                    "segment_id": s.get("segment_id"),
                    "panel": panel_of(s),
                    "text": _norm_text(s.get("corrected_text")),
                    "segment_order": s.get("segment_order"),
                }
            )
        return rows

    def panel_text_seq(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
        by_panel: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_panel[r["panel"]].append(r)
        out: dict[str, list[str]] = {}
        for panel, items in by_panel.items():
            items = sorted(
                items,
                key=lambda r: (
                    r["segment_order"] is None,
                    r["segment_order"] if r["segment_order"] is not None else 0,
                    r["segment_id"] or "",
                ),
            )
            out[panel] = [r["text"] for r in items]
        return out

    reports: list[dict[str, Any]] = []
    files = {parse_filename_identity(p.name).stem: p for p in list_metadata_files(metadata_dir)}
    for tid in eligible:
        s_stem, ns_stem = f"{tid}S", f"{tid}NS"
        s_path, ns_path = files.get(s_stem), files.get(ns_stem)
        if s_path is None or ns_path is None:
            reports.append(
                {
                    "trial_id": tid,
                    "ok": False,
                    "hard_fail": True,
                    "message": f"Missing S/NS pair: S={s_path}, NS={ns_path}",
                }
            )
            continue
        s_data = uio.read_json(s_path)
        ns_data = uio.read_json(ns_path)
        s_rows = nonstar_rows(s_data)
        ns_rows = nonstar_rows(ns_data)

        s_bag = Counter((r["panel"], r["text"]) for r in s_rows)
        ns_bag = Counter((r["panel"], r["text"]) for r in ns_rows)

        unmatched_ns = []
        remaining_s = s_bag.copy()
        for k, n in ns_bag.items():
            have = remaining_s.get(k, 0)
            if have < n:
                unmatched_ns.append(
                    {
                        "panel": k[0],
                        "text_preview": k[1][:120],
                        "needed": n,
                        "have_in_s": have,
                    }
                )
            remaining_s[k] = have - n

        star_conditional = []
        unexpected_s = []
        for k, n in list(remaining_s.items()):
            if n <= 0:
                continue
            panel, text = k
            if text == "" or _is_star_conditional_text(text, patterns):
                star_conditional.append(
                    {
                        "panel": panel,
                        "text_preview": text[:120],
                        "count": n,
                        "is_star_conditional": True,
                    }
                )
            else:
                unexpected_s.append(
                    {"panel": panel, "text_preview": text[:120], "count": n}
                )

        # Within-panel order among shared texts (exclude star-conditional S extras)
        rem = ns_bag.copy()
        s_shared = []
        for r in sorted(
            s_rows,
            key=lambda x: (
                x["segment_order"] is None,
                x["segment_order"] if x["segment_order"] is not None else 0,
                x["segment_id"] or "",
            ),
        ):
            key = (r["panel"], r["text"])
            if rem.get(key, 0) > 0:
                s_shared.append(r)
                rem[key] -= 1
        order_mismatch_panels = []
        s_seq = panel_text_seq(s_shared)
        ns_seq = panel_text_seq(ns_rows)
        for panel in sorted(set(s_seq) | set(ns_seq)):
            if s_seq.get(panel, []) != ns_seq.get(panel, []):
                order_mismatch_panels.append(panel)

        s_types = _aoi_type_multiset(s_data)
        ns_types = _aoi_type_multiset(ns_data)
        aoi_type_ok = s_types == ns_types

        s_core = _strip_star_content(s_data)
        ns_core = _strip_star_content(ns_data)
        soft_diffs = [
            k
            for k in ("aoi_geometry", "text_boxes", "aoi_id_types")
            if s_core[k] != ns_core[k]
        ]

        hard_fail = (
            bool(unmatched_ns)
            or bool(unexpected_s)
            or bool(order_mismatch_panels)
            or (not aoi_type_ok)
        )
        ok = not hard_fail
        parts = []
        if ok:
            parts.append("NS<->S correspondence OK")
        if unmatched_ns:
            parts.append(f"{len(unmatched_ns)} unmatched NS segments")
        if unexpected_s:
            parts.append(f"{len(unexpected_s)} unexpected S-only segments")
        if star_conditional:
            n_sc = sum(x["count"] for x in star_conditional)
            parts.append(f"{n_sc} allowlisted star-conditional S-only")
        if order_mismatch_panels:
            parts.append(f"within-panel order mismatch: {order_mismatch_panels}")
        if not aoi_type_ok:
            parts.append(f"AOI type multiset differ S={s_types} NS={ns_types}")
        if soft_diffs:
            parts.append(f"soft drift in {soft_diffs}")

        reports.append(
            {
                "trial_id": tid,
                "ok": ok,
                "hard_fail": hard_fail,
                "unmatched_ns": unmatched_ns,
                "unexpected_s_only": unexpected_s,
                "star_conditional_s_only": star_conditional,
                "order_mismatch_panels": order_mismatch_panels,
                "aoi_type_multiset_ok": aoi_type_ok,
                "soft_diffs": soft_diffs,
                "message": "; ".join(parts),
            }
        )
    return reports


def run_p0(repo_root: Optional[Path] = None) -> dict[str, Any]:
    """Build all P0 registries; write outputs; return a summary dict."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    cfg = load_data_config(repo_root)
    data_version = str(cfg.data_version)
    eligible = list(cfg.star_eligible_trials)
    metadata_dir = _repo_path(repo_root, cfg.paths.metadata_dir)
    image_dir = _repo_path(repo_root, cfg.paths.document_images_dir)
    gaze_dir = _repo_path(repo_root, cfg.paths.gaze_dir)
    out_dir = _repo_path(repo_root, cfg.paths.processed_root) / data_version / "registry"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_errors: list[str] = []

    trial_rows, file_errors = build_trial_file_index(metadata_dir, image_dir, eligible)
    all_errors.extend(file_errors)

    dim_rows = build_document_dimension_registry(image_dir)

    assignments, qtype, qid, gaze_errors = build_star_and_question_tables(
        gaze_dir,
        eligible,
        star_on_per_participant=int(cfg.expected.star_on_per_participant),
    )
    all_errors.extend(gaze_errors)

    # Attach question_type / question_id onto trial registry rows
    trial_registry = []
    for r in trial_rows:
        tid = r["trial_id"]
        trial_registry.append(
            {
                **r,
                "question_id": qid.get(tid, tid),
                "question_type": qtype.get(tid),
            }
        )

    pre_cfg = OmegaConf.load(repo_root / "configs" / "preprocessing.yaml")
    panel_map = dict(pre_cfg.canonical_panel_map)
    sc_patterns = list(pre_cfg.get("star_conditional_text_patterns", []) or [])
    variant_report = check_variant_consistency(
        metadata_dir,
        eligible,
        star_conditional_patterns=sc_patterns or None,
        panel_map=panel_map,
    )
    soft_warnings: list[str] = []
    for vr in variant_report:
        if vr.get("hard_fail"):
            all_errors.append(f"Variant consistency {vr['trial_id']}: {vr['message']}")
        else:
            notes = []
            if vr.get("star_conditional_s_only"):
                n_sc = sum(x["count"] for x in vr["star_conditional_s_only"])
                notes.append(f"{n_sc} star-conditional S-only")
            if vr.get("soft_diffs"):
                notes.append(f"soft drift in {vr['soft_diffs']}")
            if notes:
                soft_warnings.append(
                    f"Variant {vr['trial_id']}: {'; '.join(notes)} — {vr['message']}"
                )

    # Star-condition schema document
    star_doc = {
        "schema_version": data_version,
        "eligible_trials": list(eligible),
        "assignments": assignments,
    }
    if assignments:
        try:
            uio.validate(star_doc, "star_conditions")
        except Exception as e:  # noqa: BLE001
            all_errors.append(f"star_conditions schema validation failed: {e}")

    # Write outputs
    uio.write_json(out_dir / "trial_registry.json", trial_registry)
    uio.write_json(out_dir / "document_dimensions.json", dim_rows)
    uio.write_json(out_dir / "star_conditions.json", star_doc)
    uio.write_json(
        out_dir / "question_types.json",
        {
            "question_type_by_trial": qtype,
            "question_id_by_trial": qid,
        },
    )
    uio.write_json(out_dir / "variant_consistency.json", variant_report)

    # Parquet for star conditions (plan asks for parquet)
    if assignments:
        pd.DataFrame(assignments).to_parquet(out_dir / "star_conditions.parquet", index=False)
    pd.DataFrame(trial_registry).to_parquet(out_dir / "trial_registry.parquet", index=False)
    pd.DataFrame(dim_rows).to_parquet(out_dir / "document_dimensions.parquet", index=False)

    summary = {
        "data_version": data_version,
        "out_dir": str(out_dir),
        "n_trial_variants": len(trial_registry),
        "n_document_images": len(dim_rows),
        "n_star_assignments": len(assignments),
        "n_question_types": len(qtype),
        "variant_ok": all(v["ok"] for v in variant_report),
        "soft_warnings": soft_warnings,
        "errors": all_errors,
        "ok": len(all_errors) == 0,
    }
    uio.write_json(out_dir / "p0_summary.json", summary)
    return summary
