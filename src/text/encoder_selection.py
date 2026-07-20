"""M2 — encoder bake-off: ranking accuracy on reviewed (anchor, related, unrelated) triples."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils import io as uio


REQUIRED_COLS = (
    "pair_id",
    "category",
    "anchor_text",
    "related_text",
    "unrelated_text",
    "reviewed",
)

NEGATIVE_HARD = "hard_within_trial"
NEGATIVE_EASY = "easy_cross_trial"
VALID_NEGATIVE_TYPES = frozenset({NEGATIVE_HARD, NEGATIVE_EASY})

# Rubric / examiner-instruction phrasing — never use as *related* (may be unrelated).
_RUBRIC_RE = re.compile(
    r"(?:^\s*(?:any\s+(?:two|three|four|\d+)\s+from)\s*$)"
    r"|(?:\b(?:"
    r"allow(?:\s+named\s+example)?"
    r"|do\s+not\s+accept"
    r"|do\s+not\s+allow"
    r"|ignore\b"
    r"|accept\s+(?:either|any)"
    r"|or\s+equivalent"
    r"|credit\s+(?:references?\s+to|for)"
    r"|reject\b"
    r"|ignore\s+(?:references?\s+to)?"
    r")\b)",
    re.IGNORECASE | re.DOTALL,
)

ACTIVE_CATEGORIES = ("response_mark_scheme", "commentary_paraphrase")


def ranking_accuracy(
    anchor_emb: np.ndarray,
    related_emb: np.ndarray,
    unrelated_emb: np.ndarray,
) -> float:
    """Fraction of triples where cos(anchor, related) > cos(anchor, unrelated)."""
    if len(anchor_emb) == 0:
        return float("nan")

    def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        )

    rel = _cos(anchor_emb, related_emb)
    unr = _cos(anchor_emb, unrelated_emb)
    return float(np.mean(rel > unr))


def load_pair_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="utf-8")
    else:
        df = pd.read_parquet(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"pair table missing columns: {missing}")
    return df


def infer_negative_type(row: pd.Series) -> str:
    """Derive hard/easy from trial ids when present; same-trial unrelated is valid (hard)."""
    a = row.get("anchor_trial")
    u = row.get("unrelated_trial")
    if pd.isna(a) or pd.isna(u) or a is None or u is None or str(a) == "" or str(u) == "":
        stored = row.get("negative_type")
        if stored in VALID_NEGATIVE_TYPES:
            return str(stored)
        return NEGATIVE_EASY
    if str(a) == str(u):
        return NEGATIVE_HARD
    return NEGATIVE_EASY


def sync_negative_types(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute negative_type from trials after owner edits; do not enforce draft mix."""
    out = df.copy()
    out["negative_type"] = [infer_negative_type(r) for _, r in out.iterrows()]
    return out


def reviewed_triples(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["reviewed"].astype(bool)].copy()
    for col in ("anchor_text", "related_text", "unrelated_text"):
        out = out[out[col].astype(str).str.strip().str.len() > 0]
    out = sync_negative_types(out)
    return out.reset_index(drop=True)


def encode_texts(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
    normalize: bool = True,
    prefix: str = "",
) -> np.ndarray:
    prepared = [f"{prefix}{t}" if prefix else str(t) for t in texts]
    # HFMeanPoolEncoder / ST-compatible encode()
    emb = model.encode(
        prepared,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    return np.asarray(emb, dtype=np.float32)


def evaluate_candidate(
    model: Any,
    triples: pd.DataFrame,
    *,
    batch_size: int = 32,
    normalize: bool = True,
    prefix: str = "",
) -> dict[str, Any]:
    anchors = triples["anchor_text"].astype(str).tolist()
    related = triples["related_text"].astype(str).tolist()
    unrelated = triples["unrelated_text"].astype(str).tolist()
    a = encode_texts(model, anchors, batch_size=batch_size, normalize=normalize, prefix=prefix)
    r = encode_texts(model, related, batch_size=batch_size, normalize=normalize, prefix=prefix)
    u = encode_texts(model, unrelated, batch_size=batch_size, normalize=normalize, prefix=prefix)
    acc = ranking_accuracy(a, r, u)

    by_cat: dict[str, float] = {}
    cats = triples["category"].astype(str).to_numpy()
    for cat in sorted(set(cats.tolist())):
        mask = cats == cat
        by_cat[cat] = ranking_accuracy(a[mask], r[mask], u[mask])

    by_neg: dict[str, Any] = {}
    if "negative_type" in triples.columns:
        negs = triples["negative_type"].astype(str).to_numpy()
        for nt in sorted(set(negs.tolist())):
            mask = negs == nt
            by_neg[nt] = {
                "ranking_accuracy": ranking_accuracy(a[mask], r[mask], u[mask]),
                "n": int(mask.sum()),
            }

    hard_acc = by_neg.get(NEGATIVE_HARD, {}).get("ranking_accuracy")
    easy_acc = by_neg.get(NEGATIVE_EASY, {}).get("ranking_accuracy")

    return {
        "ranking_accuracy": acc,
        "ranking_accuracy_hard": hard_acc,
        "ranking_accuracy_easy": easy_acc,
        "n_triples": int(len(triples)),
        "per_category": by_cat,
        "per_negative_type": by_neg,
        "embedding_dim": int(a.shape[1]),
    }


def _sort_key(r: dict[str, Any]) -> tuple[float, float]:
    """Overall accuracy primary; hard-negative accuracy is the tie-breaker."""
    overall = float(r.get("ranking_accuracy") or 0.0)
    hard = r.get("ranking_accuracy_hard")
    hard_f = float(hard) if hard is not None and hard == hard else -1.0
    return (overall, hard_f)


def run_bakeoff(
    repo_root: Path,
    *,
    require_reviewed: bool = True,
) -> dict[str, Any]:
    """Score candidates on reviewed pairs. Raises if review gate not met."""
    cfg = OmegaConf.load(repo_root / "configs" / "encoder_selection.yaml")
    reviewed_path = repo_root / str(cfg.paths.reviewed_pairs)
    draft_path = repo_root / str(cfg.paths.draft_pairs)

    if reviewed_path.is_file():
        pairs = load_pair_table(reviewed_path)
    elif draft_path.is_file() and not require_reviewed:
        pairs = load_pair_table(draft_path)
    else:
        return {
            "ok": False,
            "blocked": True,
            "message": (
                f"Reviewed pair set not found at {reviewed_path}. "
                "Curate reports/encoder_pairs/draft_pairs_v1.csv (set reviewed=true), "
                "then promote with scripts/promote_encoder_pairs.py."
            ),
        }

    triples = reviewed_triples(pairs) if require_reviewed else sync_negative_types(pairs)
    min_n = int(cfg.min_reviewed_triples)
    if require_reviewed and len(triples) < min_n:
        return {
            "ok": False,
            "blocked": True,
            "message": (
                f"Only {len(triples)} reviewed triples; need ≥ {min_n}. "
                "Continue curation before bake-off scoring."
            ),
            "n_reviewed": int(len(triples)),
        }

    from src.text.hf_encoder import load_text_encoder

    results = []
    for cand in cfg.candidates:
        model = load_text_encoder(
            str(cand.model_name),
            text_prefix=str(cand.get("text_prefix") or ""),
            batch_size=int(cfg.batch_size),
            normalise_l2=bool(cfg.normalise_l2),
        )
        # prefix already applied inside encoder when constructed; don't double-prefix
        metrics = evaluate_candidate(
            model,
            triples,
            batch_size=int(cfg.batch_size),
            normalize=bool(cfg.normalise_l2),
            prefix="",
        )
        results.append(
            {
                "id": str(cand.id),
                "model_name": str(cand.model_name),
                "text_prefix": str(cand.get("text_prefix") or ""),
                **metrics,
            }
        )

    results_sorted = sorted(results, key=_sort_key, reverse=True)
    winner = results_sorted[0]
    n_hard = int((triples["negative_type"] == NEGATIVE_HARD).sum()) if "negative_type" in triples else 0
    n_easy = int((triples["negative_type"] == NEGATIVE_EASY).sum()) if "negative_type" in triples else 0
    summary = {
        "ok": True,
        "blocked": False,
        "n_triples": int(len(triples)),
        "n_hard_within_trial": n_hard,
        "n_easy_cross_trial": n_easy,
        "metric": "ranking_accuracy",
        "tie_breaker": "ranking_accuracy_hard",
        "results": results_sorted,
        "winner_id": winner["id"],
        "winner_model": winner["model_name"],
        "winner_accuracy": winner["ranking_accuracy"],
        "winner_accuracy_hard": winner.get("ranking_accuracy_hard"),
        "winner_accuracy_easy": winner.get("ranking_accuracy_easy"),
    }

    report_path = repo_root / str(cfg.paths.bakeoff_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Encoder bake-off v1",
        "",
        f"Metric: ranking accuracy (related > unrelated), n={len(triples)} reviewed triples "
        f"(hard_within_trial={n_hard}, easy_cross_trial={n_easy}).",
        "Tie-breaker between equal overall scores: hard-negative ranking accuracy.",
        "",
        "| id | model | overall | hard | easy | dim |",
        "|---|---|---|---|---|---|",
    ]
    for r in results_sorted:
        hard = r.get("ranking_accuracy_hard")
        easy = r.get("ranking_accuracy_easy")
        hard_s = f"{hard:.4f}" if hard is not None and hard == hard else "—"
        easy_s = f"{easy:.4f}" if easy is not None and easy == easy else "—"
        lines.append(
            f"| {r['id']} | `{r['model_name']}` | {r['ranking_accuracy']:.4f} | "
            f"{hard_s} | {easy_s} | {r['embedding_dim']} |"
        )
    lines += [
        "",
        f"**Winner:** `{winner['id']}` — `{winner['model_name']}` "
        f"(overall={winner['ranking_accuracy']:.4f}"
        + (
            f", hard={winner['ranking_accuracy_hard']:.4f}"
            if winner.get("ranking_accuracy_hard") is not None
            and winner["ranking_accuracy_hard"] == winner["ranking_accuracy_hard"]
            else ""
        )
        + ").",
        "",
    ]
    uio.write_text(report_path, "\n".join(lines) + "\n")
    uio.write_json(repo_root / "reports" / "encoder_bakeoff_v1.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Segment filters / scoring for draft proposal
# ---------------------------------------------------------------------------


def _norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "").strip().lower())


def _tokens(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", _norm_text(t)) if len(w) > 2}


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_level_segment(s: dict[str, Any]) -> bool:
    b = s.get("bools") or {}
    return bool(b.get("is_level_descriptor")) or s.get("segment_role") == "level_descriptor"


def is_rubric_instruction(s: dict[str, Any]) -> bool:
    """Examiner meta-instructions — exclude from related; OK as unrelated distractors."""
    b = s.get("bools") or {}
    if b.get("contains_allow_instruction") or b.get("contains_reject_instruction"):
        return True
    text = str(s.get("corrected_text") or "")
    return bool(_RUBRIC_RE.search(text))


def is_content_mark_scheme(s: dict[str, Any]) -> bool:
    if str(s.get("panel_label")) != "mark_scheme":
        return False
    if is_level_segment(s):
        return False
    if is_rubric_instruction(s):
        return False
    return True


def _seg_key(s: dict[str, Any]) -> tuple[Any, Any]:
    return (s.get("trial_id"), s.get("segment_id"))


def _pick(rng: np.random.Generator, pool: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not pool:
        return None
    return pool[int(rng.integers(0, len(pool)))]


def _best_related(
    anchor: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    used_related: set[tuple[Any, Any]],
    min_jaccard: float = 0.05,
) -> Optional[dict[str, Any]]:
    """Prefer unused content candidates with highest Jaccard to anchor; never identical text."""
    a_text = anchor["corrected_text"]
    a_norm = _norm_text(a_text)
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in candidates:
        if _seg_key(c) == _seg_key(anchor):
            continue
        if _norm_text(c["corrected_text"]) == a_norm:
            continue
        score = jaccard(a_text, c["corrected_text"])
        # Mild preference for unused relateds
        bonus = 0.02 if _seg_key(c) not in used_related else 0.0
        scored.append((score + bonus, c))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    # Require some lexical overlap for response↔MS; commentary can be looser
    best_score, best = scored[0]
    if best_score < min_jaccard and min_jaccard > 0:
        # Still take best if nothing better; caller may filter
        pass
    return best


def _load_segments(repo_root: Path, min_chars: int) -> list[dict[str, Any]]:
    data_cfg = OmegaConf.load(repo_root / "configs" / "data.yaml")
    meta_dir = (
        repo_root
        / str(data_cfg.paths.processed_root)
        / str(data_cfg.data_version)
        / "metadata"
    )
    segs: list[dict[str, Any]] = []
    for path in sorted(meta_dir.glob("*__segments.json")):
        for s in uio.read_json(path):
            text = str(s.get("corrected_text") or "").strip()
            if len(text) < min_chars:
                continue
            segs.append(
                {
                    "trial_id": s.get("trial_id"),
                    "star_condition": s.get("star_condition"),
                    "segment_id": s.get("segment_id"),
                    "panel_label": s.get("panel_label"),
                    "segment_role": s.get("segment_role"),
                    "corrected_text": text,
                    "bools": s.get("bools") or {},
                }
            )
    return segs


def propose_draft_pairs(
    repo_root: Path,
    *,
    n_new: Optional[int] = None,
    keepers: Optional[pd.DataFrame] = None,
    start_pair_index: int = 0,
) -> pd.DataFrame:
    """Sample new triples (response_mark_scheme + commentary_paraphrase only).

    When *n_new* is set, sample that many total (~half each category, 50/50 hard/easy).
    Otherwise use config max_pairs_per_category for each active category.
    """
    cfg = OmegaConf.load(repo_root / "configs" / "encoder_selection.yaml")
    min_chars = int(cfg.draft.min_text_chars)
    hard_frac = float(cfg.draft.get("hard_fraction", 0.5))
    seed = int(cfg.seed)
    # Offset seed when appending so batch-2 differs from batch-1
    if keepers is not None and len(keepers):
        seed = seed + 1000 + int(len(keepers))
    rng = np.random.default_rng(seed)

    segs = _load_segments(repo_root, min_chars)
    by_panel: dict[str, list[dict[str, Any]]] = {}
    for s in segs:
        by_panel.setdefault(str(s["panel_label"]), []).append(s)

    responses = by_panel.get("response", [])
    mark_content = [m for m in by_panel.get("mark_scheme", []) if is_content_mark_scheme(m)]
    commentary = by_panel.get("commentary", [])

    same_trial_ms: dict[Any, list] = {}
    for m in mark_content:
        same_trial_ms.setdefault(m["trial_id"], []).append(m)
    by_trial_comm: dict[Any, list] = {}
    for c in commentary:
        by_trial_comm.setdefault(c["trial_id"], []).append(c)

    # Avoid reusing anchors / relateds already in keepers
    used_anchor_keys: set[tuple[Any, Any]] = set()
    used_related_keys: set[tuple[Any, Any]] = set()
    used_anchor_texts: set[str] = set()
    if keepers is not None and len(keepers):
        for _, row in keepers.iterrows():
            at, asid = row.get("anchor_trial"), row.get("anchor_segment_id")
            if pd.notna(at) and pd.notna(asid):
                used_anchor_keys.add((at, asid))
            rt, rsid = row.get("related_trial"), row.get("related_segment_id")
            if pd.notna(rt) and pd.notna(rsid):
                used_related_keys.add((rt, rsid))
            used_anchor_texts.add(_norm_text(str(row.get("anchor_text") or "")))

    if n_new is not None:
        n_rms = (n_new + 1) // 2
        n_comm = n_new - n_rms
    else:
        n_rms = int(cfg.draft.max_pairs_per_category)
        n_comm = int(cfg.draft.max_pairs_per_category)

    rows: list[dict[str, Any]] = []
    pair_i = start_pair_index

    def _add(
        category: str,
        anchor: dict,
        related: dict,
        unrelated: dict,
        negative_type: str,
    ) -> bool:
        nonlocal pair_i
        if _norm_text(anchor["corrected_text"]) == _norm_text(related["corrected_text"]):
            return False
        if _norm_text(related["corrected_text"]) == _norm_text(unrelated["corrected_text"]):
            return False
        if negative_type == NEGATIVE_HARD and anchor["trial_id"] != unrelated["trial_id"]:
            return False
        if negative_type == NEGATIVE_EASY and anchor["trial_id"] == unrelated["trial_id"]:
            return False
        # Related must be same trial as anchor for both active categories
        if anchor["trial_id"] != related["trial_id"]:
            return False
        rows.append(
            {
                "pair_id": f"draft_{pair_i:04d}",
                "category": category,
                "negative_type": negative_type,
                "anchor_trial": anchor.get("trial_id"),
                "anchor_segment_id": anchor.get("segment_id"),
                "related_trial": related.get("trial_id"),
                "related_segment_id": related.get("segment_id"),
                "unrelated_trial": unrelated.get("trial_id"),
                "unrelated_segment_id": unrelated.get("segment_id"),
                "anchor_text": anchor["corrected_text"],
                "related_text": related["corrected_text"],
                "unrelated_text": unrelated["corrected_text"],
                "reviewed": False,
                "owner_notes": "",
            }
        )
        pair_i += 1
        used_anchor_keys.add(_seg_key(anchor))
        used_related_keys.add(_seg_key(related))
        used_anchor_texts.add(_norm_text(anchor["corrected_text"]))
        return True

    def _fill_category(
        category: str,
        n_total: int,
        build_hard: Any,
        build_easy: Any,
    ) -> None:
        n_hard = max(1, int(round(n_total * hard_frac))) if n_total > 1 else n_total
        n_easy = n_total - n_hard
        # Oversample candidates then take
        hard_cands = build_hard(max(n_hard * 8, n_hard + 5))
        easy_cands = build_easy(max(n_easy * 8, n_easy + 5))
        rng.shuffle(hard_cands)
        rng.shuffle(easy_cands)
        added_h = 0
        for triple in hard_cands:
            if added_h >= n_hard:
                break
            if _add(category, *triple, NEGATIVE_HARD):
                added_h += 1
        added_e = 0
        for triple in easy_cands:
            if added_e >= n_easy:
                break
            if _add(category, *triple, NEGATIVE_EASY):
                added_e += 1

    def _build_rms(n_want: int, hard: bool) -> list[tuple]:
        out: list[tuple] = []
        pool = [r for r in responses if _seg_key(r) not in used_anchor_keys]
        rng.shuffle(pool)
        for r in pool:
            if len(out) >= n_want:
                break
            if _norm_text(r["corrected_text"]) in used_anchor_texts:
                continue
            ms_list = same_trial_ms.get(r["trial_id"]) or []
            if len(ms_list) < (2 if hard else 1):
                continue
            rel = _best_related(r, ms_list, used_related=used_related_keys, min_jaccard=0.04)
            if rel is None:
                continue
            # Prefer some lexical overlap for related
            if jaccard(r["corrected_text"], rel["corrected_text"]) < 0.04:
                continue
            if hard:
                hard_pool = [
                    m
                    for m in ms_list
                    if _seg_key(m) != _seg_key(rel)
                    and _norm_text(m["corrected_text"]) != _norm_text(rel["corrected_text"])
                ]
                # Prefer lower overlap with anchor than related has
                hard_pool.sort(
                    key=lambda m: jaccard(r["corrected_text"], m["corrected_text"])
                )
                if not hard_pool:
                    continue
                u = hard_pool[0] if len(hard_pool) == 1 else _pick(rng, hard_pool[: max(2, len(hard_pool) // 2)])
            else:
                easy_pool = [
                    m
                    for m in mark_content
                    if m["trial_id"] != r["trial_id"]
                    and _norm_text(m["corrected_text"]) != _norm_text(rel["corrected_text"])
                ]
                u = _pick(rng, easy_pool)
            if u is None:
                continue
            out.append((r, rel, u))
        return out

    def _build_comm(n_want: int, hard: bool) -> list[tuple]:
        out: list[tuple] = []
        trial_ids = list(by_trial_comm.keys())
        rng.shuffle(trial_ids)
        for tid in trial_ids:
            if len(out) >= n_want:
                break
            group = by_trial_comm[tid]
            # Distinct texts only
            uniq: list[dict] = []
            seen_t: set[str] = set()
            for c in group:
                nt = _norm_text(c["corrected_text"])
                if nt in seen_t:
                    continue
                seen_t.add(nt)
                uniq.append(c)
            need = 3 if hard else 2
            if len(uniq) < need:
                continue
            # Pick anchor unused
            anchors = [c for c in uniq if _seg_key(c) not in used_anchor_keys]
            if not anchors:
                continue
            a = _pick(rng, anchors)
            assert a is not None
            if _norm_text(a["corrected_text"]) in used_anchor_texts:
                continue
            others = [c for c in uniq if _seg_key(c) != _seg_key(a)]
            rel = _best_related(a, others, used_related=used_related_keys, min_jaccard=0.0)
            if rel is None:
                continue
            if hard:
                hard_pool = [
                    c
                    for c in others
                    if _seg_key(c) != _seg_key(rel)
                    and _norm_text(c["corrected_text"]) != _norm_text(rel["corrected_text"])
                ]
                u = _pick(rng, hard_pool)
            else:
                easy_pool = [
                    c
                    for c in commentary
                    if c["trial_id"] != tid
                    and _norm_text(c["corrected_text"]) != _norm_text(rel["corrected_text"])
                ]
                u = _pick(rng, easy_pool)
            if u is None:
                continue
            out.append((a, rel, u))
        return out

    _fill_category(
        "response_mark_scheme",
        n_rms,
        lambda n: _build_rms(n, hard=True),
        lambda n: _build_rms(n, hard=False),
    )
    _fill_category(
        "commentary_paraphrase",
        n_comm,
        lambda n: _build_comm(n, hard=True),
        lambda n: _build_comm(n, hard=False),
    )

    return pd.DataFrame(rows)


def write_draft_pairs(repo_root: Path, df: pd.DataFrame) -> Path:
    cfg = OmegaConf.load(repo_root / "configs" / "encoder_selection.yaml")
    out_path = repo_root / str(cfg.paths.draft_pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8", lineterminator="\n")
    uio.write_json(out_path.with_suffix(".json"), df.to_dict(orient="records"))
    return out_path


def propose_and_write(repo_root: Path) -> dict[str, Any]:
    """Full regenerate (no keepers) — active categories only."""
    df = propose_draft_pairs(repo_root)
    path = write_draft_pairs(repo_root, df)
    return _summary(df, path, message="Draft pairs written (response_mark_scheme + commentary_paraphrase only).")


def append_new_to_keepers(
    repo_root: Path,
    keepers_path: Path,
    *,
    n_new: int = 15,
) -> dict[str, Any]:
    """Append ~n_new unreviewed candidates to owner-reviewed keepers; write draft_pairs_v1.csv."""
    keepers_path = Path(keepers_path)
    if not keepers_path.is_file():
        return {
            "ok": False,
            "message": (
                f"Keepers file not found: {keepers_path}. "
                "Place draft_pairs_v1_reviewed.csv under reports/encoder_pairs/."
            ),
        }
    keepers = load_pair_table(keepers_path)
    # Normalise reviewed flag
    keepers = keepers.copy()
    keepers["reviewed"] = True
    keepers = sync_negative_types(keepers)

    # Drop retired categories if any slipped through
    keepers = keepers[~keepers["category"].isin(["command_word", "level_descriptor"])].reset_index(
        drop=True
    )

    start = 0
    if "pair_id" in keepers.columns:
        for pid in keepers["pair_id"].astype(str):
            m = re.search(r"(\d+)$", pid)
            if m:
                start = max(start, int(m.group(1)) + 1)
    else:
        start = len(keepers)

    new_df = propose_draft_pairs(
        repo_root,
        n_new=n_new,
        keepers=keepers,
        start_pair_index=start,
    )
    # Align columns
    for col in keepers.columns:
        if col not in new_df.columns:
            new_df[col] = "" if col != "reviewed" else False
    for col in new_df.columns:
        if col not in keepers.columns:
            keepers[col] = "" if col != "reviewed" else True
    cols = list(dict.fromkeys(list(keepers.columns) + list(new_df.columns)))
    combined = pd.concat([keepers[cols], new_df[cols]], ignore_index=True)
    path = write_draft_pairs(repo_root, combined)
    return _summary(
        combined,
        path,
        message=(
            f"Appended {len(new_df)} new candidates (reviewed=false) to {len(keepers)} keepers. "
            "Review the new rows to clear ≥20."
        ),
        extra={"n_keepers": int(len(keepers)), "n_new": int(len(new_df))},
    )


def _summary(
    df: pd.DataFrame,
    path: Path,
    *,
    message: str,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    by_cat_neg: dict[str, Any] = {}
    if len(df):
        ct = df.groupby(["category", "negative_type"]).size().to_dict()
        by_cat_neg = {f"{c}|{n}": int(v) for (c, n), v in ct.items()}
    out: dict[str, Any] = {
        "ok": True,
        "n_pairs": int(len(df)),
        "by_category": df["category"].value_counts().to_dict() if len(df) else {},
        "by_negative_type": df["negative_type"].value_counts().to_dict() if len(df) else {},
        "by_category_negative_type": by_cat_neg,
        "n_reviewed_true": int(df["reviewed"].astype(bool).sum()) if len(df) else 0,
        "path": str(path),
        "message": message,
    }
    if extra:
        out.update(extra)
    return out


def validate_pairs_for_promote(df: pd.DataFrame) -> list[str]:
    """Light checks for promote. Same-trial unrelated is allowed (hard negatives)."""
    errors: list[str] = []
    for i, row in df.iterrows():
        pid = row.get("pair_id", i)
        for col in ("anchor_text", "related_text", "unrelated_text"):
            if not str(row.get(col) or "").strip():
                errors.append(f"{pid}: empty {col}")
        if str(row.get("related_text")).strip() == str(row.get("unrelated_text")).strip():
            errors.append(f"{pid}: related_text == unrelated_text")
        if _norm_text(str(row.get("anchor_text"))) == _norm_text(str(row.get("related_text"))):
            errors.append(f"{pid}: anchor_text == related_text")
        nt = row.get("negative_type")
        if nt is not None and not (isinstance(nt, float) and np.isnan(nt)):
            if str(nt) not in VALID_NEGATIVE_TYPES:
                errors.append(f"{pid}: invalid negative_type={nt!r}")
    return errors


def promote_reviewed_pairs(repo_root: Path) -> dict[str, Any]:
    """Copy draft rows with reviewed=true into artifacts/encoder_eval_pairs_v1.parquet."""
    cfg = OmegaConf.load(repo_root / "configs" / "encoder_selection.yaml")
    draft_path = repo_root / str(cfg.paths.draft_pairs)
    if not draft_path.is_file():
        return {"ok": False, "message": f"missing draft {draft_path}"}
    df = load_pair_table(draft_path)
    kept = reviewed_triples(df)
    errors = validate_pairs_for_promote(kept)
    if errors:
        return {"ok": False, "n_reviewed": int(len(kept)), "errors": errors[:20]}
    if len(kept) < int(cfg.min_reviewed_triples):
        return {
            "ok": False,
            "n_reviewed": int(len(kept)),
            "message": f"need ≥{cfg.min_reviewed_triples} reviewed triples; have {len(kept)}",
        }
    out_path = repo_root / str(cfg.paths.reviewed_pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(out_path, index=False)
    kept.to_csv(out_path.with_suffix(".csv"), index=False, encoding="utf-8", lineterminator="\n")
    return {
        "ok": True,
        "n_reviewed": int(len(kept)),
        "by_negative_type": kept["negative_type"].value_counts().to_dict(),
        "path": str(out_path),
        "note": (
            "negative_type re-derived from anchor_trial vs unrelated_trial where present; "
            "no hard/easy mix quota enforced."
        ),
    }
