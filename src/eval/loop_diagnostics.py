"""M7 pre-registered loop-diagnostic gate (D1–D3) on frozen M6 embeddings."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.eval.m6_predictive import build_fold_loaders
from src.train.loop import build_behaviour_model
from src.utils import io as uio


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _probe() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def return_within_horizon_labels(
    node_index: np.ndarray,
    *,
    horizon: int,
) -> np.ndarray:
    """Label[t]=1 if same segment reappears in (t, t+horizon]; -1 = ignore."""
    t_len = int(node_index.shape[0])
    lab = np.full(t_len, -1, dtype=np.int64)
    for t in range(t_len):
        sid = int(node_index[t])
        if sid < 0:
            continue
        end = min(t_len, t + 1 + int(horizon))
        if t + 1 >= end:
            continue
        lab[t] = 1 if np.any(node_index[t + 1 : end] == sid) else 0
    return lab


@torch.no_grad()
def collect_episode_arrays(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    node_dim: int,
) -> list[dict[str, Any]]:
    """Per-episode embeddings, token side features, node index, panel, templates."""
    model.eval()
    episodes: list[dict[str, Any]] = []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        y = model.encode(batch)  # [B,T,D]
        tokens = batch["tokens"]
        bsz = int(tokens.size(0))
        lengths = batch["lengths"].detach().cpu().numpy().astype(int)
        for i in range(bsz):
            L = int(lengths[i])
            if L <= 1:
                continue
            side = tokens[i, :L, 2 * node_dim :].detach().cpu().numpy()
            xv = tokens[i, :L, :node_dim].detach().cpu().numpy()
            emb = y[i, :L].detach().cpu().numpy()
            node_index = batch["node_index"][i, :L].detach().cpu().numpy().astype(int)
            panel = batch["panel_id"][i, :L].detach().cpu().numpy().astype(int)
            meta = {
                "participant_id": (
                    batch["participant_id"][i]
                    if isinstance(batch.get("participant_id"), list)
                    else None
                ),
                "trial_id": (
                    batch["trial_id"][i] if isinstance(batch.get("trial_id"), list) else None
                ),
                "star_condition": (
                    batch["star_condition"][i]
                    if isinstance(batch.get("star_condition"), list)
                    else None
                ),
            }
            episodes.append(
                {
                    "emb": emb.astype(np.float32),
                    "feat": np.concatenate([xv, side], axis=-1).astype(np.float32),
                    "node_index": node_index,
                    "panel_id": panel,
                    "participant_id": meta.get("participant_id"),
                    "trial_id": meta.get("trial_id"),
                    "star_condition": meta.get("star_condition"),
                }
            )
    return episodes


def _attach_loop_templates_from_panels(
    episodes: list[dict[str, Any]],
    *,
    panel_classes: Sequence[str],
    templates: list[list[str]],
    max_loop_gap: int = 20,
    fixations_root: Optional[Path] = None,
) -> dict[str, int]:
    """Detect loop templates; prefer full fixation rows (segment_role) from disk.

    Reconstructing from ``panel_id`` alone drops ``segment_role``, which collapses
    ``mark_scheme_level_descriptor`` into plain ``mark_scheme`` and zeros the
    LoR-specific template. When ``fixations_root`` is set, reload parquet rows.
    """
    from src.data.loops import annotate_loops
    from src.utils.arrow_cuda import read_parquet

    corpus_counts: Counter[str] = Counter()
    id_to_panel = {i: str(p) for i, p in enumerate(panel_classes)}
    for ep in episodes:
        sc = str(ep.get("star_condition") or "not_eligible")
        fixes = None
        if fixations_root is not None and ep.get("participant_id") and ep.get("trial_id"):
            pq = (
                Path(fixations_root)
                / str(ep["participant_id"])
                / f"{ep['trial_id']}__{sc}.parquet"
            )
            if pq.is_file():
                df = read_parquet(pq)
                fixes = df.to_dict("records")
                # Align length to truncated episode if needed
                T = len(ep["panel_id"])
                if len(fixes) > T:
                    fixes = fixes[:T]
        if fixes is None:
            fixes = []
            for t in range(len(ep["panel_id"])):
                pid = int(ep["panel_id"][t])
                fixes.append(
                    {
                        "panel_label": id_to_panel.get(pid, "unknown"),
                        "segment_id": str(ep["node_index"][t])
                        if int(ep["node_index"][t]) >= 0
                        else None,
                    }
                )
        annotated, counts = annotate_loops(
            fixes,
            templates=templates,
            max_loop_gap_events=max_loop_gap,
            star_condition=sc,
        )
        for k, v in counts.items():
            corpus_counts[k] += int(v)
        primary = []
        for row in annotated:
            tid = str(row.get("loop_template_id") or "")
            primary.append(tid.split("|")[0] if tid else "")
        ep["loop_template_primary"] = primary
        ep["loop_role"] = [str(r.get("loop_role") or "none") for r in annotated]
        ep["is_return"] = [bool(r.get("is_return")) for r in annotated]
    return dict(corpus_counts)


def run_d1(
    train_eps: list[dict[str, Any]],
    val_eps: list[dict[str, Any]],
    *,
    horizon: int,
    min_margin: float,
) -> dict[str, Any]:
    def pack(eps: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xs_e, Xs_f, ys = [], [], []
        for ep in eps:
            lab = return_within_horizon_labels(ep["node_index"], horizon=horizon)
            ok = lab >= 0
            if not ok.any():
                continue
            Xs_e.append(ep["emb"][ok])
            Xs_f.append(ep["feat"][ok])
            ys.append(lab[ok])
        if not ys:
            z = np.zeros((0, 1))
            return z, z, np.zeros((0,), dtype=int)
        return (
            np.concatenate(Xs_e, axis=0),
            np.concatenate(Xs_f, axis=0),
            np.concatenate(ys, axis=0).astype(int),
        )

    Xe_tr, Xf_tr, y_tr = pack(train_eps)
    Xe_va, Xf_va, y_va = pack(val_eps)
    out: dict[str, Any] = {
        "horizon_events": horizon,
        "n_train": int(y_tr.size),
        "n_val": int(y_va.size),
        "val_pos_rate": float(y_va.mean()) if y_va.size else float("nan"),
    }
    if y_tr.size == 0 or y_va.size == 0 or len(np.unique(y_tr)) < 2:
        out.update(
            {
                "auc_embedding": float("nan"),
                "auc_feature_only": float("nan"),
                "margin": float("nan"),
                "pass": False,
                "note": "insufficient labels",
            }
        )
        return out

    pe, pf = _probe(), _probe()
    pe.fit(Xe_tr, y_tr)
    pf.fit(Xf_tr, y_tr)
    score_e = pe.predict_proba(Xe_va)[:, 1]
    score_f = pf.predict_proba(Xf_va)[:, 1]
    auc_e = _safe_auc(y_va, score_e)
    auc_f = _safe_auc(y_va, score_f)
    margin = auc_e - auc_f
    fpr_e, tpr_e, _ = roc_curve(y_va, score_e)
    fpr_f, tpr_f, _ = roc_curve(y_va, score_f)
    out.update(
        {
            "auc_embedding": auc_e,
            "auc_feature_only": auc_f,
            "margin": margin,
            "min_margin": min_margin,
            "pass": bool(margin >= min_margin - 1e-12),
            "roc_embedding": {"fpr": fpr_e.tolist(), "tpr": tpr_e.tolist()},
            "roc_feature_only": {"fpr": fpr_f.tolist(), "tpr": tpr_f.tolist()},
        }
    )
    return out


def run_d2(
    train_eps: list[dict[str, Any]],
    val_eps: list[dict[str, Any]],
    *,
    active_templates: Sequence[str],
    min_margin: float,
    seed: int = 13,
) -> dict[str, Any]:
    """Multinomial template probe vs within-episode label shuffle."""
    label_names = ["none"] + list(active_templates)
    name_to_i = {n: i for i, n in enumerate(label_names)}

    def pack(eps: list[dict[str, Any]], shuffle: bool) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        Xs, ys = [], []
        for ep in eps:
            labs = list(ep.get("loop_template_primary") or [""] * len(ep["emb"]))
            # Map unknowns / dropped templates → none
            mapped = []
            for t in labs:
                if t in name_to_i and t != "none":
                    mapped.append(t)
                elif t in active_templates:
                    mapped.append(t)
                else:
                    mapped.append("none")
            if shuffle:
                # Within-episode shuffle of non-pad labels
                mapped = list(rng.permutation(mapped))
            Xs.append(ep["emb"])
            ys.append(np.array([name_to_i[m] for m in mapped], dtype=int))
        if not Xs:
            return np.zeros((0, 1)), np.zeros((0,), dtype=int)
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    X_tr, y_tr = pack(train_eps, shuffle=False)
    X_va, y_va = pack(val_eps, shuffle=False)
    _, y_va_shuf = pack(val_eps, shuffle=True)

    result: dict[str, Any] = {
        "active_templates": list(active_templates),
        "n_train": int(y_tr.size),
        "n_val": int(y_va.size),
        "label_names": label_names,
    }
    if y_tr.size == 0 or len(np.unique(y_tr)) < 2:
        result.update(
            {
                "macro_f1": float("nan"),
                "macro_f1_shuffled": float("nan"),
                "margin": float("nan"),
                "pass": False,
                "note": "insufficient labels",
            }
        )
        return result

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=800,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_va)
    # Shuffled baseline: same features, shuffled labels → chance macro-F1 of predicting true labels?
    # Spec: probe vs within-episode label-shuffled baseline — train on true, evaluate
    # macro-F1 when *labels* are shuffled (feature–label alignment destroyed).
    # Equivalently: score model predictions against shuffled val labels.
    f1 = float(f1_score(y_va, pred, average="macro", zero_division=0))
    f1_shuf = float(f1_score(y_va_shuf, pred, average="macro", zero_division=0))
    # Better baseline: retrain is wrong; use shuffled-label F1 of the same preds
    # as null. Margin = F1_true - F1_shuffled_labels.
    margin = f1 - f1_shuf
    result.update(
        {
            "macro_f1": f1,
            "macro_f1_shuffled": f1_shuf,
            "margin": margin,
            "min_margin": min_margin,
            "pass": bool(margin >= min_margin - 1e-12),
            "val_label_counts": {label_names[i]: int((y_va == i).sum()) for i in range(len(label_names))},
        }
    )
    return result


def run_d3(
    train_eps: list[dict[str, Any]],
    val_eps: list[dict[str, Any]],
    *,
    window: int,
    n_neg: int,
    seed: int = 13,
) -> dict[str, Any]:
    """Binary probe: true window vs locally shuffled window (flattened emb)."""

    def make_windows(eps: list[dict[str, Any]], rng: np.random.Generator):
        X, y = [], []
        for ep in eps:
            emb = ep["emb"]
            T = emb.shape[0]
            if T < window:
                continue
            for start in range(0, T - window + 1, max(1, window // 2)):
                w = emb[start : start + window]
                X.append(w.reshape(-1))
                y.append(1)
                for _ in range(n_neg):
                    perm = rng.permutation(window)
                    X.append(w[perm].reshape(-1))
                    y.append(0)
        if not X:
            return np.zeros((0, 1)), np.zeros((0,), dtype=int)
        return np.stack(X, axis=0), np.asarray(y, dtype=int)

    rng_tr = np.random.default_rng(seed)
    rng_va = np.random.default_rng(seed + 1)
    X_tr, y_tr = make_windows(train_eps, rng_tr)
    X_va, y_va = make_windows(val_eps, rng_va)
    out: dict[str, Any] = {
        "window": window,
        "n_negatives_per_positive": n_neg,
        "n_train": int(y_tr.size),
        "n_val": int(y_va.size),
    }
    if y_tr.size == 0 or len(np.unique(y_tr)) < 2:
        out.update({"auc": float("nan"), "accuracy": float("nan"), "pass": False})
        return out
    clf = _probe()
    clf.fit(X_tr, y_tr)
    score = clf.predict_proba(X_va)[:, 1]
    pred = (score >= 0.5).astype(int)
    auc = _safe_auc(y_va, score)
    acc = float(accuracy_score(y_va, pred))
    # Pre-registered: D3 enabled → pass if AUC > 0.55 (weak but above chance)
    # Threshold not in config historically — use chance+margin style: AUC >= 0.55
    out.update(
        {
            "auc": auc,
            "accuracy": acc,
            "pass": bool(auc >= 0.55),
            "pass_rule": "auc >= 0.55 (above-chance local-order sensitivity)",
        }
    )
    return out


def fixation_vs_visit_table(
    episodes: list[dict[str, Any]],
    *,
    horizon: int,
) -> dict[str, Any]:
    """Compare return-probe label rates / mean emb norms at fixation vs visit ends."""
    fix_n = visit_n = 0
    fix_ret = visit_ret = 0
    for ep in episodes:
        ni = ep["node_index"]
        lab = return_within_horizon_labels(ni, horizon=horizon)
        ok = lab >= 0
        fix_n += int(ok.sum())
        fix_ret += int((lab[ok] == 1).sum())
        # Visit ends: last index of each run of identical node_index
        T = len(ni)
        ends = []
        t = 0
        while t < T:
            j = t + 1
            while j < T and ni[j] == ni[t]:
                j += 1
            ends.append(j - 1)
            t = j
        for e in ends:
            if lab[e] >= 0:
                visit_n += 1
                visit_ret += int(lab[e] == 1)
    return {
        "fixation": {
            "n": fix_n,
            "return_within_h_rate": (fix_ret / fix_n) if fix_n else float("nan"),
        },
        "visit_boundary": {
            "n": visit_n,
            "return_within_h_rate": (visit_ret / visit_n) if visit_n else float("nan"),
        },
        "note": (
            "Visit-boundary = last fixation of each contiguous same-segment run. "
            "Full visit-token retrain is ablation #6; this table is the M7 diagnostic slice."
        ),
    }


def write_roc_svg(d1: dict[str, Any], out_path: Path) -> Path:
    """Minimal SVG ROC: embedding vs feature-only with margin annotation."""
    re = d1.get("roc_embedding") or {}
    rf = d1.get("roc_feature_only") or {}

    def poly(fpr, tpr, color: str) -> str:
        if not fpr:
            return ""
        pts = " ".join(
            f"{40 + 300 * float(x):.1f},{340 - 300 * float(y):.1f}"
            for x, y in zip(fpr, tpr)
        )
        return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}" />'

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="420" height="400">
  <rect x="40" y="40" width="300" height="300" fill="#fafafa" stroke="#ccc"/>
  <line x1="40" y1="340" x2="340" y2="40" stroke="#ddd" stroke-dasharray="4"/>
  {poly(re.get("fpr") or [], re.get("tpr") or [], "#1f77b4")}
  {poly(rf.get("fpr") or [], rf.get("tpr") or [], "#ff7f0e")}
  <text x="40" y="24" font-family="Georgia" font-size="14">D1 return probe ROC</text>
  <text x="40" y="370" font-family="Georgia" font-size="12">
    emb AUC={d1.get("auc_embedding"):.3f} · feat AUC={d1.get("auc_feature_only"):.3f} ·
    margin={d1.get("margin"):.3f} (need ≥ {d1.get("min_margin")})
  </text>
  <text x="50" y="60" font-family="Georgia" font-size="11" fill="#1f77b4">embedding</text>
  <text x="50" y="76" font-family="Georgia" font-size="11" fill="#ff7f0e">feature-only</text>
</svg>
"""
    out_path = Path(out_path)
    uio.write_text(out_path, svg)
    return out_path


def run_m7_gate(
    repo: Path,
    checkpoint: Path,
    *,
    fold: int = 0,
    seed: int = 13,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    repo = Path(repo)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = OmegaConf.load(repo / "configs" / "train.yaml")
    pre_cfg = OmegaConf.load(repo / "configs" / "preprocessing.yaml")
    dcfg = OmegaConf.load(repo / "configs" / "dataset.yaml")
    diag = train_cfg.diagnostics

    train_loader, val_loader, meta = build_fold_loaders(
        repo, fold=fold, seed=seed, batch_size=batch_size
    )
    model = build_behaviour_model(repo, device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    print("Collecting train embeddings...", flush=True)
    train_eps = collect_episode_arrays(
        model, train_loader, device=device, node_dim=int(meta["gnn_out_dim"])
    )
    print("Collecting val embeddings...", flush=True)
    val_eps = collect_episode_arrays(
        model, val_loader, device=device, node_dim=int(meta["gnn_out_dim"])
    )

    templates = [list(t) for t in list(pre_cfg.loops.templates)]
    print("Attaching loop templates...", flush=True)
    fix_root = repo / str(dcfg.paths.fixations_root)
    corpus_counts = _attach_loop_templates_from_panels(
        train_eps + val_eps,
        panel_classes=meta["panel_classes"],
        templates=templates,
        max_loop_gap=int(pre_cfg.loops.max_loop_gap_events),
        fixations_root=fix_root,
    )
    min_count = int(diag.D2_loop_template_probe.min_template_corpus_count)
    active = []
    dropped = []
    for t in templates:
        tid = "→".join(t)
        c = int(corpus_counts.get(tid, 0))
        if c >= min_count:
            active.append(tid)
        else:
            dropped.append({"template": tid, "count": c})

    print("D1 return probe...", flush=True)
    d1 = run_d1(
        train_eps,
        val_eps,
        horizon=int(diag.D1_return_probe.horizon_events),
        min_margin=float(diag.D1_return_probe.min_auc_margin_over_feature_only),
    )
    print("D2 loop-template probe...", flush=True)
    d2 = run_d2(
        train_eps,
        val_eps,
        active_templates=active,
        min_margin=float(diag.D2_loop_template_probe.min_macro_f1_margin),
        seed=seed,
    )
    d2["corpus_counts"] = corpus_counts
    d2["dropped_templates"] = dropped

    d3 = {"enabled": False, "pass": True}
    if bool(diag.D3_subsequence.enabled):
        print("D3 subsequence probe...", flush=True)
        d3 = run_d3(
            train_eps,
            val_eps,
            window=int(diag.D3_subsequence.window),
            n_neg=int(diag.D3_subsequence.n_negatives_per_positive),
            seed=seed,
        )
        d3["enabled"] = True

    temporal = {}
    if bool(getattr(diag, "temporal_comparison", {}).get("enabled", True)):
        temporal = fixation_vs_visit_table(
            val_eps, horizon=int(diag.D1_return_probe.horizon_events)
        )

    gates = {"D1": bool(d1.get("pass")), "D2": bool(d2.get("pass")), "D3": bool(d3.get("pass"))}
    all_pass = all(gates.values())
    decision = (
        "PASS — keep return/loop auxiliary losses disabled"
        if all_pass
        else "FAIL — enable return/loop aux losses and retrain (see DECISIONS)"
    )
    return {
        "checkpoint": str(checkpoint),
        "fold": fold,
        "seed": seed,
        "device": str(device),
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "temporal_fixation_vs_visit": temporal,
        "gates": gates,
        "all_pass": all_pass,
        "decision": decision,
        "fold_meta": {
            "val_participants": meta["val_participants"],
            "train_participants": meta["train_participants"],
        },
    }


def write_m7_report(summary: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "m7_diagnostics.json"
    md_path = out_dir / "m7_diagnostics.md"
    uio.write_json(json_path, summary)

    d1, d2, d3 = summary["D1"], summary["D2"], summary["D3"]
    if d1.get("roc_embedding"):
        write_roc_svg(d1, out_dir / "d1_roc.svg")

    lines = [
        "# M7 diagnostic gate (frozen M6 embeddings)",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Fold **{summary['fold']}** · seed **{summary['seed']}**",
        f"- Val participants: {', '.join(summary['fold_meta']['val_participants'])}",
        f"- Episodes: train {summary['n_train_episodes']} / val {summary['n_val_episodes']}",
        "",
        f"## Decision: **{'PASS' if summary['all_pass'] else 'FAIL'}**",
        "",
        summary["decision"],
        "",
        "| gate | pass | metric |",
        "|---|---|---|",
        f"| D1 return | {summary['gates']['D1']} | emb AUC={d1.get('auc_embedding'):.4f} · "
        f"feat={d1.get('auc_feature_only'):.4f} · margin={d1.get('margin'):.4f} "
        f"(need ≥ {d1.get('min_margin')}) |",
        f"| D2 loop template | {summary['gates']['D2']} | macro-F1={d2.get('macro_f1'):.4f} · "
        f"shuffled={d2.get('macro_f1_shuffled'):.4f} · margin={d2.get('margin'):.4f} "
        f"(need ≥ {d2.get('min_margin')}) |",
        f"| D3 subsequence | {summary['gates']['D3']} | AUC={d3.get('auc')} · "
        f"acc={d3.get('accuracy')} |",
        "",
        "## D2 templates",
        "",
        f"Active: {', '.join(f'`{t}`' for t in d2.get('active_templates') or [])}",
        "",
    ]
    if d2.get("dropped_templates"):
        lines.append("Dropped (< min count):")
        for row in d2["dropped_templates"]:
            lines.append(f"- `{row['template']}` count={row['count']}")
        lines.append("")
    temp = summary.get("temporal_fixation_vs_visit") or {}
    if temp:
        lines.extend(
            [
                "## Fixation vs visit (diagnostic slice)",
                "",
                f"- Fixation steps: n={temp.get('fixation', {}).get('n')} · "
                f"return-within-H rate={temp.get('fixation', {}).get('return_within_h_rate')}",
                f"- Visit boundaries: n={temp.get('visit_boundary', {}).get('n')} · "
                f"return-within-H rate={temp.get('visit_boundary', {}).get('return_within_h_rate')}",
                f"- {temp.get('note', '')}",
                "",
            ]
        )
    if (out_dir / "d1_roc.svg").is_file():
        lines.extend(["## D1 ROC", "", "![D1 ROC](d1_roc.svg)", ""])
    uio.write_text(md_path, "\n".join(lines) + "\n")
    return json_path, md_path
