"""M6 predictive metrics on a frozen checkpoint (grouped-val)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Subset

from src.data.episode_dataset import LazyRealEpisodeDataset, collate_episodes, discover_real_episodes
from src.data.splits import grouped_participant_folds
from src.data.targets import RELATION_NAME_TO_IDX
from src.train.loop import build_behaviour_model
from src.train.relation_weights import resolve_clipped_from_train_cfg
from src.utils import io as uio


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _transition_mask(mask: torch.Tensor) -> torch.Tensor:
    """Valid next-step positions (both t and t+1 real)."""
    step_ok = mask.clone()
    if step_ok.size(1) > 1:
        step_ok[:, :-1] = mask[:, :-1] & mask[:, 1:]
    step_ok[:, -1] = False
    return step_ok


def average_precision_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or y_true.sum() == 0:
        return float("nan")
    if y_true.sum() == len(y_true):
        return 1.0
    return float(average_precision_score(y_true, y_score))


def ranking_metrics_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    cand_mask: np.ndarray,
) -> dict[str, float]:
    """MRR and hits@{1,3,5} over steps with a positive candidate."""
    mrrs: list[float] = []
    hits = {1: [], 3: [], 5: []}
    n = scores.shape[0]
    for i in range(n):
        m = cand_mask[i].astype(bool)
        lab = labels[i]
        if not m.any() or not (lab[m] > 0.5).any():
            continue
        sc = scores[i].copy()
        sc[~m] = -1e9
        order = np.argsort(-sc)
        # rank of first positive (1-based)
        rank = None
        for r, j in enumerate(order, start=1):
            if lab[j] > 0.5 and m[j]:
                rank = r
                break
        if rank is None:
            continue
        mrrs.append(1.0 / rank)
        for k in hits:
            hits[k].append(1.0 if rank <= k else 0.0)
    out = {
        "n_ranked_steps": float(len(mrrs)),
        "mrr": float(np.mean(mrrs)) if mrrs else float("nan"),
    }
    for k, vals in hits.items():
        out[f"hits@{k}"] = float(np.mean(vals)) if vals else float("nan")
    return out


def _cosine_rank_scores(
    batch: dict[str, Any],
    node_dim: int,
) -> torch.Tensor:
    """Feature-only baseline: cosine(query x_v, cand x_v); empty query → 0."""
    tokens = batch["tokens"]
    b, t, _ = tokens.shape
    q = tokens[:, :, :node_dim]
    cand_idx = batch["rank_candidates"]
    _, _, c = cand_idx.shape
    n_nodes = batch["node_x_v"].size(1)
    safe = cand_idx.clamp(min=0)
    batch_ix = torch.arange(b, device=tokens.device).view(b, 1, 1).expand(b, t, c)
    cand_x = batch["node_x_v"][batch_ix, safe]
    valid = batch["rank_mask"] & (cand_idx >= 0) & (cand_idx < n_nodes)
    qn = torch.nn.functional.normalize(q, dim=-1).unsqueeze(2)
    cn = torch.nn.functional.normalize(cand_x, dim=-1)
    scores = (qn * cn).sum(dim=-1)
    # Empty-space / invalid query nodes → zero score
    empty = batch["is_empty"] | (batch["node_index"] < 0)
    scores = scores.masked_fill(empty.unsqueeze(-1), 0.0)
    neg = torch.finfo(scores.dtype).min / 2
    scores = scores.masked_fill(~valid, neg)
    return scores


@torch.no_grad()
def collect_val_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    active_labels: Sequence[str],
    device: torch.device,
    node_dim: int,
    train_next_node_freq: Optional[dict[int, float]] = None,
) -> dict[str, Any]:
    model.eval()
    active_idx = [RELATION_NAME_TO_IDX[n] for n in active_labels]
    rel_true: list[np.ndarray] = []
    rel_score: list[np.ndarray] = []
    panel_true: list[np.ndarray] = []
    panel_pred: list[np.ndarray] = []
    rank_model_s: list[np.ndarray] = []
    rank_freq_s: list[np.ndarray] = []
    rank_feat_s: list[np.ndarray] = []
    rank_lab: list[np.ndarray] = []
    rank_msk: list[np.ndarray] = []

    for batch in loader:
        batch = _batch_to_device(batch, device)
        out = model(batch)
        step_ok = _transition_mask(batch["mask"])

        # Relations
        logits = out["relation_logits"]
        probs = torch.sigmoid(logits)
        tgt = batch["next_relation"][:, :, active_idx]
        m = step_ok.unsqueeze(-1).expand_as(probs)
        rel_score.append(probs[m].view(-1, probs.size(-1)).cpu().numpy())
        rel_true.append(tgt[m].view(-1, tgt.size(-1)).cpu().numpy())

        # Panels
        panel_logits = out["panel_logits"]
        pred = panel_logits.argmax(dim=-1)
        pt = batch["next_panel"]
        ok = step_ok & (pt >= 0)
        panel_true.append(pt[ok].cpu().numpy())
        panel_pred.append(pred[ok].cpu().numpy())

        # Ranking — model
        rs = out["rank_scores"]
        lab = batch["rank_labels"]
        cm = batch["rank_mask"] & (batch["rank_candidates"] >= 0)
        # Only steps with a positive
        has_pos = ((lab > 0.5) & cm).any(dim=-1) & step_ok
        if has_pos.any():
            rank_model_s.append(rs[has_pos].cpu().numpy())
            rank_lab.append(lab[has_pos].cpu().numpy())
            rank_msk.append(cm[has_pos].cpu().numpy())

            # Frequency baseline scores
            cand = batch["rank_candidates"][has_pos].cpu().numpy()
            freq_scores = np.zeros_like(rs[has_pos].cpu().numpy(), dtype=np.float32)
            freqs = train_next_node_freq or {}
            for i in range(cand.shape[0]):
                for j in range(cand.shape[1]):
                    ni = int(cand[i, j])
                    freq_scores[i, j] = float(freqs.get(ni, 0.0)) if ni >= 0 else -1e9
            rank_freq_s.append(freq_scores)

            # Feature-only cosine
            feat = _cosine_rank_scores(batch, node_dim)
            rank_feat_s.append(feat[has_pos].cpu().numpy())

    return {
        "rel_true": np.concatenate(rel_true, axis=0) if rel_true else np.zeros((0, len(active_labels))),
        "rel_score": np.concatenate(rel_score, axis=0) if rel_score else np.zeros((0, len(active_labels))),
        "panel_true": np.concatenate(panel_true) if panel_true else np.zeros((0,), dtype=np.int64),
        "panel_pred": np.concatenate(panel_pred) if panel_pred else np.zeros((0,), dtype=np.int64),
        "rank_model_scores": np.concatenate(rank_model_s, axis=0) if rank_model_s else np.zeros((0, 1)),
        "rank_freq_scores": np.concatenate(rank_freq_s, axis=0) if rank_freq_s else np.zeros((0, 1)),
        "rank_feat_scores": np.concatenate(rank_feat_s, axis=0) if rank_feat_s else np.zeros((0, 1)),
        "rank_labels": np.concatenate(rank_lab, axis=0) if rank_lab else np.zeros((0, 1)),
        "rank_mask": np.concatenate(rank_msk, axis=0) if rank_msk else np.zeros((0, 1), dtype=bool),
        "active_labels": list(active_labels),
    }


def estimate_train_next_node_freq(
    loader: DataLoader,
    *,
    max_batches: int = 200,
) -> dict[int, float]:
    """Empirical P(next node) on train fold (for frequency ranking baseline)."""
    counts: dict[int, int] = defaultdict(int)
    total = 0
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        lab = batch["rank_labels"]
        cand = batch["rank_candidates"]
        cm = batch["rank_mask"]
        has_pos = (lab > 0.5) & cm
        pos_idx = has_pos.float().argmax(dim=-1)
        b, t = has_pos.shape[:2]
        for i in range(b):
            for j in range(t):
                if not has_pos[i, j].any():
                    continue
                ni = int(cand[i, j, int(pos_idx[i, j])])
                if ni >= 0:
                    counts[ni] += 1
                    total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def summarise_predictions(
    preds: dict[str, Any],
    *,
    panel_classes: Sequence[str],
    operating_threshold: float = 0.5,
) -> dict[str, Any]:
    labels = list(preds["active_labels"])
    y_true = preds["rel_true"]
    y_score = preds["rel_score"]
    n = int(y_true.shape[0])
    per_label = []
    semantic_flag = None
    for i, name in enumerate(labels):
        yt = y_true[:, i].astype(np.float64)
        ys = y_score[:, i].astype(np.float64)
        pos = int(yt.sum())
        base_rate = float(pos / n) if n else float("nan")
        ap = average_precision_safe(yt, ys)
        # Constant predictor AP ≡ base rate (sklearn tie behaviour ≈ prevalence)
        ap_baseline = base_rate
        yhat = (ys >= operating_threshold).astype(np.int64)
        prec = float(precision_score(yt, yhat, zero_division=0)) if pos else float("nan")
        rec = float(recall_score(yt, yhat, zero_division=0)) if pos else float("nan")
        row = {
            "relation": name,
            "n_pos": pos,
            "n_steps": n,
            "base_rate": round(base_rate, 6),
            "ap": None if np.isnan(ap) else round(ap, 6),
            "ap_baseline": round(ap_baseline, 6),
            "ap_minus_baseline": None if np.isnan(ap) else round(ap - ap_baseline, 6),
            "precision@thr": None if np.isnan(prec) else round(prec, 6),
            "recall@thr": None if np.isnan(rec) else round(rec, 6),
            "threshold": operating_threshold,
        }
        if name == "SEMANTIC_CANDIDATE":
            semantic_flag = {
                "ap": row["ap"],
                "ap_baseline": row["ap_baseline"],
                "at_or_below_baseline": bool(np.isnan(ap) or ap <= ap_baseline + 1e-12),
            }
        per_label.append(row)

    ranking = {
        "model": ranking_metrics_from_scores(
            preds["rank_model_scores"], preds["rank_labels"], preds["rank_mask"]
        ),
        "transition_frequency_baseline": ranking_metrics_from_scores(
            preds["rank_freq_scores"], preds["rank_labels"], preds["rank_mask"]
        ),
        "feature_only_cosine_probe": ranking_metrics_from_scores(
            preds["rank_feat_scores"], preds["rank_labels"], preds["rank_mask"]
        ),
    }

    pt = preds["panel_true"]
    pp = preds["panel_pred"]
    n_cls = len(panel_classes)
    if pt.size:
        labels_present = sorted(set(int(x) for x in pt.tolist()))
        cm = confusion_matrix(pt, pp, labels=list(range(n_cls))).tolist()
        f1_per = f1_score(pt, pp, labels=list(range(n_cls)), average=None, zero_division=0)
        # Macro-F1 over classes with ≥1 true instance (exclude zero-support, e.g. ui)
        macro_f1_supported = float(
            f1_score(pt, pp, labels=labels_present, average="macro", zero_division=0)
        )
        weighted_f1 = float(f1_score(pt, pp, average="weighted", zero_division=0))
        panel = {
            "n": int(pt.size),
            "macro_f1": macro_f1_supported,
            "macro_f1_all_classes": float(
                f1_score(pt, pp, labels=list(range(n_cls)), average="macro", zero_division=0)
            ),
            "weighted_f1": weighted_f1,
            "accuracy": float((pt == pp).mean()),
            "supported_classes": [panel_classes[i] for i in labels_present],
            "per_class_f1": {
                panel_classes[i]: round(float(f1_per[i]), 6) for i in range(n_cls)
            },
            "per_class_support": {
                panel_classes[i]: int((pt == i).sum()) for i in range(n_cls)
            },
            "confusion_matrix": cm,
            "classes": list(panel_classes),
        }
    else:
        panel = {"n": 0, "macro_f1": float("nan"), "accuracy": float("nan")}

    go = semantic_flag is not None and not semantic_flag["at_or_below_baseline"]
    return {
        "n_relation_steps": n,
        "relation_per_label": per_label,
        "semantic_candidate_gate": semantic_flag,
        "go_nogo": "GO" if go else "NO-GO",
        "ranking": ranking,
        "next_panel": panel,
        "operating_threshold": operating_threshold,
    }


def build_fold_loaders(
    repo: Path,
    *,
    fold: int,
    seed: int,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    repo = Path(repo)
    dcfg = OmegaConf.load(repo / "configs" / "dataset.yaml")
    tcfg = OmegaConf.load(repo / "configs" / "model_transformer.yaml")
    scfg = OmegaConf.load(repo / "configs" / "splits.yaml")
    dcfg.max_seq_len = int(tcfg.max_seq_len)
    dcfg.build_pair_relations = bool(tcfg.biases.graph_relation.enabled)
    triples = discover_real_episodes(repo / str(dcfg.paths.fixations_root))
    graphs_root = repo / str(dcfg.paths.graphs_root) / str(dcfg.graph_version)
    keys: list[tuple[str, str, str]] = []
    for pid, tid, sc in triples:
        if (graphs_root / f"{tid}__{sc}.pt").is_file() and (
            repo / str(dcfg.paths.fixations_root) / pid / f"{tid}__{sc}.parquet"
        ).is_file():
            keys.append((pid, tid, sc))
    folds = grouped_participant_folds(keys, n_folds=int(scfg.n_folds), seed=int(scfg.seed))
    fold_info = folds[fold]
    ds = LazyRealEpisodeDataset(
        keys,
        dataset_cfg=dcfg,
        fixations_root=repo / str(dcfg.paths.fixations_root),
        graphs_root=repo / str(dcfg.paths.graphs_root),
        graph_version=str(dcfg.graph_version),
        gnn=None,
        seed=seed,
    )
    train_loader = DataLoader(
        Subset(ds, fold_info["train_idx"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
    )
    val_loader = DataLoader(
        Subset(ds, fold_info["val_idx"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
    )
    meta = {
        "fold": fold,
        "seed": seed,
        "train_participants": fold_info["train_participants"],
        "val_participants": fold_info["val_participants"],
        "n_train": len(fold_info["train_idx"]),
        "n_val": len(fold_info["val_idx"]),
        "panel_classes": list(dcfg.panel_classes),
        "gnn_out_dim": int(dcfg.gnn_out_dim),
    }
    return train_loader, val_loader, meta


def evaluate_checkpoint(
    repo: Path,
    checkpoint: Path,
    *,
    fold: int = 0,
    seed: int = 13,
    batch_size: int = 8,
    device: Optional[torch.device] = None,
    operating_threshold: float = 0.5,
) -> dict[str, Any]:
    repo = Path(repo)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = OmegaConf.load(repo / "configs" / "train.yaml")
    active = list(train_cfg.relation_weights.active_labels)
    train_loader, val_loader, meta = build_fold_loaders(
        repo, fold=fold, seed=seed, batch_size=batch_size
    )
    print("Estimating train next-node frequencies (ranking baseline)...", flush=True)
    freq = estimate_train_next_node_freq(train_loader)
    print(f"  unique next-nodes with mass: {len(freq)}", flush=True)

    model = build_behaviour_model(repo, device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Evaluating {checkpoint} on fold {fold} val ({meta['n_val']} eps)...", flush=True)
    preds = collect_val_predictions(
        model,
        val_loader,
        active_labels=active,
        device=device,
        node_dim=int(meta["gnn_out_dim"]),
        train_next_node_freq=freq,
    )
    summary = summarise_predictions(
        preds,
        panel_classes=meta["panel_classes"],
        operating_threshold=operating_threshold,
    )
    summary["checkpoint"] = str(checkpoint)
    summary["fold_meta"] = meta
    summary["relation_weights"] = resolve_clipped_from_train_cfg(train_cfg, repo)
    return summary


def write_eval_report(summary: dict[str, Any], out_json: Path, out_md: Path) -> None:
    uio.write_json(out_json, summary)
    gate = summary.get("semantic_candidate_gate") or {}
    lines = [
        "# M6 predictive eval (grouped-val)",
        "",
        f"- Checkpoint: `{summary.get('checkpoint')}`",
        f"- Fold: **{summary['fold_meta']['fold']}** · seed used for dataset RNG: "
        f"{summary['fold_meta']['seed']}",
        f"- Val participants: {', '.join(summary['fold_meta']['val_participants'])}",
        f"- Relation steps scored: **{summary['n_relation_steps']}**",
        f"- Operating threshold (sigmoid): **{summary['operating_threshold']}**",
        f"- **Go/no-go: {summary['go_nogo']}** "
        f"(SEMANTIC_CANDIDATE AP={gate.get('ap')} vs baseline={gate.get('ap_baseline')})",
        "",
        "## Per-label next-relation",
        "",
        "| relation | n_pos | base_rate | AP | AP baseline | Δ | P@thr | R@thr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["relation_per_label"]:
        lines.append(
            f"| `{r['relation']}` | {r['n_pos']} | {r['base_rate']:.4f} | "
            f"{r['ap']} | {r['ap_baseline']:.4f} | {r['ap_minus_baseline']} | "
            f"{r['precision@thr']} | {r['recall@thr']} |"
        )
    lines.extend(["", "## Ranking", ""])
    for name, m in summary["ranking"].items():
        lines.append(
            f"- **{name}**: MRR={m.get('mrr'):.4f} · "
            f"hits@1={m.get('hits@1'):.4f} · hits@3={m.get('hits@3'):.4f} · "
            f"hits@5={m.get('hits@5'):.4f} · n={int(m.get('n_ranked_steps', 0))}"
        )
    panel = summary["next_panel"]
    lines.extend(
        [
            "",
            "## Next-panel",
            "",
            f"- n={panel.get('n')} · accuracy={panel.get('accuracy'):.4f} · "
            f"macro-F1 (supported)={panel.get('macro_f1'):.4f} · "
            f"weighted-F1={panel.get('weighted_f1', float('nan')):.4f}",
            "",
            f"Supported classes: {', '.join(panel.get('supported_classes') or [])}",
            "",
            "Per-class F1 (support):",
            "",
        ]
    )
    for k, v in (panel.get("per_class_f1") or {}).items():
        supp = (panel.get("per_class_support") or {}).get(k, "?")
        lines.append(f"- `{k}`: {v:.4f} (n={supp})")
    lines.extend(["", "Confusion matrix (rows=true, cols=pred):", "", "```"])
    cm = panel.get("confusion_matrix") or []
    classes = panel.get("classes") or []
    if classes:
        lines.append("     " + " ".join(f"{c[:4]:>5}" for c in classes))
    for i, row in enumerate(cm):
        lab = classes[i][:4] if i < len(classes) else str(i)
        lines.append(f"{lab:>4} " + " ".join(f"{x:5d}" for x in row))
    lines.append("```")
    lines.append("")
    uio.write_text(out_md, "\n".join(lines) + "\n")
