"""D1 ceiling diagnostics on a frozen return_aux checkpoint (no training).

Reports:
1. ReturnHead validation AUC on return-within-H
2. Probe vs head representation alignment
3. Gradient-boosted ceiling on raw token features + history aggregates
4. Confirmation of the shipped balance fix (pos_weight / horizon / pos rate)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in ("-1",):
    del os.environ["CUDA_VISIBLE_DEVICES"]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.arrow_cuda import warmup_parquet_io

_sample = ROOT / "data_processed" / "v0_p0" / "fixations" / "P01" / "T01__not_eligible.parquet"
warmup_parquet_io(_sample if _sample.is_file() else None)

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from src.data.episode_dataset import LazyRealEpisodeDataset, collate_episodes, discover_real_episodes
from src.data.splits import grouped_participant_folds
from src.eval.loop_diagnostics import return_within_horizon_labels
from src.train.loop import build_behaviour_model
from src.utils import io as uio


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _history_aggregates(node_index: np.ndarray) -> np.ndarray:
    """Explicit return-history features per step (no future leakage)."""
    T = len(node_index)
    out = np.zeros((T, 6), dtype=np.float32)
    last_seen: dict[int, int] = {}
    visit_count: dict[int, int] = {}
    for t, nid in enumerate(node_index):
        nid = int(nid)
        if nid < 0:
            continue
        prev = last_seen.get(nid)
        vc = visit_count.get(nid, 0)
        out[t, 0] = float(vc)  # visits so far (before current)
        out[t, 1] = 1.0 if prev is not None else 0.0  # is_return (past)
        out[t, 2] = float(t - prev) if prev is not None else -1.0  # gap since last
        out[t, 3] = float(t)  # absolute time index
        out[t, 4] = float(t) / max(T - 1, 1)  # relative trial time
        # returns in past window of 20 among any segment? count revisits in past 20
        start = max(0, t - 20)
        window = node_index[start:t]
        out[t, 5] = float(np.sum(window == nid)) if t > 0 else 0.0
        visit_count[nid] = vc + 1
        last_seen[nid] = t
    return out


@torch.no_grad()
def collect(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    node_dim: int,
    horizon: int,
) -> dict[str, np.ndarray]:
    model.eval()
    Xs_emb, Xs_feat, Xs_hist, ys, scores = [], [], [], [], []
    for batch in loader:
        batch = _batch_to_device(batch, device)
        out = model(batch)
        y = out["y"]
        ret_logits = out["return_logits"]
        lengths = batch["lengths"].detach().cpu().numpy().astype(int)
        tokens = batch["tokens"]
        for i in range(int(tokens.size(0))):
            L = int(lengths[i])
            if L <= 1:
                continue
            emb = y[i, :L].detach().cpu().numpy().astype(np.float32)
            xv = tokens[i, :L, :node_dim].detach().cpu().numpy()
            side = tokens[i, :L, 2 * node_dim :].detach().cpu().numpy()
            feat = np.concatenate([xv, side], axis=-1).astype(np.float32)
            ni = batch["node_index"][i, :L].detach().cpu().numpy().astype(int)
            lab = return_within_horizon_labels(ni, horizon=horizon)
            ok = lab >= 0
            if not ok.any():
                continue
            hist = _history_aggregates(ni)
            Xs_emb.append(emb[ok])
            Xs_feat.append(feat[ok])
            Xs_hist.append(hist[ok])
            ys.append(lab[ok].astype(int))
            scores.append(ret_logits[i, :L].detach().cpu().numpy()[ok])
    return {
        "emb": np.concatenate(Xs_emb, axis=0),
        "feat": np.concatenate(Xs_feat, axis=0),
        "hist": np.concatenate(Xs_hist, axis=0),
        "y": np.concatenate(ys, axis=0),
        "head_logit": np.concatenate(scores, axis=0),
    }


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "runs"
        / "m6_fullseq_graphbias_return_aux"
        / "fold0_seed13"
        / "checkpoint_best.pt",
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "d1_ceiling_diagnostics.json",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    train_cfg = OmegaConf.load(ROOT / "configs" / "train.yaml")
    # Prefer the run's frozen train_config if present
    run_cfg_path = args.checkpoint.parent / "train_config.yaml"
    if run_cfg_path.is_file():
        run_train_cfg = OmegaConf.load(run_cfg_path)
    else:
        run_train_cfg = train_cfg
    tcfg = OmegaConf.load(ROOT / "configs" / "model_transformer.yaml")
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    scfg = OmegaConf.load(ROOT / "configs" / "splits.yaml")
    dcfg.max_seq_len = int(tcfg.max_seq_len)
    dcfg.build_pair_relations = bool(tcfg.biases.graph_relation.enabled)
    horizon = int(train_cfg.diagnostics.D1_return_probe.horizon_events)

    keys = discover_real_episodes(ROOT / str(dcfg.paths.fixations_root))
    folds = grouped_participant_folds(keys, n_folds=int(scfg.n_folds), seed=int(scfg.seed))
    fold = folds[args.fold]
    ds = LazyRealEpisodeDataset(
        keys,
        dataset_cfg=dcfg,
        fixations_root=ROOT / str(dcfg.paths.fixations_root),
        graphs_root=ROOT / str(dcfg.paths.graphs_root),
        graph_version=str(dcfg.graph_version),
        gnn=None,
        seed=args.seed,
    )
    train_loader = DataLoader(
        Subset(ds, fold["train_idx"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
    )
    val_loader = DataLoader(
        Subset(ds, fold["val_idx"]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
    )

    model = build_behaviour_model(ROOT, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    node_dim = int(dcfg.gnn_out_dim)

    print("Collecting train...", flush=True)
    tr = collect(model, train_loader, device=device, node_dim=node_dim, horizon=horizon)
    print("Collecting val...", flush=True)
    va = collect(model, val_loader, device=device, node_dim=node_dim, horizon=horizon)

    pos_rate_val = float(va["y"].mean())
    pos_rate_train = float(tr["y"].mean())

    # 1) ReturnHead AUC
    head_prob = 1.0 / (1.0 + np.exp(-va["head_logit"]))
    auc_head = _safe_auc(va["y"], head_prob)

    # 2) Alignment: D1-style linear probe on emb (same as run_d1)
    probe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500, class_weight="balanced", solver="lbfgs"
                ),
            ),
        ]
    )
    probe.fit(tr["emb"], tr["y"])
    auc_emb_probe = _safe_auc(va["y"], probe.predict_proba(va["emb"])[:, 1])

    feat_probe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500, class_weight="balanced", solver="lbfgs"
                ),
            ),
        ]
    )
    feat_probe.fit(tr["feat"], tr["y"])
    auc_feat_probe = _safe_auc(va["y"], feat_probe.predict_proba(va["feat"])[:, 1])

    # Head vs probe score correlation on val
    probe_score = probe.predict_proba(va["emb"])[:, 1]
    corr = float(np.corrcoef(probe_score, head_prob)[0, 1])

    # 3) GBM ceiling on feat + history aggregates
    Xtr = np.concatenate([tr["feat"], tr["hist"]], axis=1)
    Xva = np.concatenate([va["feat"], va["hist"]], axis=1)
    gbm = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=args.seed,
    )
    gbm.fit(Xtr, tr["y"])
    auc_gbm = _safe_auc(va["y"], gbm.predict_proba(Xva)[:, 1])

    # Also GBM on hist-only and feat-only for context
    gbm_feat = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=200, random_state=args.seed
    )
    gbm_feat.fit(tr["feat"], tr["y"])
    auc_gbm_feat = _safe_auc(va["y"], gbm_feat.predict_proba(va["feat"])[:, 1])
    gbm_hist = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=200, random_state=args.seed
    )
    gbm_hist.fit(tr["hist"], tr["y"])
    auc_gbm_hist = _safe_auc(va["y"], gbm_hist.predict_proba(va["hist"])[:, 1])

    # 4) Balance fix confirmation from frozen run config
    ra = run_train_cfg.losses.return_aux
    balance = {
        "return_aux_enabled": bool(ra.enabled),
        "weight": float(ra.weight),
        "pos_weight": float(getattr(ra, "pos_weight", 1.0) or 1.0),
        "horizon_events": horizon,
        "val_pos_rate": pos_rate_val,
        "train_pos_rate": pos_rate_train,
        "expected_balance_pos_weight": round((1.0 - pos_rate_val) / pos_rate_val, 4)
        if pos_rate_val and pos_rate_val < 1
        else None,
        "fix_balance_ok": abs(float(getattr(ra, "pos_weight", 1.0) or 1.0) - 0.23) < 0.02,
    }

    alignment = {
        "return_head_input": "per-token transformer output y = encode(batch) [B,T,d_model]",
        "d1_probe_input": "same per-token y (collected as emb); no pooling",
        "return_head": "Linear(d_model -> 1) on y",
        "d1_probe": "StandardScaler + LogisticRegression(class_weight=balanced) on y",
        "same_layer": True,
        "same_pooling": True,
        "pooling": "none (token-level)",
        "val_score_corr_head_vs_probe": corr,
    }

    margin_probe = auc_emb_probe - auc_feat_probe
    summary = {
        "checkpoint": str(args.checkpoint),
        "fold": args.fold,
        "seed": args.seed,
        "horizon_events": horizon,
        "n_train": int(tr["y"].size),
        "n_val": int(va["y"].size),
        "auc_return_head": auc_head,
        "auc_embedding_probe": auc_emb_probe,
        "auc_feature_probe": auc_feat_probe,
        "margin_probe": margin_probe,
        "auc_gbm_feat_plus_history": auc_gbm,
        "auc_gbm_feat_only": auc_gbm_feat,
        "auc_gbm_history_only": auc_gbm_hist,
        "alignment": alignment,
        "balance": balance,
        "decision_hint": {
            "head_near_probe": abs(auc_head - auc_emb_probe) < 0.03,
            "probe_near_ceiling": abs(auc_emb_probe - auc_gbm) < 0.03
            or auc_emb_probe >= auc_gbm - 0.01,
            "margin_vs_050": margin_probe,
            "ceiling_minus_feat": auc_gbm - auc_feat_probe,
            "050_exceeds_ceiling_margin": (auc_gbm - auc_feat_probe) < 0.05,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    uio.write_json(args.out, summary)

    md = args.out.with_suffix(".md")
    lines = [
        "# D1 ceiling diagnostics",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Fold {args.fold} · seed {args.seed} · H={horizon}",
        f"- n_train={summary['n_train']} · n_val={summary['n_val']} · val pos rate={pos_rate_val:.4f}",
        "",
        "## AUCs",
        "",
        f"| Source | AUC |",
        f"|---|---:|",
        f"| ReturnHead (trained) | **{auc_head:.4f}** |",
        f"| D1 embedding probe | **{auc_emb_probe:.4f}** |",
        f"| D1 feature-only probe | {auc_feat_probe:.4f} |",
        f"| GBM feat + history (ceiling) | **{auc_gbm:.4f}** |",
        f"| GBM feat only | {auc_gbm_feat:.4f} |",
        f"| GBM history only | {auc_gbm_hist:.4f} |",
        "",
        f"- Probe margin (emb − feat): **{margin_probe:.4f}** (need ≥ 0.05)",
        f"- Ceiling − feat: **{auc_gbm - auc_feat_probe:.4f}**",
        f"- Head vs probe score corr: {corr:.4f}",
        "",
        "## Alignment",
        "",
        f"- Same representation: **yes** — both consume per-token `y = encode(batch)` "
        f"(no pooling).",
        f"- ReturnHead: `Linear(d_model→1)` on `y`.",
        f"- D1 probe: `StandardScaler + LogisticRegression(balanced)` on the same `y`.",
        "",
        "## Balance fix shipped",
        "",
        f"- `return_aux.enabled={balance['return_aux_enabled']}`, weight={balance['weight']}",
        f"- **pos_weight={balance['pos_weight']}** (not shorter H); horizon **{horizon}**",
        f"- Val positive rate: **{pos_rate_val:.4f}** (train {pos_rate_train:.4f})",
        f"- Balance OK (pos_weight≈0.23): **{balance['fix_balance_ok']}**",
        "",
    ]
    uio.write_text(md, "\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "alignment"}, indent=2))
    print(f"Wrote {args.out}", flush=True)
    print(f"Wrote {md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
