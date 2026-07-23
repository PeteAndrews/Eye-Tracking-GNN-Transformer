"""Collect frozen per-token embeddings + interpretable features (val fold only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.targets import RELATION_VOCAB
from src.eval.m6_predictive import build_fold_loaders
from src.train.loop import build_behaviour_model
from src.utils import io as uio
from src.utils.arrow_cuda import read_parquet


def _align_parquet_features(
    ep: dict[str, Any],
    *,
    fixations_root: Path,
) -> dict[str, np.ndarray | list]:
    """Pull interpretable columns from fixation parquet, truncated to emb length."""
    T = int(ep["emb"].shape[0])
    pid = str(ep["participant_id"])
    tid = str(ep["trial_id"])
    sc = str(ep.get("star_condition") or "not_eligible")
    pq = fixations_root / pid / f"{tid}__{sc}.parquet"
    empty = {
        "duration_ms": np.zeros(T, dtype=np.float32),
        "assignment_confidence": np.zeros(T, dtype=np.float32),
        "visit_count": np.ones(T, dtype=np.float32),
        "is_return": np.zeros(T, dtype=np.float32),
        "prev_saccade_amplitude": np.zeros(T, dtype=np.float32),
        "rel_t": np.linspace(0.0, 1.0, T, dtype=np.float32),
        "loop_role": ["none"] * T,
        "loop_template_primary": [""] * T,
        "panel_label": ["unknown"] * T,
        "x_doc": np.full(T, np.nan, dtype=np.float32),
        "y_doc": np.full(T, np.nan, dtype=np.float32),
        "fixation_id": [str(i) for i in range(T)],
    }
    if not pq.is_file():
        return empty
    df = read_parquet(pq)
    if len(df) > T:
        df = df.iloc[:T].copy()
    n = len(df)
    if n == 0:
        return empty

    def _col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in df.columns:
            return np.full(n, default, dtype=np.float32)
        return df[name].to_numpy(dtype=np.float32, copy=True)

    dur = _col("duration_ms")
    t0 = _col("t_start_ms")
    ep_dur = float(t0[-1] + dur[-1]) if n else 1.0
    rel_t = (t0 / max(ep_dur, 1.0)).astype(np.float32)
    roles = (
        df["loop_role"].astype(str).fillna("none").tolist()
        if "loop_role" in df.columns
        else ["none"] * n
    )
    tmpls: list[str] = []
    if "loop_template_id" in df.columns:
        for v in df["loop_template_id"].tolist():
            s = str(v) if v is not None and str(v) not in ("nan", "None", "") else ""
            tmpls.append(s.split("|")[0] if s else "")
    else:
        tmpls = [""] * n
    panels = (
        df["panel_label"].astype(str).fillna("unknown").tolist()
        if "panel_label" in df.columns
        else ["unknown"] * n
    )
    conf = _col("assignment_confidence")
    visit = _col("visit_count", 1.0)
    is_ret = _col("is_return")
    sacc = _col("prev_saccade_amplitude")
    x = _col("x_doc", np.nan)
    y = _col("y_doc", np.nan)
    fix_ids = (
        df["fixation_id"].astype(str).tolist()
        if "fixation_id" in df.columns
        else [str(i) for i in range(n)]
    )
    if n < T:
        pad = T - n
        dur = np.concatenate([dur, np.zeros(pad, dtype=np.float32)])
        rel_t = np.concatenate([rel_t, np.ones(pad, dtype=np.float32)])
        roles = roles + ["none"] * pad
        tmpls = tmpls + [""] * pad
        panels = panels + ["unknown"] * pad
        conf = np.concatenate([conf, np.zeros(pad, dtype=np.float32)])
        visit = np.concatenate([visit, np.ones(pad, dtype=np.float32)])
        is_ret = np.concatenate([is_ret, np.zeros(pad, dtype=np.float32)])
        sacc = np.concatenate([sacc, np.zeros(pad, dtype=np.float32)])
        x = np.concatenate([x, np.full(pad, np.nan, dtype=np.float32)])
        y = np.concatenate([y, np.full(pad, np.nan, dtype=np.float32)])
        fix_ids = fix_ids + [str(i) for i in range(n, T)]
    return {
        "duration_ms": dur.astype(np.float32),
        "assignment_confidence": conf.astype(np.float32),
        "visit_count": visit.astype(np.float32),
        "is_return": is_ret.astype(np.float32),
        "prev_saccade_amplitude": sacc.astype(np.float32),
        "rel_t": rel_t.astype(np.float32),
        "loop_role": roles,
        "loop_template_primary": tmpls,
        "panel_label": panels,
        "x_doc": x.astype(np.float32),
        "y_doc": y.astype(np.float32),
        "fixation_id": fix_ids,
    }


@torch.no_grad()
def collect_val_with_relations(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    node_dim: int,
    active_labels: list[str],
) -> list[dict[str, Any]]:
    """Encode val episodes once; keep active next-relation multi-hot per token."""
    model.eval()
    active_idx = [RELATION_VOCAB.index(n) for n in active_labels]
    episodes: list[dict[str, Any]] = []
    for batch in loader:
        batch_d = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
        y = model.encode(batch_d)
        tokens = batch_d["tokens"]
        lengths = batch_d["lengths"].detach().cpu().numpy().astype(int)
        rel = batch_d["next_relation"].detach().cpu().numpy()
        bsz = int(tokens.size(0))
        for i in range(bsz):
            L = int(lengths[i])
            if L <= 1:
                continue
            side = tokens[i, :L, 2 * node_dim :].detach().cpu().numpy()
            xv = tokens[i, :L, :node_dim].detach().cpu().numpy()
            emb = y[i, :L].detach().cpu().numpy()
            node_index = batch_d["node_index"][i, :L].detach().cpu().numpy().astype(int)
            panel = batch_d["panel_id"][i, :L].detach().cpu().numpy().astype(int)
            episodes.append(
                {
                    "emb": emb.astype(np.float32),
                    "feat": np.concatenate([xv, side], axis=-1).astype(np.float32),
                    "node_index": node_index,
                    "panel_id": panel,
                    "next_relation_active": rel[i, :L, :][:, active_idx].astype(
                        np.float32
                    ),
                    "participant_id": (
                        batch_d["participant_id"][i]
                        if isinstance(batch_d.get("participant_id"), list)
                        else None
                    ),
                    "trial_id": (
                        batch_d["trial_id"][i]
                        if isinstance(batch_d.get("trial_id"), list)
                        else None
                    ),
                    "star_condition": (
                        batch_d["star_condition"][i]
                        if isinstance(batch_d.get("star_condition"), list)
                        else None
                    ),
                }
            )
    return episodes


def collect_and_save(
    repo: Path,
    checkpoint: Path,
    out_dir: Path,
    *,
    fold: int = 0,
    seed: int = 13,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Encode grouped-val tokens, join parquet features, dump reproducibility artefacts."""
    repo = Path(repo)
    out_dir = Path(out_dir)
    tok_dir = out_dir / "tokens"
    tok_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg = OmegaConf.load(repo / "configs" / "train.yaml")
    dcfg = OmegaConf.load(repo / "configs" / "dataset.yaml")
    active_labels = list(train_cfg.relation_weights.active_labels)

    _, val_loader, meta = build_fold_loaders(repo, fold=fold, seed=seed, batch_size=4)
    model = build_behaviour_model(repo, device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(
        f"Collecting val embeddings (fold={fold}, participants={meta['val_participants']})...",
        flush=True,
    )
    episodes = collect_val_with_relations(
        model,
        val_loader,
        device=device,
        node_dim=int(meta["gnn_out_dim"]),
        active_labels=active_labels,
    )
    fix_root = repo / str(dcfg.paths.fixations_root)
    for ep in episodes:
        feats = _align_parquet_features(ep, fixations_root=fix_root)
        ep.update(feats)

    # Flatten for clustering / fingerprints
    rows: list[dict[str, Any]] = []
    emb_list: list[np.ndarray] = []
    for ep_i, ep in enumerate(episodes):
        T = int(ep["emb"].shape[0])
        emb_list.append(ep["emb"])
        for t in range(T):
            rows.append(
                {
                    "episode_idx": ep_i,
                    "t": t,
                    "participant_id": ep["participant_id"],
                    "trial_id": ep["trial_id"],
                    "star_condition": ep.get("star_condition"),
                    "panel_id": int(ep["panel_id"][t]),
                    "panel_label": ep["panel_label"][t],
                    "node_index": int(ep["node_index"][t]),
                    "duration_ms": float(ep["duration_ms"][t]),
                    "assignment_confidence": float(ep["assignment_confidence"][t]),
                    "visit_count": float(ep["visit_count"][t]),
                    "is_return": float(ep["is_return"][t]),
                    "prev_saccade_amplitude": float(ep["prev_saccade_amplitude"][t]),
                    "rel_t": float(ep["rel_t"][t]),
                    "loop_role": ep["loop_role"][t],
                    "loop_template_primary": ep["loop_template_primary"][t],
                    "x_doc": float(ep["x_doc"][t]),
                    "y_doc": float(ep["y_doc"][t]),
                    "fixation_id": str(ep["fixation_id"][t]),
                    **{
                        f"rel_{lab}": float(ep["next_relation_active"][t, j])
                        for j, lab in enumerate(active_labels)
                    },
                }
            )

    emb = np.concatenate(emb_list, axis=0).astype(np.float32)
    import pandas as pd

    df = pd.DataFrame(rows)
    assert len(df) == emb.shape[0]

    np.savez_compressed(tok_dir / "embeddings.npz", emb=emb)
    df.to_parquet(tok_dir / "token_table.parquet", index=False)
    # Also keep per-episode list for exemplar contiguous search
    uio.write_json(
        tok_dir / "episodes_meta.json",
        [
            {
                "episode_idx": i,
                "participant_id": ep["participant_id"],
                "trial_id": ep["trial_id"],
                "star_condition": ep.get("star_condition"),
                "n_tokens": int(ep["emb"].shape[0]),
            }
            for i, ep in enumerate(episodes)
        ],
    )
    meta_out = {
        "diagnostic": True,
        "label": "m8_diagnostic_peek",
        "note": (
            "GMM is fit on grouped-val embeddings only (deliberate peek). "
            "Real M8 fits per-fold on that fold's train and describes its val; "
            "chosen k / labels from this peek feed nothing downstream."
        ),
        "checkpoint": str(checkpoint),
        "fold": fold,
        "seed": seed,
        "val_participants": list(meta["val_participants"]),
        "n_val_episodes": len(episodes),
        "n_tokens": int(emb.shape[0]),
        "emb_dim": int(emb.shape[1]),
        "active_relation_labels": active_labels,
        "panel_classes": list(meta["panel_classes"]),
        "device": str(device),
        "tokens_dir": str(tok_dir),
    }
    uio.write_json(tok_dir / "collect_meta.json", meta_out)
    print(
        f"Saved {emb.shape[0]} tokens × dim {emb.shape[1]} under {tok_dir}",
        flush=True,
    )
    return meta_out
