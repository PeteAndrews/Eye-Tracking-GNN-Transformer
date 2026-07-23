"""Step 1 — PCA + GMM BIC + dual-regime stability + UMAP (diagnostic only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils import io as uio


def _mean_pairwise_ami(labels: list[np.ndarray]) -> float:
    vals = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            vals.append(float(adjusted_mutual_info_score(labels[i], labels[j])))
    return float(np.mean(vals)) if vals else float("nan")


def _within_participant_shuffle(
    labels: np.ndarray,
    participant_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    out = labels.copy()
    for pid in np.unique(participant_ids):
        mask = participant_ids == pid
        idx = np.where(mask)[0]
        out[idx] = rng.permutation(out[idx])
    return out


def _null_ami(
    fitted_labels: list[np.ndarray],
    participant_ids: np.ndarray,
    *,
    n_reps: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_reps):
        shuffled = [
            _within_participant_shuffle(lab, participant_ids, rng) for lab in fitted_labels
        ]
        scores.append(_mean_pairwise_ami(shuffled))
    return {
        "null_ami_mean": float(np.mean(scores)),
        "null_ami_std": float(np.std(scores)),
        "null_ami_reps": n_reps,
    }


def pca_retain_variance(
    X: np.ndarray, *, target_var: float = 0.90, seed: int = 0
) -> tuple[np.ndarray, PCA, StandardScaler, int]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca_full = PCA(random_state=seed)
    pca_full.fit(Xs)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    n = int(np.searchsorted(cum, target_var) + 1)
    n = max(2, min(n, Xs.shape[1], Xs.shape[0] - 1))
    pca = PCA(n_components=n, random_state=seed)
    Z = pca.fit_transform(Xs)
    return Z, pca, scaler, n


def select_gmm_bic(
    Z: np.ndarray,
    *,
    k_min: int = 4,
    k_max: int = 12,
    seed: int = 0,
) -> dict[str, Any]:
    bic_curve = []
    best = None
    best_k = k_min
    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=seed,
            n_init=3,
            max_iter=300,
        )
        gmm.fit(Z)
        bic = float(gmm.bic(Z))
        bic_curve.append({"k": k, "bic": bic})
        if best is None or bic < best:
            best = bic
            best_k = k
            best_gmm = gmm
    return {"k": best_k, "gmm": best_gmm, "bic_curve": bic_curve}


def _episode_indices(df: pd.DataFrame) -> np.ndarray:
    return df["episode_idx"].to_numpy()


def _fit_predict_full(
    Z: np.ndarray,
    fit_idx: np.ndarray,
    *,
    k: int,
    seed: int,
) -> np.ndarray:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        random_state=seed,
        n_init=3,
        max_iter=300,
    )
    gmm.fit(Z[fit_idx])
    return gmm.predict(Z)


def stability_random_tokens(
    Z: np.ndarray,
    df: pd.DataFrame,
    *,
    k: int,
    n_fits: int = 3,
    frac: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    """Regime A: random 80% of tokens (may drop participants)."""
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    labels = []
    compositions = []
    for i in range(n_fits):
        m = max(1, int(round(frac * n)))
        fit_idx = rng.choice(n, size=m, replace=False)
        lab = _fit_predict_full(Z, fit_idx, k=k, seed=seed + 100 + i)
        labels.append(lab)
        pids = sorted(set(df.iloc[fit_idx]["participant_id"].astype(str)))
        compositions.append(
            {
                "fit": i,
                "n_tokens_fit": int(m),
                "participants_in_fit": pids,
                "n_participants_in_fit": len(pids),
            }
        )
    ami = _mean_pairwise_ami(labels)
    null = _null_ami(labels, df["participant_id"].astype(str).to_numpy(), seed=seed + 7)
    return {
        "regime": "random_80pct_tokens",
        "mean_pairwise_ami": ami,
        "compositions": compositions,
        **null,
        "labels": labels,
    }


def stability_stratified_episodes(
    Z: np.ndarray,
    df: pd.DataFrame,
    *,
    k: int,
    n_fits: int = 3,
    frac: float = 0.8,
    seed: int = 0,
) -> dict[str, Any]:
    """Regime B: all participants kept; 80% of each participant's episodes."""
    rng = np.random.default_rng(seed + 99)
    ep_meta = (
        df.groupby("episode_idx", sort=True)
        .agg(participant_id=("participant_id", "first"))
        .reset_index()
    )
    labels = []
    compositions = []
    for i in range(n_fits):
        keep_eps: list[int] = []
        for pid, g in ep_meta.groupby("participant_id"):
            eids = g["episode_idx"].to_numpy()
            m = max(1, int(round(frac * len(eids))))
            chosen = rng.choice(eids, size=min(m, len(eids)), replace=False)
            keep_eps.extend(int(x) for x in chosen)
        keep_eps_arr = np.array(sorted(set(keep_eps)), dtype=int)
        fit_mask = df["episode_idx"].isin(keep_eps_arr).to_numpy()
        fit_idx = np.where(fit_mask)[0]
        lab = _fit_predict_full(Z, fit_idx, k=k, seed=seed + 200 + i)
        labels.append(lab)
        pids = sorted(set(df.iloc[fit_idx]["participant_id"].astype(str)))
        compositions.append(
            {
                "fit": i,
                "n_tokens_fit": int(fit_idx.size),
                "n_episodes_fit": int(len(keep_eps_arr)),
                "participants_in_fit": pids,
                "n_participants_in_fit": len(pids),
            }
        )
    ami = _mean_pairwise_ami(labels)
    null = _null_ami(labels, df["participant_id"].astype(str).to_numpy(), seed=seed + 17)
    return {
        "regime": "stratified_80pct_episodes_all_participants",
        "mean_pairwise_ami": ami,
        "compositions": compositions,
        **null,
        "labels": labels,
    }


def gate_verdict(ami_a: float, ami_b: float) -> dict[str, Any]:
    """Apply extended stability gate."""
    hi = 0.7
    lo = 0.5
    gap = abs(ami_a - ami_b)
    both_hi = ami_a > hi and ami_b > hi
    both_lo = ami_a < lo and ami_b < lo
    if both_hi:
        verdict = "stable"
        proceed = True
        note = "AMI > 0.7 across both regimes."
    elif both_lo:
        verdict = "no_structure"
        proceed = False
        note = "AMI < 0.5 in both regimes — stop after Step 1."
    else:
        # Mid-band on either regime, or a large cross-regime gap.
        verdict = "sample_fragile"
        proceed = True
        note = (
            "Structure present but sample-fragile "
            f"(AMI_random={ami_a:.3f}, AMI_stratified={ami_b:.3f}, gap={gap:.3f})."
        )
    return {
        "verdict": verdict,
        "proceed_to_step2": proceed,
        "ami_random": ami_a,
        "ami_stratified": ami_b,
        "ami_gap": gap,
        "note": note,
    }


def run_structure(
    tokens_dir: Path,
    out_dir: Path,
    *,
    target_var: float = 0.90,
    k_min: int = 4,
    k_max: int = 12,
    seed: int = 0,
) -> dict[str, Any]:
    tokens_dir = Path(tokens_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = np.load(tokens_dir / "embeddings.npz")["emb"]
    df = pd.read_parquet(tokens_dir / "token_table.parquet")
    collect_meta = uio.read_json(tokens_dir / "collect_meta.json")

    print(f"PCA to ~{target_var:.0%} variance...", flush=True)
    Z, pca, scaler, n_comp = pca_retain_variance(emb, target_var=target_var, seed=seed)
    var_explained = float(np.sum(pca.explained_variance_ratio_))

    print(f"GMM BIC over k in [{k_min},{k_max}]...", flush=True)
    sel = select_gmm_bic(Z, k_min=k_min, k_max=k_max, seed=seed)
    k = int(sel["k"])
    gmm = sel["gmm"]
    hard = gmm.predict(Z)
    post = gmm.predict_proba(Z)
    conf = post.max(axis=1)

    print("Stability regime A (random 80% tokens)...", flush=True)
    stab_a = stability_random_tokens(Z, df, k=k, seed=seed)
    print("Stability regime B (stratified 80% episodes)...", flush=True)
    stab_b = stability_stratified_episodes(Z, df, k=k, seed=seed)
    gate = gate_verdict(stab_a["mean_pairwise_ami"], stab_b["mean_pairwise_ami"])

    # Persist assignments for later steps
    assign = df.copy()
    assign["prototype"] = hard.astype(int)
    assign["posterior_max"] = conf.astype(np.float32)
    for j in range(k):
        assign[f"post_{j}"] = post[:, j].astype(np.float32)
    assign.to_parquet(out_dir / "assignments.parquet", index=False)
    np.savez_compressed(out_dir / "pca_Z.npz", Z=Z.astype(np.float32))

    # BIC plot
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ks = [r["k"] for r in sel["bic_curve"]]
    bics = [r["bic"] for r in sel["bic_curve"]]
    ax.plot(ks, bics, marker="o")
    ax.axvline(k, color="C3", ls="--", label=f"chosen k={k}")
    ax.set_xlabel("k")
    ax.set_ylabel("BIC")
    ax.set_title("GMM BIC curve (diagnostic peek)")
    ax.legend()
    fig.tight_layout()
    bic_path = out_dir / "bic_curve.png"
    fig.savefig(bic_path, dpi=140)
    plt.close(fig)

    # UMAP (optional) / PCA2 fallback
    umap_note = ""
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
        xy = reducer.fit_transform(Z)
        umap_note = "UMAP-2D on PCA features."
    except Exception as exc:  # noqa: BLE001
        xy = Z[:, :2]
        umap_note = f"UMAP unavailable ({exc}); PCA dims 1–2 used instead."

    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=hard, cmap="tab10", s=3, alpha=0.5, linewidths=0)
    ax.set_title(f"2D embedding coloured by hard GMM label (k={k})")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    fig.colorbar(sc, ax=ax, label="prototype")
    fig.tight_layout()
    umap_path = out_dir / "umap_scatter.png"
    fig.savefig(umap_path, dpi=140)
    plt.close(fig)

    # Drop label arrays from JSON (huge)
    stab_a_j = {kk: vv for kk, vv in stab_a.items() if kk != "labels"}
    stab_b_j = {kk: vv for kk, vv in stab_b.items() if kk != "labels"}

    summary = {
        "diagnostic": True,
        "label": "m8_diagnostic_peek_step1",
        "fit_protocol": (
            "GMM fit on grouped-val embeddings only. Deliberate for this peek; "
            "real M8 fits on train per fold. Chosen k/labels feed nothing downstream."
        ),
        "val_participants": collect_meta.get("val_participants"),
        "n_tokens": int(emb.shape[0]),
        "emb_dim": int(emb.shape[1]),
        "pca_n_components": n_comp,
        "pca_variance_retained": var_explained,
        "pca_target_variance": target_var,
        "bic_curve": sel["bic_curve"],
        "chosen_k": k,
        "stability_random": stab_a_j,
        "stability_stratified": stab_b_j,
        "gate": gate,
        "umap_note": umap_note,
        "artefacts": {
            "bic_curve_png": str(bic_path),
            "umap_scatter_png": str(umap_path),
            "assignments_parquet": str(out_dir / "assignments.parquet"),
            "tokens_dir": str(tokens_dir),
        },
    }
    uio.write_json(out_dir / "structure.json", summary)

    # Markdown report
    lines = [
        "# M8 diagnostic peek — Step 1 structure",
        "",
        "> **Diagnostic only** — not an M8 finding. GMM fit on **grouped-val**",
        "> participants only (held-out). Real M8 will fit on train per fold;",
        "> this peek's chosen k / labels feed **nothing** downstream.",
        "",
        f"- Checkpoint / tokens: `{collect_meta.get('checkpoint')}`",
        f"- Val participants: {', '.join(collect_meta.get('val_participants') or [])}",
        f"- Tokens: **{emb.shape[0]}** × emb_dim **{emb.shape[1]}**",
        f"- PCA: **{n_comp}** components retaining **{var_explained:.3f}** variance "
        f"(target {target_var:.0%})",
        f"- BIC-chosen **k = {k}**",
        "",
        "## BIC curve",
        "",
        f"![BIC]({bic_path.name})",
        "",
        "| k | BIC |",
        "|---|-----|",
    ]
    for row in sel["bic_curve"]:
        mark = " ←" if row["k"] == k else ""
        lines.append(f"| {row['k']} | {row['bic']:.1f}{mark} |")
    lines += [
        "",
        "## Stability",
        "",
        f"| Regime | Mean pairwise AMI | Within-participant null AMI (mean±std) |",
        f"|--------|-------------------|----------------------------------------|",
        f"| Random 80% tokens | **{stab_a['mean_pairwise_ami']:.3f}** | "
        f"{stab_a['null_ami_mean']:.3f} ± {stab_a['null_ami_std']:.3f} |",
        f"| Stratified 80% episodes (all 5 Pids) | **{stab_b['mean_pairwise_ami']:.3f}** | "
        f"{stab_b['null_ami_mean']:.3f} ± {stab_b['null_ami_std']:.3f} |",
        "",
        "### Random-regime subsample composition",
        "",
    ]
    for c in stab_a["compositions"]:
        lines.append(
            f"- Fit {c['fit']}: n_tokens={c['n_tokens_fit']}, "
            f"participants={c['participants_in_fit']} "
            f"(n={c['n_participants_in_fit']})"
        )
    lines += ["", "### Stratified-regime subsample composition", ""]
    for c in stab_b["compositions"]:
        lines.append(
            f"- Fit {c['fit']}: n_tokens={c['n_tokens_fit']}, "
            f"n_episodes={c['n_episodes_fit']}, "
            f"participants={c['participants_in_fit']} "
            f"(n={c['n_participants_in_fit']})"
        )
    lines += [
        "",
        "## Gate verdict",
        "",
        f"**{gate['verdict']}** — {gate['note']}",
        "",
        f"- Proceed to Step 2: **{gate['proceed_to_step2']}**",
        "",
        "## 2D scatter (eyeball only — do not over-read)",
        "",
        umap_note,
        "",
        f"![scatter]({umap_path.name})",
        "",
        "## Reproducibility",
        "",
        f"- Per-token embeddings: `{tokens_dir / 'embeddings.npz'}`",
        f"- Token table: `{tokens_dir / 'token_table.parquet'}`",
        f"- Assignments: `{out_dir / 'assignments.parquet'}`",
        f"- Collect meta (participants): `{tokens_dir / 'collect_meta.json'}`",
        "",
    ]
    (out_dir / "structure.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Step 1 verdict: {gate['verdict']} (proceed={gate['proceed_to_step2']})", flush=True)
    return summary
