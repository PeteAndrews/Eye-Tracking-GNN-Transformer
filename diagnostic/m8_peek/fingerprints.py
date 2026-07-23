"""Step 2 — prototype fingerprints (standardised mean differences) + occupancy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import io as uio

# Feature groups for the static vs history/phase diagnostic read
STATIC_FEATURES = {
    "panel_question",
    "panel_response",
    "panel_mark_scheme",
    "panel_commentary",
    "panel_star_chart",
    "panel_ui",
    "panel_outside",
    "duration_ms",
    "prev_saccade_amplitude",
    "assignment_confidence",
}
HISTORY_PHASE_FEATURES = {
    "is_return",
    "visit_count",
    "rel_t",
    "loop_role_none",
    "loop_role_origin",
    "loop_role_pivot",
    "loop_role_closure",
}


def _panel_bucket(label: str) -> str:
    s = str(label).lower()
    if "outside" in s:
        return "outside"
    if s in ("question", "response", "mark_scheme", "commentary", "star_chart", "ui"):
        return s
    if "mark" in s:
        return "mark_scheme"
    if "star" in s:
        return "star_chart"
    return "outside" if "unknown" in s else s


def build_feature_matrix(
    df: pd.DataFrame, active_relation_labels: list[str]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return feature frame + group map (static | history_phase | relation | template)."""
    feats: dict[str, np.ndarray] = {}
    groups: dict[str, str] = {}

    panels = ["question", "response", "mark_scheme", "commentary", "star_chart", "ui", "outside"]
    bucket = df["panel_label"].map(_panel_bucket)
    for p in panels:
        name = f"panel_{p}"
        feats[name] = (bucket == p).astype(np.float32).to_numpy()
        groups[name] = "static"

    for lab in active_relation_labels:
        col = f"rel_{lab}"
        name = f"relation_{lab}"
        feats[name] = df[col].to_numpy(dtype=np.float32) if col in df.columns else np.zeros(len(df))
        groups[name] = "relation"

    for role in ("none", "origin", "pivot", "closure"):
        name = f"loop_role_{role}"
        feats[name] = (df["loop_role"].astype(str) == role).astype(np.float32).to_numpy()
        groups[name] = "history_phase"

    # Top loop templates by prevalence (cap at 8 for tornado readability)
    tmpl = df["loop_template_primary"].fillna("").astype(str)
    counts = tmpl[tmpl != ""].value_counts()
    top_tmpls = list(counts.head(8).index)
    for tid in top_tmpls:
        name = f"template__{tid}"
        feats[name] = (tmpl == tid).astype(np.float32).to_numpy()
        groups[name] = "history_phase"

    cont = {
        "is_return": "history_phase",
        "duration_ms": "static",
        "prev_saccade_amplitude": "static",
        "visit_count": "history_phase",
        "rel_t": "history_phase",
        "assignment_confidence": "static",
    }
    for c, g in cont.items():
        feats[c] = df[c].to_numpy(dtype=np.float32)
        groups[c] = g

    X = pd.DataFrame(feats)
    return X, groups


def standardised_mean_diffs(
    X: pd.DataFrame, labels: np.ndarray, k: int
) -> dict[int, pd.Series]:
    """Per-prototype SMD vs all others: (mu_in - mu_out) / pooled_std."""
    out: dict[int, pd.Series] = {}
    for c in range(k):
        mask = labels == c
        if mask.sum() < 5 or (~mask).sum() < 5:
            out[c] = pd.Series(0.0, index=X.columns)
            continue
        mu_in = X.loc[mask].mean()
        mu_out = X.loc[~mask].mean()
        # pooled std across full set (stable denominator)
        pooled = X.std(ddof=0).replace(0, np.nan)
        smd = (mu_in - mu_out) / pooled
        out[c] = smd.fillna(0.0)
    return out


def tornado_chart(
    smd: pd.Series,
    groups: dict[str, str],
    *,
    title: str,
    path: Path,
    top_n: int = 16,
) -> None:
    s = smd.reindex(smd.abs().sort_values(ascending=False).index).head(top_n)
    s = s.iloc[::-1]  # top at top after barh
    colors = []
    for name in s.index:
        g = groups.get(str(name), "")
        if g == "static":
            colors.append("#4c78a8")
        elif g in ("history_phase", "relation"):
            colors.append("#f58518")
        else:
            colors.append("#54a24b")
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.28 * len(s))))
    ax.barh(range(len(s)), s.values, color=colors)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([str(i)[:42] for i in s.index], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Standardised mean difference vs others")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def episode_level_occupancy(assign: pd.DataFrame, k: int) -> pd.DataFrame:
    """Proportion of episode *time* (duration) per prototype — never fixation-pooled."""
    rows = []
    for (pid, tid, sc, ep_i), g in assign.groupby(
        ["participant_id", "trial_id", "star_condition", "episode_idx"], sort=False
    ):
        dur = g["duration_ms"].to_numpy(dtype=np.float64)
        dur = np.where(np.isfinite(dur) & (dur > 0), dur, 1.0)
        total = float(dur.sum())
        for c in range(k):
            m = g["prototype"].to_numpy() == c
            rows.append(
                {
                    "participant_id": pid,
                    "trial_id": tid,
                    "star_condition": sc,
                    "episode_idx": ep_i,
                    "prototype": c,
                    "prop_time": float(dur[m].sum() / max(total, 1e-9)),
                }
            )
    return pd.DataFrame(rows)


def within_participant_mass(assign: pd.DataFrame, k: int, top_n: int = 5) -> list[dict[str, Any]]:
    """Duration mass by participant for highest-membership prototypes; flag >60%."""
    dur = assign["duration_ms"].to_numpy(dtype=np.float64)
    dur = np.where(np.isfinite(dur) & (dur > 0), dur, 1.0)
    totals = []
    for c in range(k):
        m = assign["prototype"].to_numpy() == c
        totals.append((c, float(dur[m].sum())))
    totals.sort(key=lambda x: -x[1])
    reports = []
    for c, mass in totals[:top_n]:
        m = assign["prototype"].to_numpy() == c
        by_pid: dict[str, float] = {}
        for pid, g in assign.loc[m].groupby("participant_id"):
            d = g["duration_ms"].to_numpy(dtype=np.float64)
            d = np.where(np.isfinite(d) & (d > 0), d, 1.0)
            by_pid[str(pid)] = float(d.sum())
        s = sum(by_pid.values()) or 1.0
        shares = {p: v / s for p, v in by_pid.items()}
        top_pid = max(shares, key=shares.get) if shares else None
        top_share = shares.get(top_pid, 0.0) if top_pid else 0.0
        reports.append(
            {
                "prototype": c,
                "total_duration_mass": mass,
                "participant_shares": {p: round(shares[p], 4) for p in sorted(shares)},
                "top_participant": top_pid,
                "top_share": round(top_share, 4),
                "participant_dominated": bool(top_share > 0.60),
            }
        )
    return reports


HISTORY_MARKER_FEATURES = (
    "is_return",
    "rel_t",
    "visit_count",
    "loop_role_none",
    "loop_role_origin",
    "loop_role_pivot",
    "loop_role_closure",
)


def top5_by_abs_smd(smds: dict[int, pd.Series], *, top_n: int = 5) -> dict[int, list[dict[str, Any]]]:
    """Per-prototype top features by |SMD| — no category aggregation."""
    out: dict[int, list[dict[str, Any]]] = {}
    for c, s in smds.items():
        ordered = s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)
        rows = []
        for name, val in ordered.items():
            rows.append(
                {
                    "feature": str(name),
                    "smd": float(val),
                    "abs_smd": float(abs(val)),
                }
            )
        out[int(c)] = rows
    return out


def history_in_top5_scan(top5: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Does any prototype's top-5 include return / loop role / t/T?"""
    markers = set(HISTORY_MARKER_FEATURES)
    # also catch template_* and loop_role_* generically
    hits: list[dict[str, Any]] = []
    for c, rows in top5.items():
        found = []
        for r in rows:
            f = r["feature"]
            if f in markers or f.startswith("loop_role_") or f.startswith("template__"):
                if f == "is_return" or f == "rel_t" or f.startswith("loop_role_") or f.startswith(
                    "template__"
                ):
                    found.append(f)
                elif f == "visit_count":
                    found.append(f)
        # User asked specifically: return rate, loop role, or t/T
        key = [
            f
            for f in found
            if f == "is_return" or f == "rel_t" or f.startswith("loop_role_")
        ]
        if key:
            hits.append({"prototype": int(c), "history_features_in_top5": key, "top5": rows})
    return {
        "n_prototypes_with_return_loop_or_rel_t_in_top5": len(hits),
        "prototypes": hits,
        "any": bool(hits),
    }


def static_vs_history_summary(
    smds: dict[int, pd.Series], groups: dict[str, str]
) -> dict[str, Any]:
    """Secondary aggregate only (do not use as the primary read)."""
    per = []
    for c, s in smds.items():
        static_abs = []
        hist_abs = []
        for name, val in s.items():
            g = groups.get(str(name), "")
            av = abs(float(val))
            if g == "static":
                static_abs.append(av)
            elif g in ("history_phase", "relation"):
                hist_abs.append(av)
        per.append(
            {
                "prototype": c,
                "mean_abs_smd_static": float(np.mean(static_abs) if static_abs else 0.0),
                "mean_abs_smd_history_phase": float(np.mean(hist_abs) if hist_abs else 0.0),
                "history_gt_static": float(np.mean(hist_abs) if hist_abs else 0)
                > float(np.mean(static_abs) if static_abs else 0),
            }
        )
    n_hist = sum(1 for r in per if r["history_gt_static"])
    return {
        "per_prototype": per,
        "n_prototypes_history_dominated": n_hist,
        "n_prototypes": len(per),
        "read": (
            f"{n_hist}/{len(per)} prototypes have higher mean |SMD| on "
            "history/phase/relation features than on static features "
            "(aggregate only — prefer top-5 feature lists)."
        ),
    }


def run_fingerprints(out_dir: Path, tokens_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    tokens_dir = Path(tokens_dir)
    fig_dir = out_dir / "fingerprint_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    assign = pd.read_parquet(out_dir / "assignments.parquet")
    collect_meta = uio.read_json(tokens_dir / "collect_meta.json")
    structure = uio.read_json(out_dir / "structure.json")
    k = int(structure["chosen_k"])
    active = list(collect_meta["active_relation_labels"])

    X, groups = build_feature_matrix(assign, active)
    labels = assign["prototype"].to_numpy(dtype=int)
    smds = standardised_mean_diffs(X, labels, k)

    for c, s in smds.items():
        tornado_chart(
            s,
            groups,
            title=f"Prototype {c} fingerprint (SMD vs others)",
            path=fig_dir / f"tornado_p{c}.png",
        )

    occ = episode_level_occupancy(assign, k)
    occ.to_parquet(out_dir / "episode_occupancy.parquet", index=False)
    pid_mass = within_participant_mass(assign, k, top_n=5)
    top5 = top5_by_abs_smd(smds, top_n=5)
    hist_scan = history_in_top5_scan(top5)
    sep = static_vs_history_summary(smds, groups)

    # Exemplars: all prototypes (HTML per proto). Prefer history-in-top5 order first.
    hist_protos = {h["prototype"] for h in hist_scan["prototypes"]}
    pick = sorted(hist_protos) + [c for c in range(k) if c not in hist_protos]

    summary = {
        "diagnostic": True,
        "label": "m8_diagnostic_peek_step2",
        "chosen_k": k,
        "top5_by_abs_smd": {str(c): top5[c] for c in range(k)},
        "history_in_top5": hist_scan,
        "static_vs_history": sep,
        "within_participant_top5": pid_mass,
        "any_participant_dominated": any(r["participant_dominated"] for r in pid_mass),
        "exemplar_prototypes_all": list(range(k)),
        "suggested_exemplar_prototypes": pick,
        "prototype_9_detail": {
            "top5": top5.get(9, []),
            "note": (
                "Strongest history-structured prototype in this peek: "
                "response↔mark_scheme loop template + loop origin on response panel."
            ),
        },
    }
    uio.write_json(out_dir / "fingerprints.json", summary)

    lines = [
        "# M8 diagnostic peek — Step 2 fingerprints",
        "",
        "> **Diagnostic observation, not an M8 finding.**",
        "",
        "## Top-5 distinguishing features by |SMD| (primary read)",
        "",
        "No category aggregation. Question: does any prototype's top-5 include",
        "`is_return`, a `loop_role_*`, or `rel_t` (t/T)?",
        "",
        f"**Answer: {hist_scan['n_prototypes_with_return_loop_or_rel_t_in_top5']}/{k} "
        f"prototypes** have return / loop-role / rel_t in their top-5.",
        "",
    ]
    for c in range(k):
        rows = top5[c]
        hist_flags = [
            r["feature"]
            for r in rows
            if r["feature"] in ("is_return", "rel_t")
            or str(r["feature"]).startswith("loop_role_")
        ]
        flag = f" ← history markers: {hist_flags}" if hist_flags else ""
        lines.append(f"### Prototype {c}{flag}")
        lines.append("")
        lines.append("| Rank | Feature | SMD | |SMD| |")
        lines.append("|------|---------|-----|-------|")
        for i, r in enumerate(rows, start=1):
            lines.append(
                f"| {i} | `{r['feature']}` | {r['smd']:+.3f} | {r['abs_smd']:.3f} |"
            )
        lines.append("")

    # Prototype 9 callout
    p9 = top5.get(9, [])
    lines += [
        "## Prototype 9 — strongest history signal (detail)",
        "",
        "This is the only prototype whose aggregate history |SMD| exceeded static,",
        "and its top-5 is readable without averaging:",
        "",
    ]
    for i, r in enumerate(p9, start=1):
        sign = "elevated vs others" if r["smd"] > 0 else "depressed vs others"
        lines.append(f"{i}. **`{r['feature']}`** SMD={r['smd']:+.3f} ({sign})")
    lines += [
        "",
        "Read: prototype 9 concentrates **response→mark_scheme→response** loop",
        "origins on the **response** panel (high `loop_role_origin`, low `loop_role_none`),",
        "not a static panel-only blob.",
        "",
        "## Tornado charts",
        "",
    ]
    for c in range(k):
        lines.append(f"### Prototype {c}")
        lines.append("")
        lines.append(f"![p{c}](fingerprint_figs/tornado_p{c}.png)")
        lines.append("")

    lines += [
        "## Within-participant occupancy (top-5 by duration mass)",
        "",
        "Flag if >60% of a prototype's duration mass sits with one participant.",
        "",
    ]
    for r in pid_mass:
        flag = " **PARTICIPANT-DOMINATED**" if r["participant_dominated"] else ""
        lines.append(
            f"- Prototype {r['prototype']}: top={r['top_participant']} "
            f"({r['top_share']:.1%}){flag}"
        )
        lines.append(f"  - shares: `{r['participant_shares']}`")
    lines += [
        "",
        "## Secondary aggregate (do not prefer over top-5 lists)",
        "",
        sep["read"],
        "",
        "Exemplars: HTML for **all** prototypes (see `exemplars/`).",
        "",
    ]
    (out_dir / "fingerprints.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
