#!/usr/bin/env python
"""M8 diagnostic peek — val-only clustering on a frozen fold0 checkpoint.

NOT real M8. Outputs under reports/m8_diagnostic_peek/. Observations must be
logged to DECISIONS.md as diagnostic only.
"""

from __future__ import annotations

import argparse
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

import torch

from diagnostic.m8_peek.collect import collect_and_save
from diagnostic.m8_peek.fingerprints import run_fingerprints
from diagnostic.m8_peek.structure import run_structure
from diagnostic.m8_peek.exemplar_render import run_exemplars
from src.utils import io as uio


def _write_summary(out_dir: Path, structure: dict, fp: dict | None, ex: dict | None) -> None:
    gate = structure["gate"]
    verdict = gate["verdict"]
    if verdict == "no_structure":
        s1 = (
            f"Structure: no — AMI random={gate['ami_random']:.3f}, "
            f"stratified={gate['ami_stratified']:.3f} (both < 0.5)."
        )
        s2 = "Interpretable: not assessed (stopped after Step 1)."
        s3 = "Participant-idiosyncratic: not assessed."
    else:
        s1 = (
            f"Structure: {verdict} (AMI random={gate['ami_random']:.3f}, "
            f"stratified={gate['ami_stratified']:.3f}; "
            f"nulls ~{structure['stability_random']['null_ami_mean']:.2f}/"
            f"{structure['stability_stratified']['null_ami_mean']:.2f})."
        )
        if fp:
            hist = fp.get("history_in_top5") or {}
            n_hist = hist.get("n_prototypes_with_return_loop_or_rel_t_in_top5", "?")
            s2 = (
                f"Interpretable (top-5 |SMD|, not aggregates): {n_hist}/"
                f"{fp.get('chosen_k', '?')} prototypes include return / loop-role / rel_t "
                f"in their top-5 distinguishing features."
            )
            s3 = (
                "Participant-idiosyncratic: YES — at least one top prototype is "
                "participant-dominated (>60% mass)."
                if fp.get("any_participant_dominated")
                else "Participant-idiosyncratic: no top-5 prototype exceeded the 60% single-participant mass flag."
            )
        else:
            s2 = "Interpretable: Step 2 not run."
            s3 = "Participant-idiosyncratic: Step 2 not run."
    text = "\n".join(
        [
            "# M8 diagnostic peek — SUMMARY",
            "",
            "> Diagnostic observation, not an M8 finding. Val-only GMM fit.",
            "",
            s1,
            "",
            s2,
            "",
            s3,
            "",
            f"- Exemplars written: {bool(ex and ex.get('exemplars'))}",
            "",
        ]
    )
    (out_dir / "SUMMARY.md").write_text(text, encoding="utf-8")


def _append_decisions(repo: Path, structure: dict, fp: dict | None) -> None:
    path = repo / "reports" / "DECISIONS.md"
    lessons = []
    gate = structure["gate"]
    if gate["verdict"] == "sample_fragile":
        lessons.append(
            "Stability was sample-fragile between random-token and stratified-episode "
            "regimes — real M8 should treat BIC k as provisional and prioritise "
            "cross-fold / cross-seed AMI before freezing a prototype set."
        )
    elif gate["verdict"] == "stable":
        lessons.append(
            f"Val-only peek found stable structure at k={structure['chosen_k']}; "
            "still do not transfer this k — real M8 must refit on train per fold."
        )
    else:
        lessons.append(
            "Val embedding space did not support stable GMM modes in this peek "
            "(AMI < 0.5 both regimes); real M8 should verify train-fit stability "
            "before investing in fingerprint / exemplar work."
        )
    if fp and fp.get("any_participant_dominated"):
        lessons.append(
            "At least one high-mass prototype was participant-dominated (>60% "
            "duration mass in one examiner) — prioritise a participant-identity "
            "probe in real M8 before naming / freezing prototypes."
        )
    elif fp:
        lessons.append(
            "Top-5 prototypes were not single-participant dominated in this peek; "
            "still run the participant-identity probe in real M8 as hygiene."
        )
    else:
        lessons.append(
            "Fingerprints skipped; no additional occupancy lesson from this peek."
        )

    block = "\n".join(
        [
            "",
            "---",
            "",
            "## M8-diag — Diagnostic peek on fold0 val embeddings (2026-07-23)",
            "",
            "**Diagnostic observation, not an M8 finding.**",
            "",
            "Val-only GMM on `runs/m6_fullseq_graphbias_return_aux/fold0_seed13` "
            f"(k={structure.get('chosen_k')}, verdict=`{gate['verdict']}`). "
            "Chosen k / labels feed nothing downstream; real M8 fits on train.",
            "",
            "Two lessons to carry to real M8:",
            f"1. {lessons[0]}",
            f"2. {lessons[1]}",
            "",
            "Artefacts: `reports/m8_diagnostic_peek/`.",
            "",
        ]
    )
    prev = path.read_text(encoding="utf-8") if path.is_file() else "# Decisions log\n"
    if "M8-diag — Diagnostic peek" not in prev:
        path.write_text(prev.rstrip() + "\n" + block, encoding="utf-8")


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
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "m8_diagnostic_peek",
    )
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument(
        "--skip-collect",
        action="store_true",
        help="Reuse existing tokens/ under out-dir.",
    )
    ap.add_argument("--force-step2", action="store_true")
    ap.add_argument("--skip-step3", action="store_true")
    ap.add_argument(
        "--from-assignments",
        action="store_true",
        help="Skip collect+structure; regenerate fingerprints + exemplars only.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = out_dir / "tokens"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    fp = None
    ex = None
    if args.from_assignments:
        print("Regenerating fingerprints + exemplars from existing assignments...", flush=True)
        structure = uio.read_json(out_dir / "structure.json")
        fp = run_fingerprints(out_dir, tokens_dir)
        if not args.skip_step3:
            ex = run_exemplars(ROOT, out_dir, per_prototype=1)
    else:
        if not args.skip_collect:
            collect_and_save(
                ROOT,
                Path(args.checkpoint),
                out_dir,
                fold=args.fold,
                seed=args.seed,
                device=device,
            )
        else:
            print(f"Reusing tokens at {tokens_dir}", flush=True)

        structure = run_structure(tokens_dir, out_dir, seed=0)
        if structure["gate"]["proceed_to_step2"] or args.force_step2:
            print("Step 2 fingerprints...", flush=True)
            fp = run_fingerprints(out_dir, tokens_dir)
            if not args.skip_step3:
                sep = fp["static_vs_history"]
                if sep["n_prototypes"] > 0:
                    print("Step 3 exemplars (all prototypes)...", flush=True)
                    ex = run_exemplars(ROOT, out_dir, per_prototype=1)
                else:
                    print("Step 3 skipped — fingerprints empty.", flush=True)
        else:
            print("Steps 2–3 skipped (gate: no_structure).", flush=True)

    _write_summary(out_dir, structure, fp, ex)
    if not args.from_assignments:
        _append_decisions(ROOT, structure, fp)
    print(f"Done. See {out_dir / 'SUMMARY.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
