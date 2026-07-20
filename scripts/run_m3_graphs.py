#!/usr/bin/env python
"""M3: encode texts, build all trial graphs, run NS↔S correspondence + diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from src.data.registry import parse_filename_identity
from src.graph.annotate import dims_for_stem, flag_star_conditional
from src.graph.build import build_graph_dict, save_graph_pt
from src.graph.config_check import assert_encoder_graph_dim_match
from src.graph.correspondence import match_ns_s
from src.graph.diagnostics import summarise_graph, write_diagnostics_report
from src.graph.embeddings import load_or_build_embeddings
from src.text.encoder import load_encoder_from_card
from src.utils import io as uio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument(
        "--max-variants",
        type=int,
        default=0,
        help="If >0, only build this many variants (smoke).",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()

    print("M3: checking encoder/graph dim...", flush=True)
    dim_info = assert_encoder_graph_dim_match(root)
    print(json.dumps(dim_info, indent=2), flush=True)

    data_cfg = OmegaConf.load(root / "configs" / "data.yaml")
    graph_cfg = OmegaConf.load(root / "configs" / "graph.yaml")
    pre_cfg = OmegaConf.load(root / "configs" / "preprocessing.yaml")
    patterns = list(pre_cfg.get("star_conditional_text_patterns") or [])
    graph_version = str(graph_cfg.graph_version)
    data_version = str(data_cfg.data_version)
    meta_dir = root / str(data_cfg.paths.processed_root) / data_version / "metadata"
    graphs_dir = root / str(graph_cfg.paths.graphs_root) / graph_version
    diag_dir = root / str(graph_cfg.paths.diagnostics_root) / graph_version

    dim_path = (
        root
        / str(data_cfg.paths.processed_root)
        / data_version
        / "registry"
        / "document_dimensions.json"
    )
    dim_rows = uio.read_json(dim_path) if dim_path.is_file() else []

    card = root / str(graph_cfg.paths.encoder_card)
    print("M3: loading TextEncoderV1 (CPU for stable batch encode)...", flush=True)
    encoder = load_encoder_from_card(card, device="cpu")

    seg_files = sorted(meta_dir.glob("*__segments.json"))
    if args.max_variants > 0:
        seg_files = seg_files[: args.max_variants]
    print(f"M3: building {len(seg_files)} graphs...", flush=True)

    summaries: list[dict] = []
    by_trial_star: dict[tuple[str, str], list[dict]] = {}
    stem_to_segs: dict[str, list[dict]] = {}

    for i, path in enumerate(seg_files, start=1):
        stem = path.name.replace("__segments.json", "")
        ident = parse_filename_identity(stem)
        segs = flag_star_conditional(uio.read_json(path), patterns)
        stem_to_segs[stem] = segs
        by_trial_star.setdefault((ident.trial_id, ident.star_condition), segs)

        print(f"  [{i}/{len(seg_files)}] {stem} encode...", flush=True)
        emb = load_or_build_embeddings(
            root,
            stem=stem,
            segments=segs,
            graph_version=graph_version,
            encoder=encoder,
            force=args.force_embeddings,
        )
        w, h = dims_for_stem(
            dim_rows,
            trial_id=ident.trial_id,
            star_condition=ident.star_condition,
            stem=stem,
        )
        print(f"  [{i}/{len(seg_files)}] {stem} edges...", flush=True)
        graph = build_graph_dict(
            segs,
            emb,
            trial_id=ident.trial_id,
            star_condition=ident.star_condition,
            graph_cfg=graph_cfg,
            doc_w=w,
            doc_h=h,
        )
        graph["stem"] = stem
        out_pt = graphs_dir / f"{ident.trial_id}__{ident.star_condition}.pt"
        save_graph_pt(graph, out_pt)
        summaries.append(summarise_graph(graph))
        # Free transient tensors between variants (Windows stability)
        del emb, graph
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # NS↔S correspondence for eligible trials
    eligible = list(data_cfg.star_eligible_trials)
    correspondence: list[dict] = []
    for tid in eligible:
        ns = stem_to_segs.get(f"{tid}NS")
        s = stem_to_segs.get(f"{tid}S")
        if ns is None or s is None:
            # may be missing if --max-variants truncated
            if args.max_variants:
                continue
            correspondence.append(
                {
                    "trial_id": tid,
                    "ok": False,
                    "error": "missing NS or S segments",
                }
            )
            continue
        star_ids = {seg["segment_id"] for seg in s if seg.get("is_star_conditional")}
        result = match_ns_s(ns, s, star_conditional_ids=star_ids)
        result["trial_id"] = tid
        correspondence.append(result)

    report = write_diagnostics_report(summaries, correspondence, diag_dir)
    all_ok = all(c.get("ok") for c in correspondence) if correspondence else True
    summary = {
        "ok": len(summaries) == len(seg_files) and (all_ok or bool(args.max_variants)),
        "n_graphs": len(summaries),
        "graph_version": graph_version,
        "graphs_dir": str(graphs_dir),
        "diagnostics": str(report),
        "correspondence_all_ok": all_ok,
        "n_correspondence_fail": sum(1 for c in correspondence if not c.get("ok")),
        "encoder": dim_info,
    }
    uio.write_json(diag_dir / "m3_run_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
