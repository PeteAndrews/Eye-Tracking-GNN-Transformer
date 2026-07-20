#!/usr/bin/env python
"""M5: per-label next-relation frequency table on real fixation episodes.

Feeds M6 class weights for multi-label BCE. Does not require CompactGNN —
only graph edge types between consecutive assigned nodes.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.episode_dataset import _is_empty_segment, discover_real_episodes
from src.data.targets import RELATION_VOCAB, build_edge_relation_lookup, next_relation_multihot
from src.models.tokens import flatten_fixation_row
from src.utils import io as uio


def _count_episode(
    fix_rows: list[dict],
    graph: dict,
    *,
    include_no_direct: bool,
    empty_label: str,
) -> Counter:
    node_ids = list(graph["node_ids"])
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    lookup = build_edge_relation_lookup(graph["edge_index"], graph["edge_type"])
    counts: Counter = Counter()
    prev_empty = None
    prev_idx = None
    for i, row in enumerate(fix_rows):
        empty = _is_empty_segment(row)
        if empty:
            idx = None
        else:
            sid = str(row["segment_id"]).strip()
            idx = id_to_idx.get(sid)
            if idx is None:
                empty = True
        if i > 0:
            vec = next_relation_multihot(
                prev_idx,
                idx,
                edge_lookup=lookup,
                src_is_empty=bool(prev_empty),
                dst_is_empty=bool(empty),
                include_no_direct=include_no_direct,
                empty_label=empty_label,
            )
            for j, name in enumerate(RELATION_VOCAB):
                if vec[j] > 0.5:
                    counts[name] += 1
            counts["__transitions__"] += 1
        prev_empty = empty
        prev_idx = None if empty else idx
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--repo", type=Path, default=ROOT)
    args = ap.parse_args()
    repo = args.repo.resolve()
    cfg = OmegaConf.load(repo / "configs" / "dataset.yaml")
    fix_root = repo / str(cfg.paths.fixations_root)
    graphs_root = repo / str(cfg.paths.graphs_root)
    gver = str(cfg.graph_version)

    triples = discover_real_episodes(fix_root, max_episodes=args.max_episodes)
    if not triples:
        print(f"No fixation episodes under {fix_root}", file=sys.stderr)
        return 1

    graph_cache: dict[str, dict] = {}
    totals: Counter = Counter()
    n_ok = 0
    n_skip = 0
    t0 = time.perf_counter()

    for pid, tid, sc in triples:
        key = f"{tid}__{sc}"
        gpath = graphs_root / gver / f"{key}.pt"
        fpath = fix_root / pid / f"{key}.parquet"
        if not gpath.is_file() or not fpath.is_file():
            n_skip += 1
            continue
        if key not in graph_cache:
            graph_cache[key] = torch.load(gpath, map_location="cpu", weights_only=False)
        df = pd.read_parquet(fpath)
        rows = [flatten_fixation_row(r) for r in df.to_dict(orient="records")]
        totals.update(
            _count_episode(
                rows,
                graph_cache[key],
                include_no_direct=bool(cfg.targets.include_no_direct_relation),
                empty_label=str(cfg.targets.empty_space_transition_label),
            )
        )
        n_ok += 1

    elapsed = time.perf_counter() - t0
    n_trans = int(totals.pop("__transitions__", 0))
    rows_out = []
    for name in RELATION_VOCAB:
        c = int(totals.get(name, 0))
        freq = c / n_trans if n_trans else 0.0
        w = (n_trans / c) if c > 0 else 0.0
        rows_out.append(
            {
                "relation": name,
                "count": c,
                "freq": round(freq, 6),
                "inv_freq_weight": round(w, 4) if c > 0 else None,
            }
        )

    report = {
        "n_episodes": n_ok,
        "n_skipped": n_skip,
        "n_transitions": n_trans,
        "n_graphs_cached": len(graph_cache),
        "elapsed_s": round(elapsed, 3),
        "episodes_per_s": round(n_ok / elapsed, 2) if elapsed > 0 else None,
        "labels": rows_out,
        "graph_version": gver,
    }
    out_json = repo / str(cfg.paths.relation_freq_json)
    out_md = repo / str(cfg.paths.relation_freq_report)
    uio.write_json(out_json, report)

    lines = [
        "# M5 next-relation label frequencies",
        "",
        f"- Episodes counted: **{n_ok}** (skipped {n_skip})",
        f"- Transitions: **{n_trans}**",
        f"- Graph version: `{gver}`",
        f"- Throughput: **{report['episodes_per_s']}** episodes/s",
        "",
        "| relation | count | freq | inv_freq_weight |",
        "|---|---:|---:|---:|",
    ]
    for r in rows_out:
        w = r["inv_freq_weight"] if r["inv_freq_weight"] is not None else "—"
        lines.append(f"| `{r['relation']}` | {r['count']} | {r['freq']:.4f} | {w} |")
    lines.append("")
    lines.append(
        "Weights are raw inverse frequency (n_trans / count) for M6 BCE class weighting; "
        "normalize/clip in `configs/train.yaml` before training."
    )
    lines.append("")
    uio.write_text(out_md, "\n".join(lines) + "\n")
    print(f"Wrote {out_md.relative_to(repo)} and {out_json.relative_to(repo)}")
    print(f"episodes={n_ok} transitions={n_trans} rate={report['episodes_per_s']}/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
