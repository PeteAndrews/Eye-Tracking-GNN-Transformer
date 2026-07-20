"""Per-trial graph diagnostics (node/edge counts, semantic density)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from src.graph.build import RELATION_TO_ID
from src.utils import io as uio


def summarise_graph(graph: dict[str, Any]) -> dict[str, Any]:
    et = graph["edge_type"].cpu().numpy() if hasattr(graph["edge_type"], "cpu") else np.asarray(graph["edge_type"])
    id_to_rel = {v: k for k, v in RELATION_TO_ID.items()}
    counts = Counter(id_to_rel.get(int(t), str(t)) for t in et.tolist())
    n_nodes = int(graph["x"].shape[0])
    n_seg = int(graph.get("n_segments") or 0)
    n_edges = int(et.shape[0])
    sem = counts.get("SEMANTIC_CANDIDATE", 0)
    return {
        "trial_id": graph.get("trial_id"),
        "star_condition": graph.get("star_condition"),
        "n_nodes": n_nodes,
        "n_segments": n_seg,
        "n_panel_nodes": n_nodes - n_seg,
        "n_edges": n_edges,
        "edges_by_type": dict(counts),
        "semantic_edges": sem,
        "text_embedding_dim": graph.get("text_embedding_dim"),
        "graph_version": graph.get("graph_version"),
    }


def write_diagnostics_report(
    summaries: list[dict[str, Any]],
    correspondence: list[dict[str, Any]],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    uio.write_json(out_dir / "graph_summaries.json", summaries)
    uio.write_json(out_dir / "correspondence.json", correspondence)

    total_edges = Counter()
    for s in summaries:
        for k, v in (s.get("edges_by_type") or {}).items():
            total_edges[k] += int(v)

    lines = [
        "# Graph diagnostics — M3",
        "",
        f"Graphs: **{len(summaries)}**",
        "",
        "## Edge totals",
        "",
        "| relation | count |",
        "|---|---|",
    ]
    for k, v in sorted(total_edges.items()):
        lines.append(f"| {k} | {v} |")

    lines += ["", "## Per-graph node/edge counts", "", "| trial | star | segments | nodes | edges | semantic |", "|---|---|---|---|---|---|"]
    for s in sorted(summaries, key=lambda x: (str(x["trial_id"]), str(x["star_condition"]))):
        lines.append(
            f"| {s['trial_id']} | {s['star_condition']} | {s['n_segments']} | "
            f"{s['n_nodes']} | {s['n_edges']} | {s['semantic_edges']} |"
        )

    lines += ["", "## NS↔S correspondence (M3-C1)", ""]
    if not correspondence:
        lines.append("_No eligible star trials checked._")
    else:
        lines += ["| trial | ok | matched | missing | star_conditional_excluded |", "|---|---|---|---|---|"]
        for c in correspondence:
            lines.append(
                f"| {c.get('trial_id')} | {c.get('ok')} | {c.get('n_matched')} | "
                f"{c.get('n_missing')} | {c.get('n_star_conditional_excluded')} |"
            )

    path = out_dir / "REPORT.md"
    uio.write_text(path, "\n".join(lines) + "\n")
    return path
