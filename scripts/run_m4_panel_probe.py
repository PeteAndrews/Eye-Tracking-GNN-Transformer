#!/usr/bin/env python
"""M4 throwaway panel-probe sanity on a few real graphs (not kept for Phase 1)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.gnn import CompactGNN, PanelProbe, mask_panel_features, panel_labels_from_x
from src.utils import io as uio


def run_seed(seed: int, graphs: list[dict], cfg) -> dict:
    torch.manual_seed(seed)
    mcfg = cfg.model
    model = CompactGNN(
        in_dim=int(mcfg.in_dim),
        hidden_dim=int(mcfg.hidden_dim),
        out_dim=int(mcfg.out_dim),
        n_layers=int(mcfg.n_layers),
        n_heads=int(mcfg.n_heads),
        n_relations=int(mcfg.n_relations),
        relation_emb_dim=int(mcfg.relation_emb_dim),
        edge_attr_dim=int(mcfg.edge_attr_dim),
        dropout=float(mcfg.dropout),
        edge_dropout=float(mcfg.edge_dropout),
    )
    probe = PanelProbe(int(mcfg.out_dim), n_classes=int(mcfg.panel_vocab_size))
    opt = torch.optim.Adam(
        list(model.parameters()) + list(probe.parameters()),
        lr=float(cfg.panel_probe.lr),
        weight_decay=float(cfg.panel_probe.weight_decay),
    )
    text_dim = int(mcfg.text_dim)
    n_panel = int(mcfg.panel_vocab_size)
    epochs = int(cfg.panel_probe.epochs)

    model.train()
    probe.train()
    for _ in range(epochs):
        total = 0.0
        n = 0
        for g in graphs:
            x = mask_panel_features(g["x"], text_dim=text_dim, panel_vocab_size=n_panel)
            y = panel_labels_from_x(g["x"], text_dim=text_dim, panel_vocab_size=n_panel)
            n_seg = int(g["n_segments"])
            opt.zero_grad()
            _, h = model(x, g["edge_index"], g["edge_type"], g["edge_attr"])
            loss = F.cross_entropy(probe(h[:n_seg]), y[:n_seg])
            loss.backward()
            opt.step()
            total += float(loss.detach())
            n += 1

    model.eval()
    probe.eval()
    correct = 0
    total_n = 0
    with torch.no_grad():
        for g in graphs:
            x = mask_panel_features(g["x"], text_dim=text_dim, panel_vocab_size=n_panel)
            y = panel_labels_from_x(g["x"], text_dim=text_dim, panel_vocab_size=n_panel)
            n_seg = int(g["n_segments"])
            _, h = model(x, g["edge_index"], g["edge_type"], g["edge_attr"], return_attention=True)
            pred = probe(h[:n_seg]).argmax(-1)
            correct += int((pred == y[:n_seg]).sum())
            total_n += n_seg
    acc = correct / max(total_n, 1)
    attn = model.last_attention
    return {
        "seed": seed,
        "train_loss_last_epoch_mean": total / max(n, 1),
        "panel_acc": acc,
        "attention_finite": bool(attn is not None and torch.isfinite(attn).all()),
        "n_eval_nodes": total_n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--max-graphs", type=int, default=6)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    cfg = OmegaConf.load(root / "configs" / "gnn.yaml")
    gdir = root / str(cfg.paths.graphs_root) / str(cfg.graph_version)
    paths = sorted(gdir.glob("*.pt"))[: args.max_graphs]
    if not paths:
        print(json.dumps({"ok": False, "message": f"no graphs in {gdir}"}))
        return 1
    graphs = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    results = [run_seed(int(s), graphs, cfg) for s in cfg.panel_probe.seeds]
    summary = {
        "ok": all(r["panel_acc"] >= 0.5 for r in results) and all(r["attention_finite"] for r in results),
        "n_graphs": len(graphs),
        "results": results,
        "mean_acc": sum(r["panel_acc"] for r in results) / len(results),
        "note": "Throwaway panel probe — model discarded after M4.",
    }
    out = root / str(cfg.paths.runs_root)
    out.mkdir(parents=True, exist_ok=True)
    uio.write_json(out / "panel_probe_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
