"""Memory profile with optional graph_relation bias (dense pair_relations)."""

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

warmup_parquet_io(
    ROOT / "data_processed" / "v0_p0" / "fixations" / "P01" / "T01__not_eligible.parquet"
)

import torch
from omegaconf import OmegaConf

from src.models.heads import BehaviourModel
from src.models.transformer import CausalBehaviourTransformer
from src.train.sampling import set_seed


def _build_model(device: torch.device, *, graph_bias: bool) -> BehaviourModel:
    tcfg = OmegaConf.load(ROOT / "configs" / "model_transformer.yaml")
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    train_cfg = OmegaConf.load(ROOT / "configs" / "train.yaml")
    tr = CausalBehaviourTransformer(
        token_dim=int(tcfg.token_dim),
        d_model=int(tcfg.d_model),
        n_layers=int(tcfg.n_layers),
        n_heads=int(tcfg.n_heads),
        ff_mult=int(tcfg.ff_mult),
        dropout=float(tcfg.dropout),
        use_temporal_bias=bool(tcfg.biases.temporal.enabled),
        use_graph_relation_bias=bool(graph_bias),
        use_loop_return_bias=bool(tcfg.biases.loop_return.enabled),
        n_temporal_buckets=int(tcfg.biases.temporal.n_buckets),
    )
    model = BehaviourModel(
        tr,
        n_panels=len(list(dcfg.panel_classes)),
        n_relation_labels=len(list(train_cfg.relation_weights.active_labels)),
        d_model=int(tcfg.d_model),
        node_dim=int(dcfg.gnn_out_dim),
        empty_mode=str(dcfg.empty_space.mode),
    )
    return model.to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-bias", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA required")
    print(f"device={device} graph_bias={args.graph_bias}", flush=True)
    print(torch.cuda.get_device_name(0), flush=True)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"vram_total_gb={total:.2f}", flush=True)

    set_seed(13)
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    n_panel = len(list(dcfg.panel_classes))
    model = _build_model(device, graph_bias=args.graph_bias)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def make_batch(B: int, T: int, n_nodes: int = 80) -> dict:
        if args.graph_bias:
            # Sparse-ish multi-hot: ~1% ones — still allocates full T×T×R storage.
            pair = torch.zeros(B, T, T, 5, device=device)
            # causal lower-triangular band of width 8 on relation 0
            idx = torch.arange(T, device=device)
            for w in range(8):
                i = idx[w:]
                j = idx[: T - w]
                pair[:, i, j, 0] = 1.0
        else:
            pair = torch.zeros(B, 1, 1, 5, device=device)
        return {
            "tokens": torch.randn(B, T, 284, device=device),
            "mask": torch.ones(B, T, dtype=torch.bool, device=device),
            "lengths": torch.full((B,), T, dtype=torch.long, device=device),
            "next_panel": torch.randint(0, n_panel, (B, T), device=device),
            "next_relation": torch.zeros(B, T, 6, device=device),
            "rank_positive": torch.zeros(B, T, dtype=torch.long, device=device),
            "rank_candidates": torch.randint(0, n_nodes, (B, T, 13), device=device),
            "rank_labels": torch.zeros(B, T, 13, device=device),
            "rank_mask": torch.ones(B, T, 13, dtype=torch.bool, device=device),
            "node_index": torch.randint(0, n_nodes, (B, T), device=device),
            "panel_id": torch.randint(0, n_panel, (B, T), device=device),
            "is_empty": torch.zeros(B, T, dtype=torch.bool, device=device),
            "loop_origin_index": torch.full((B, T), -1, dtype=torch.long, device=device),
            "pair_relations": pair,
            "node_x_v": torch.randn(B, n_nodes, 128, device=device),
            "node_h_v": torch.randn(B, n_nodes, 128, device=device),
            "node_mask": torch.ones(B, n_nodes, dtype=torch.bool, device=device),
        }

    results = []
    for T in (256, 512, 1024, 1536):
        for B in (8, 4, 2):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            opt.zero_grad(set_to_none=True)
            try:
                batch = make_batch(B, T)
                out = model(batch)
                logits = out["panel_logits"]
                target = batch["next_panel"].clamp(0, n_panel - 1)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, n_panel),
                    target.reshape(-1),
                ) + out["rank_scores"].pow(2).mean()
                loss.backward()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                reserved = torch.cuda.max_memory_reserved() / (1024**3)
                row = {
                    "T": T,
                    "B": B,
                    "graph_bias": bool(args.graph_bias),
                    "ok": True,
                    "peak_alloc_gb": round(float(peak), 3),
                    "peak_reserved_gb": round(float(reserved), 3),
                }
                print(
                    f"T={T} B={B} OK alloc={peak:.2f}GB reserved={reserved:.2f}GB",
                    flush=True,
                )
                del batch, out, loss, logits, target
            except RuntimeError as e:
                msg = str(e).split("\n")[0][:240]
                row = {
                    "T": T,
                    "B": B,
                    "graph_bias": bool(args.graph_bias),
                    "ok": False,
                    "error": msg,
                }
                print(f"T={T} B={B} FAIL {msg}", flush=True)
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()
            results.append(row)

    out_path = args.out or (
        ROOT
        / "reports"
        / (
            "mem_profile_seqlen_graphbias.json"
            if args.graph_bias
            else "mem_profile_seqlen.json"
        )
    )
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
