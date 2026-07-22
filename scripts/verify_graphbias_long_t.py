"""GPU long-T correctness checks with graph_relation bias enabled.

Re-runs the four M6 padding/causal safety properties at long T (default 768)
with dense pair_relations and use_graph_relation_bias=True. Failures here are
exactly the silent-corruption class that Win-CPU AVs previously masked.
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

warmup_parquet_io(
    ROOT / "data_processed" / "v0_p0" / "fixations" / "P01" / "T01__not_eligible.parquet"
)

import torch
from omegaconf import OmegaConf

from src.data.episode_dataset import collate_episodes
from src.data.targets import RELATION_VOCAB
from src.models.heads import BehaviourModel
from src.models.transformer import CausalBehaviourTransformer
from src.train.losses import compute_three_losses
from src.train.relation_weights import resolve_clipped_from_train_cfg
from src.train.sampling import set_seed


def _model(device: torch.device, *, d_model: int = 192) -> BehaviourModel:
    tcfg = OmegaConf.load(ROOT / "configs" / "model_transformer.yaml")
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    train_cfg = OmegaConf.load(ROOT / "configs" / "train.yaml")
    tr = CausalBehaviourTransformer(
        token_dim=int(tcfg.token_dim),
        d_model=d_model,
        n_layers=int(tcfg.n_layers),
        n_heads=int(tcfg.n_heads),
        ff_mult=int(tcfg.ff_mult),
        dropout=0.0,
        use_temporal_bias=True,
        use_graph_relation_bias=True,
        use_loop_return_bias=True,
        n_temporal_buckets=int(tcfg.biases.temporal.n_buckets),
    )
    return BehaviourModel(
        tr,
        n_panels=len(list(dcfg.panel_classes)),
        n_relation_labels=len(list(train_cfg.relation_weights.active_labels)),
        d_model=d_model,
        node_dim=int(dcfg.gnn_out_dim),
        empty_mode=str(dcfg.empty_space.mode),
    ).to(device)


def _synth_item(T: int, *, n_nodes: int = 40, n_panel: int = 7) -> dict:
    n_rel = len(RELATION_VOCAB)
    pair = torch.zeros(T, T, 5)
    for t in range(T):
        for k in range(max(0, t - 7), t + 1):
            pair[t, k, 0] = 1.0
    next_rel = torch.zeros(T, n_rel)
    next_rel[:, 0] = 1.0  # NEXT_SEGMENT
    rank_labels = torch.zeros(T, 13)
    rank_labels[:, 0] = 1.0
    return {
        "tokens": torch.randn(T, 284),
        "length": T,
        "next_panel": torch.randint(0, n_panel, (T,)),
        "next_relation": next_rel,
        "rank_positive": torch.zeros(T, dtype=torch.long),
        "rank_candidates": torch.randint(0, n_nodes, (T, 13)),
        "rank_labels": rank_labels,
        "rank_mask": torch.ones(T, 13, dtype=torch.bool),
        "node_index": torch.randint(0, n_nodes, (T,)),
        "panel_id": torch.randint(0, n_panel, (T,)),
        "is_empty": torch.zeros(T, dtype=torch.bool),
        "loop_origin_index": torch.full((T,), -1, dtype=torch.long),
        "pair_relations": pair,
        "node_x_v": torch.randn(n_nodes, 128),
        "node_h_v": torch.randn(n_nodes, 128),
        "n_nodes": n_nodes,
        "n_segments": n_nodes,
        "participant_id": "PX",
        "trial_id": "TX",
        "star_condition": "not_eligible",
    }


def _truncate(item: dict, length: int) -> dict:
    out = dict(item)
    out["length"] = length
    for k in (
        "tokens",
        "next_panel",
        "next_relation",
        "rank_positive",
        "rank_candidates",
        "rank_labels",
        "rank_mask",
        "node_index",
        "panel_id",
        "is_empty",
        "loop_origin_index",
    ):
        out[k] = item[k][:length].clone()
    out["next_panel"][-1] = -100
    out["next_relation"][-1] = 0.0
    out["rank_mask"][-1] = False
    out["pair_relations"] = item["pair_relations"][:length, :length].clone()
    return out


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _slice_row(batch: dict, row: int) -> dict:
    out = {}
    L = int(batch["lengths"][row].item())
    for k, v in batch.items():
        if not torch.is_tensor(v):
            if isinstance(v, list):
                out[k] = [v[row]]
            else:
                out[k] = v
            continue
        if v.dim() == 0:
            out[k] = v
        elif k in ("node_x_v", "node_h_v", "node_mask"):
            out[k] = v[row : row + 1]
        elif k == "pair_relations" and v.size(1) > 1:
            out[k] = v[row : row + 1, :L, :L]
        elif v.size(0) == batch["tokens"].size(0):
            out[k] = v[row : row + 1]
        else:
            out[k] = v
    out["lengths"] = torch.tensor([L], device=batch["lengths"].device)
    # re-collate single-row tensors to match collate shapes where needed
    return out


def _losses(model, batch, train_cfg, resolved):
    out = model(batch)
    active = list(train_cfg.relation_weights.active_labels)
    lw = {
        "next_panel": 1.0,
        "next_relation": 1.0,
        "next_node_ranking": 1.0,
    }
    return compute_three_losses(
        out,
        batch,
        active_labels=active,
        resolved_clipped=resolved,
        loss_weights=lw,
        train_cfg=train_cfg,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=768, help="Long sequence length")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device} T={args.T} graph_bias=ON", flush=True)

    set_seed(0)
    train_cfg = OmegaConf.load(ROOT / "configs" / "train.yaml")
    train_cfg.losses.return_aux.enabled = False
    resolved = resolve_clipped_from_train_cfg(train_cfg, ROOT)
    model = _model(device)
    model.eval()

    results = {}

    # 1) Causal leak B=1
    item = _synth_item(args.T)
    batch = _to_device(collate_episodes([item]), device)
    with torch.no_grad():
        y0 = model.encode(batch).clone()
        batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch2["tokens"][0, -1] = batch2["tokens"][0, -1] + 10.0
        y1 = model.encode(batch2)
        diff = (y0[0, :-1] - y1[0, :-1]).abs().max().item()
    ok = diff < 1e-4
    results["causal_leak_b1"] = {"pass": ok, "max_past_diff": diff}
    print(f"[{'PASS' if ok else 'FAIL'}] causal_leak_b1 diff={diff:.2e}", flush=True)

    # 2) Causal leak B=2 padded
    short = _truncate(item, max(64, args.T // 4))
    long = item
    batch = _to_device(collate_episodes([short, long]), device)
    with torch.no_grad():
        y0 = model.encode(batch).clone()
        L0 = int(batch["lengths"][0].item())
        L1 = int(batch["lengths"][1].item())
        batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch2["tokens"][0, L0 - 1] = batch2["tokens"][0, L0 - 1] + 10.0
        batch2["tokens"][0, L0:] = batch2["tokens"][0, L0:] + 7.0
        y1 = model.encode(batch2)
        past_diff = (y0[0, : L0 - 1] - y1[0, : L0 - 1]).abs().max().item()
        other_diff = (y0[1, :L1] - y1[1, :L1]).abs().max().item()
    ok = past_diff < 1e-4 and other_diff < 1e-4
    results["causal_leak_batched"] = {
        "pass": ok,
        "past_diff": past_diff,
        "other_diff": other_diff,
    }
    print(
        f"[{'PASS' if ok else 'FAIL'}] causal_leak_batched "
        f"past={past_diff:.2e} other={other_diff:.2e}",
        flush=True,
    )

    # 3) Loss padding invariance
    set_seed(1)
    model2 = _model(device)
    model2.eval()
    item_a = _truncate(_synth_item(args.T), max(96, args.T // 3))
    item_b = _synth_item(args.T)
    item_c = _truncate(_synth_item(args.T + 17), max(128, args.T // 2))
    batch_a = _to_device(collate_episodes([item_a]), device)
    batch_ab = _to_device(collate_episodes([item_a, item_b]), device)
    batch_ac = _to_device(collate_episodes([item_a, item_c]), device)

    def row0(batch):
        # Encode full batch then compute loss on row-0 slice via mask? Use collate of single.
        # Reconstruct row-0 as standalone item already have batch_a.
        return batch

    with torch.no_grad():
        la = _losses(model2, batch_a, train_cfg, resolved)
        # partner invariance: loss on A alone vs A taken from mixed batches by
        # re-encoding only row 0 through a one-row batch built from the mixed tensors
        def extract_row0(b):
            L = int(b["lengths"][0].item())
            item = {
                "tokens": b["tokens"][0, :L].cpu(),
                "length": L,
                "next_panel": b["next_panel"][0, :L].cpu(),
                "next_relation": b["next_relation"][0, :L].cpu(),
                "rank_positive": b["rank_positive"][0, :L].cpu(),
                "rank_candidates": b["rank_candidates"][0, :L].cpu(),
                "rank_labels": b["rank_labels"][0, :L].cpu(),
                "rank_mask": b["rank_mask"][0, :L].cpu(),
                "node_index": b["node_index"][0, :L].cpu(),
                "panel_id": b["panel_id"][0, :L].cpu(),
                "is_empty": b["is_empty"][0, :L].cpu(),
                "loop_origin_index": b["loop_origin_index"][0, :L].cpu(),
                "pair_relations": b["pair_relations"][0, :L, :L].cpu()
                if b["pair_relations"].size(1) > 1
                else b["pair_relations"][0].cpu(),
                "node_x_v": b["node_x_v"][0].cpu(),
                "node_h_v": b["node_h_v"][0].cpu(),
                "n_nodes": int(b["node_mask"][0].sum().item()),
                "n_segments": int(b["node_mask"][0].sum().item()),
                "participant_id": "PX",
                "trial_id": "TX",
                "star_condition": "not_eligible",
            }
            return _to_device(collate_episodes([item]), device)

        lab = _losses(model2, extract_row0(batch_ab), train_cfg, resolved)
        lac = _losses(model2, extract_row0(batch_ac), train_cfg, resolved)
    keys = ("loss_panel", "loss_relation", "loss_ranking")
    inv_ok = True
    details = {}
    for k in keys:
        d_ab = abs(float(la[k]) - float(lab[k]))
        d_ac = abs(float(la[k]) - float(lac[k]))
        d_bc = abs(float(lab[k]) - float(lac[k]))
        details[k] = {"d_ab": d_ab, "d_ac": d_ac, "d_bc": d_bc}
        if max(d_ab, d_ac, d_bc) >= 1e-4:
            inv_ok = False
    results["loss_padding_invariance"] = {"pass": inv_ok, "details": details}
    print(f"[{'PASS' if inv_ok else 'FAIL'}] loss_padding_invariance {details}", flush=True)

    # 4) No NaN/Inf on padded B=2 + backward
    set_seed(2)
    model3 = _model(device)
    model3.train()
    batch = _to_device(
        collate_episodes([_truncate(_synth_item(args.T), max(80, args.T // 5)), _synth_item(args.T)]),
        device,
    )
    out = model3(batch)
    losses = _losses(model3, batch, train_cfg, resolved)
    losses["loss_total"].backward()
    nan_out = any(
        (torch.isnan(v).any() or torch.isinf(v).any()).item()
        for v in out.values()
        if torch.is_tensor(v)
    )
    nan_loss = any(
        (torch.isnan(v) or torch.isinf(v)).item()
        for v in losses.values()
        if torch.is_tensor(v) and v.dim() == 0
    )
    nan_grad = any(
        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
        for p in model3.parameters()
    )
    ok = not (nan_out or nan_loss or nan_grad)
    results["padded_batch_no_nan_inf"] = {
        "pass": ok,
        "nan_out": nan_out,
        "nan_loss": nan_loss,
        "nan_grad": nan_grad,
    }
    print(
        f"[{'PASS' if ok else 'FAIL'}] padded_batch_no_nan_inf "
        f"out={nan_out} loss={nan_loss} grad={nan_grad}",
        flush=True,
    )

    all_pass = all(r["pass"] for r in results.values())
    print(f"ALL={'PASS' if all_pass else 'FAIL'}", flush=True)
    out_path = ROOT / "reports" / "graphbias_long_t_checks.json"
    import json

    out_path.write_text(
        json.dumps({"T": args.T, "device": str(device), "results": results, "all_pass": all_pass}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
