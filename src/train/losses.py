"""Exactly three M6 losses: next-panel CE, next-relation BCE, ranking CE."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F

from src.data.targets import RELATION_NAME_TO_IDX


def active_relation_indices(active_labels: Sequence[str]) -> list[int]:
    return [RELATION_NAME_TO_IDX[n] for n in active_labels]


def relation_weight_tensor(
    active_labels: Sequence[str],
    resolved_clipped: dict[str, float],
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [float(resolved_clipped[n]) for n in active_labels],
        dtype=torch.float32,
        device=device,
    )


def next_panel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax CE; targets use -100 ignore at last step / pad."""
    del mask  # ignore index handles pad/last
    b, t, c = logits.shape
    flat_logits = logits.reshape(b * t, c)
    flat_tgt = targets.reshape(b * t)
    return F.cross_entropy(flat_logits, flat_tgt, ignore_index=-100)


def next_relation_loss(
    logits: torch.Tensor,
    targets_full: torch.Tensor,
    mask: torch.Tensor,
    *,
    active_idx: Sequence[int],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Multi-label BCE with per-label pos weights; only valid transition steps."""
    tgt = targets_full[:, :, list(active_idx)]
    step_ok = mask.clone()
    if step_ok.size(1) > 1:
        step_ok[:, :-1] = mask[:, :-1] & mask[:, 1:]
    step_ok[:, -1] = False
    step_ok = step_ok.unsqueeze(-1)
    logits_m = logits.masked_select(step_ok.expand_as(logits)).view(-1, logits.size(-1))
    tgt_m = tgt.masked_select(step_ok.expand_as(tgt)).view(-1, tgt.size(-1))
    if logits_m.numel() == 0:
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits_m, tgt_m, pos_weight=weights)


def ranking_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    cand_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax-over-candidates CE; positive is the label==1 column when present."""
    has_pos = (labels > 0.5) & cand_mask
    step_ok = has_pos.any(dim=-1)
    if not step_ok.any():
        return scores.new_zeros(())
    lab = labels.masked_fill(~cand_mask, -1.0)
    target = lab.argmax(dim=-1)
    b, t, c = scores.shape
    flat_scores = scores.reshape(b * t, c)
    flat_tgt = target.reshape(b * t)
    flat_ok = step_ok.reshape(b * t)
    flat_scores = flat_scores[flat_ok]
    flat_tgt = flat_tgt[flat_ok]
    # Guard: any non-finite score → zero contrib (should be rare after token sanitise)
    if not torch.isfinite(flat_scores).all():
        flat_scores = torch.nan_to_num(flat_scores, nan=-1e9, posinf=1e9, neginf=-1e9)
    return F.cross_entropy(flat_scores, flat_tgt)


def return_within_horizon_targets(
    node_index: torch.Tensor,
    mask: torch.Tensor,
    *,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary targets + valid mask for return-within-horizon aux loss.

    Vectorised: for each offset 1..H, check whether node_index[t+offset]==node_index[t].
    """
    b, t = node_index.shape
    device = node_index.device
    horizon = int(horizon)
    hits = torch.zeros(b, t, device=device, dtype=torch.bool)
    for off in range(1, horizon + 1):
        if off >= t:
            break
        cur = node_index[:, : t - off]
        fut = node_index[:, off:]
        m = mask[:, : t - off] & mask[:, off:]
        same = (cur == fut) & (cur >= 0) & m
        hits[:, : t - off] |= same
    valid = mask.clone()
    valid[:, -1] = False
    valid &= node_index >= 0
    # last ``horizon`` steps may have truncated windows — still valid labels
    tgt = hits.float()
    return tgt, valid


def return_aux_loss(
    logits: torch.Tensor,
    node_index: torch.Tensor,
    mask: torch.Tensor,
    *,
    horizon: int,
) -> torch.Tensor:
    tgt, valid = return_within_horizon_targets(node_index, mask, horizon=horizon)
    if not valid.any():
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits[valid], tgt[valid])


def loop_role_aux_loss(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    *,
    gnn_out_dim: int,
) -> torch.Tensor:
    """CE on loop_role one-hot stored in token side features (indices 11:15)."""
    # side starts at 2*gnn_out_dim; role one-hot at side[11:15]
    side0 = 2 * int(gnn_out_dim)
    role = tokens[:, :, side0 + 11 : side0 + 15]
    target = role.argmax(dim=-1)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_tgt = target.reshape(-1)
    flat_ok = mask.reshape(-1)
    if not flat_ok.any():
        return logits.new_zeros(())
    return F.cross_entropy(flat_logits[flat_ok], flat_tgt[flat_ok])


def compute_three_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    active_labels: Sequence[str],
    resolved_clipped: dict[str, float],
    loss_weights: Optional[dict[str, float]] = None,
    train_cfg: Any = None,
    gnn_out_dim: int = 128,
) -> dict[str, torch.Tensor]:
    """Return dict with panel, relation, ranking, optional aux, and total."""
    lw = loss_weights or {
        "next_panel": 1.0,
        "next_relation": 1.0,
        "next_node_ranking": 1.0,
    }
    device = outputs["panel_logits"].device
    active_idx = active_relation_indices(active_labels)
    w = relation_weight_tensor(active_labels, resolved_clipped, device)

    l_panel = next_panel_loss(outputs["panel_logits"], batch["next_panel"], batch["mask"])
    l_rel = next_relation_loss(
        outputs["relation_logits"],
        batch["next_relation"],
        batch["mask"],
        active_idx=active_idx,
        weights=w,
    )
    l_rank = ranking_loss(
        outputs["rank_scores"],
        batch["rank_labels"],
        batch["rank_mask"],
    )
    total = (
        float(lw.get("next_panel", 1.0)) * l_panel
        + float(lw.get("next_relation", 1.0)) * l_rel
        + float(lw.get("next_node_ranking", 1.0)) * l_rank
    )
    out = {
        "loss_panel": l_panel,
        "loss_relation": l_rel,
        "loss_ranking": l_rank,
        "loss_total": total,
    }
    if train_cfg is not None:
        losses_cfg = train_cfg.losses
        if bool(losses_cfg.return_aux.enabled) and "return_logits" in outputs:
            horizon = int(
                getattr(train_cfg.diagnostics.D1_return_probe, "horizon_events", 20)
            )
            l_ret = return_aux_loss(
                outputs["return_logits"],
                batch["node_index"],
                batch["mask"],
                horizon=horizon,
            )
            w_ret = float(losses_cfg.return_aux.weight)
            out["loss_return_aux"] = l_ret
            out["loss_total"] = out["loss_total"] + w_ret * l_ret
        if bool(losses_cfg.loop_aux.enabled) and "loop_role_logits" in outputs:
            l_loop = loop_role_aux_loss(
                outputs["loop_role_logits"],
                batch["tokens"],
                batch["mask"],
                gnn_out_dim=gnn_out_dim,
            )
            w_loop = float(losses_cfg.loop_aux.weight)
            out["loss_loop_aux"] = l_loop
            out["loss_total"] = out["loss_total"] + w_loop * l_loop
    return out
