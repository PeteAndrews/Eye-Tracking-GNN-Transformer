"""Return-aux loss accepts pos_weight for class imbalance."""

from __future__ import annotations

import torch

from src.train.losses import return_aux_loss, return_within_horizon_targets


def test_return_within_horizon_targets_basic() -> None:
    # sequence: A A B A  → at t=0 return within H=2 (yes via t=1? same A consecutive;
    # at t=2 (B) return within H=2 via t=3 A? no — B!=A. at t=0 H=3 hits t=3 A.
    node = torch.tensor([[0, 0, 1, 0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    tgt, valid = return_within_horizon_targets(node, mask, horizon=2)
    assert valid[0, 0] and float(tgt[0, 0]) == 1.0  # sees same id at offset 1
    assert valid[0, 2] and float(tgt[0, 2]) == 0.0  # B does not return in window
    assert not valid[0, 3]


def test_return_aux_pos_weight_changes_loss() -> None:
    torch.manual_seed(0)
    logits = torch.zeros(1, 8)
    node = torch.tensor([[0, 1, 0, 1, 0, 1, 0, 1]])
    mask = torch.ones(1, 8, dtype=torch.bool)
    # mostly positives at H=3 on this alternating? 0 returns at +2
    loss_1 = return_aux_loss(logits, node, mask, horizon=3, pos_weight=1.0)
    loss_lo = return_aux_loss(logits, node, mask, horizon=3, pos_weight=0.23)
    # With logits=0, BCE = -log(sigmoid(0))=log(2) per sample; pos_weight scales
    # the positive terms only → loss_lo < loss_1 when positives dominate.
    assert float(loss_lo) < float(loss_1)
