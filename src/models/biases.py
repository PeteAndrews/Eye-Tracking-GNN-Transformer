"""Additive attention biases for the causal behaviour transformer (§10)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TemporalRelativeBias(nn.Module):
    """Learned bias from relative distance buckets (query i attends to key j ≤ i)."""

    def __init__(self, n_heads: int, n_buckets: int = 32) -> None:
        super().__init__()
        self.n_buckets = n_buckets
        self.emb = nn.Embedding(n_buckets, n_heads)

    def forward(self, t: int, device: torch.device) -> torch.Tensor:
        """Return [H, T, T] bias (added to attention logits)."""
        idx = torch.arange(t, device=device)
        dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).clamp(min=0, max=self.n_buckets - 1)
        return self.emb(dist).permute(2, 0, 1)


class GraphRelationBias(nn.Module):
    """Per-relation learned scalar when viewed nodes share a directed edge."""

    def __init__(self, n_heads: int, n_relations: int = 5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_relations, n_heads))

    def forward(self, pair_relations: torch.Tensor) -> torch.Tensor:
        """
        pair_relations: [B, T, T, R] float multi-hot
        returns: [B, H, T, T]
        """
        return torch.einsum("bijr,rh->bhij", pair_relations, self.weight)


class LoopReturnBias(nn.Module):
    """Bias closure tokens toward loop origin; same-segment predecessors."""

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.origin_bias = nn.Parameter(torch.zeros(n_heads))
        self.same_seg_bias = nn.Parameter(torch.zeros(n_heads))

    def forward(
        self,
        node_index: torch.Tensor,
        loop_origin_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        node_index / loop_origin_index: [B, T]
        mask: [B, T] True = valid token
        returns: [B, H, T, T]
        """
        b, t = node_index.shape
        device = node_index.device
        h = self.origin_bias.numel()

        # Closure → origin (vectorised, no in-place scatter — AV-prone on Win CPU)
        oi = loop_origin_index.clamp(min=0, max=t - 1)
        arange_t = torch.arange(t, device=device).unsqueeze(0)
        valid_o = (
            (loop_origin_index >= 0)
            & (loop_origin_index < t)
            & (loop_origin_index <= arange_t)
            & mask
            & torch.gather(mask, 1, oi)
        )
        key_oh = torch.nn.functional.one_hot(oi, num_classes=t).to(dtype=torch.float32)
        key_oh = key_oh * valid_o.unsqueeze(-1).float()
        origin = torch.einsum("bqk,h->bhqk", key_oh, self.origin_bias)

        # Same-segment predecessors (causal, exclude diagonal)
        ni = node_index.unsqueeze(2)
        nj = node_index.unsqueeze(1)
        same = (ni == nj) & (ni >= 0) & mask.unsqueeze(2) & mask.unsqueeze(1)
        causal = torch.tril(
            torch.ones(t, t, device=device, dtype=torch.bool), diagonal=-1
        )
        same = same & causal.unsqueeze(0)
        same_b = torch.einsum("bqk,h->bhqk", same.float(), self.same_seg_bias)
        return origin + same_b


class AttentionBiasBundle(nn.Module):
    """Sum of enabled temporal / graph-relation / loop-return biases."""

    def __init__(
        self,
        n_heads: int,
        *,
        n_buckets: int = 32,
        n_relations: int = 5,
        use_temporal: bool = True,
        use_graph_relation: bool = True,
        use_loop_return: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.use_graph_relation = use_graph_relation
        self.use_loop_return = use_loop_return
        self.n_heads = n_heads
        self.temporal = TemporalRelativeBias(n_heads, n_buckets) if use_temporal else None
        self.graph = GraphRelationBias(n_heads, n_relations) if use_graph_relation else None
        self.loop = LoopReturnBias(n_heads) if use_loop_return else None

    def forward(
        self,
        *,
        t: int,
        mask: torch.Tensor,
        pair_relations: Optional[torch.Tensor] = None,
        node_index: Optional[torch.Tensor] = None,
        loop_origin_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Additive bias [B, H, T, T] (B inferred from mask)."""
        b = mask.size(0)
        device = mask.device
        out = torch.zeros(b, self.n_heads, t, t, device=device)
        if self.temporal is not None:
            out = out + self.temporal(t, device).unsqueeze(0)
        if self.graph is not None and pair_relations is not None:
            out = out + self.graph(pair_relations[:, :t, :t])
        if (
            self.loop is not None
            and node_index is not None
            and loop_origin_index is not None
        ):
            out = out + self.loop(node_index[:, :t], loop_origin_index[:, :t], mask[:, :t])
        return out
