"""Causal loop-aware behaviour transformer (PLAN S2-T5 / §10)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.biases import AttentionBiasBundle


class CausalTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = dropout
        self.dropout_mod = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        attn_mask: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, T, D]
        attn_mask: [B, T] True = valid
        attn_bias: optional [B, H, T, T] additive
        """
        b, t, d = x.shape
        h = self.n_heads
        x_n = self.norm1(x)
        qkv = self.qkv(x_n).view(b, t, 3, h, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, T, Hd]
        # Always use SDPA + float mask (manual matmul path AVed on Windows CPU)
        neg = torch.finfo(q.dtype).min / 4
        causal = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        key_ok = attn_mask.unsqueeze(1).unsqueeze(2)
        allow = causal.unsqueeze(0).unsqueeze(0) & key_ok
        float_mask = torch.zeros(b, h, t, t, device=x.device, dtype=q.dtype)
        float_mask = float_mask.masked_fill(~allow, neg)
        if attn_bias is not None:
            # Detach path: add bias outside SDPA to avoid Win CPU SDPA+bias backward AV.
            # Apply as a residual on values via a second cheap path is wrong; instead
            # use manual attention only when bias is present and T is small — here we
            # fold bias into mask but detach the constant part... Actually keep bias
            # in the mask; graph bias is disabled in config on this host.
            float_mask = float_mask + attn_bias
        # Zero-out padded queries via key mask already; also mask query rows
        query_ok = attn_mask.unsqueeze(1).unsqueeze(3)
        float_mask = float_mask.masked_fill(~query_ok, neg)
        ctx = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=float_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.dropout_mod(self.out_proj(ctx))
        x = x + self.ff(self.norm2(x))
        return x


class CausalBehaviourTransformer(nn.Module):
    """Token sequence → causal behavioural embeddings y_t."""

    def __init__(
        self,
        *,
        token_dim: int = 284,
        d_model: int = 192,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        n_relation_types: int = 5,
        n_temporal_buckets: int = 32,
        use_temporal_bias: bool = True,
        use_graph_relation_bias: bool = True,
        use_loop_return_bias: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(token_dim, d_model)
        self.biases = AttentionBiasBundle(
            n_heads,
            n_buckets=n_temporal_buckets,
            n_relations=n_relation_types,
            use_temporal=use_temporal_bias,
            use_graph_relation=use_graph_relation_bias,
            use_loop_return=use_loop_return_bias,
        )
        self.layers = nn.ModuleList(
            [
                CausalTransformerLayer(d_model, n_heads, ff_mult=ff_mult, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        *,
        pair_relations: Optional[torch.Tensor] = None,
        node_index: Optional[torch.Tensor] = None,
        loop_origin_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        tokens: [B, T, token_dim]
        mask: [B, T] bool
        returns y: [B, T, d_model]
        """
        t = tokens.size(1)
        x = self.input_proj(tokens)
        bias = None
        if (
            self.biases.use_temporal
            or self.biases.use_graph_relation
            or self.biases.use_loop_return
        ):
            bias = self.biases(
                t=t,
                mask=mask,
                pair_relations=pair_relations,
                node_index=node_index,
                loop_origin_index=loop_origin_index,
            )
        for layer in self.layers:
            x = layer(x, attn_mask=mask, attn_bias=bias)
        return self.final_norm(x)
