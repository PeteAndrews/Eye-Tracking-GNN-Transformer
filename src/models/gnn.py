"""Compact edge-aware GAT (M4) — GATv2-style attention in pure PyTorch.

Outputs:
  x_v — projected original node features
  h_v — graph-contextualised representation after 2 layers

Relation-type embeddings and numeric edge attributes are injected into the
attention logits. Residuals and edge dropout follow the research plan §7.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareGATConv(nn.Module):
    """Single GATv2-style convolution with edge features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        n_heads: int = 4,
        edge_dim: int = 21,
        dropout: float = 0.1,
        residual: bool = True,
    ) -> None:
        super().__init__()
        if out_dim % n_heads != 0:
            raise ValueError("out_dim must be divisible by n_heads")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.dropout = dropout
        self.residual = residual

        self.lin_l = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_e = nn.Linear(edge_dim, out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(1, n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.att)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) if residual else None
        self.norm = nn.LayerNorm(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: [N, in_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_dim]
        """
        n = x.size(0)
        if edge_index.numel() == 0:
            out = self.bias.expand(n, -1)
            if self.res_proj is not None:
                out = out + self.res_proj(x)
            out = self.norm(out)
            attn = torch.zeros(0, device=x.device) if return_attention else None
            return out, attn

        src, dst = edge_index[0], edge_index[1]
        h_l = self.lin_l(x).view(n, self.n_heads, self.head_dim)
        h_r = self.lin_r(x).view(n, self.n_heads, self.head_dim)
        e = self.lin_e(edge_attr).view(-1, self.n_heads, self.head_dim)

        # GATv2: a^T LeakyReLU(h_l[dst] + h_r[src] + e)
        msg = F.leaky_relu(h_l[dst] + h_r[src] + e, negative_slope=0.2)
        logits = (msg * self.att).sum(dim=-1)  # [E, heads]

        # Softmax over incoming edges per destination node (and head)
        attn = self._softmax_per_dst(logits, dst, n)  # [E, heads]
        attn_drop = F.dropout(attn, p=self.dropout, training=self.training)

        # Aggregate source messages
        src_h = h_r[src]  # [E, heads, head_dim]
        weighted = src_h * attn_drop.unsqueeze(-1)
        out = torch.zeros(n, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, weighted)
        out = out.reshape(n, self.out_dim) + self.bias

        if self.res_proj is not None:
            out = out + self.res_proj(x)
        out = self.norm(F.elu(out))
        out = F.dropout(out, p=self.dropout, training=self.training)

        if return_attention:
            return out, attn.mean(dim=-1)  # [E] mean over heads
        return out, None

    @staticmethod
    def _softmax_per_dst(logits: torch.Tensor, dst: torch.Tensor, n: int) -> torch.Tensor:
        """Stable softmax of [E, H] grouped by destination node."""
        # Subtract max per (dst, head)
        max_per = torch.full(
            (n, logits.size(1)),
            float("-inf"),
            device=logits.device,
            dtype=logits.dtype,
        )
        max_per.scatter_reduce_(0, dst.unsqueeze(1).expand_as(logits), logits, reduce="amax", include_self=True)
        logits = logits - max_per[dst]
        exp = logits.exp()
        denom = torch.zeros(n, logits.size(1), device=logits.device, dtype=logits.dtype)
        denom.index_add_(0, dst, exp)
        return exp / (denom[dst] + 1e-12)


class CompactGNN(nn.Module):
    """2-layer edge-aware GAT returning (x_v, h_v)."""

    def __init__(
        self,
        *,
        in_dim: int = 1051,
        hidden_dim: int = 128,
        out_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        n_relations: int = 5,
        relation_emb_dim: int = 16,
        edge_attr_dim: int = 5,
        dropout: float = 0.1,
        edge_dropout: float = 0.2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.edge_dropout = edge_dropout
        self.relation_emb = nn.Embedding(n_relations, relation_emb_dim)
        edge_dim = relation_emb_dim + edge_attr_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        layers: list[EdgeAwareGATConv] = []
        for i in range(n_layers):
            ind = hidden_dim
            outd = out_dim if i == n_layers - 1 else hidden_dim
            layers.append(
                EdgeAwareGATConv(
                    ind,
                    outd,
                    n_heads=n_heads,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    residual=residual,
                )
            )
        self.layers = nn.ModuleList(layers)
        self._last_attention: Optional[torch.Tensor] = None

    def _edge_features(
        self,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        rel = self.relation_emb(edge_type.long())
        return torch.cat([rel, edge_attr], dim=-1)

    def _drop_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (not self.training) or self.edge_dropout <= 0 or edge_index.size(1) == 0:
            return edge_index, edge_type, edge_attr
        keep = torch.rand(edge_index.size(1), device=edge_index.device) >= self.edge_dropout
        if not keep.any():
            keep[0] = True
        return edge_index[:, keep], edge_type[keep], edge_attr[keep]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attr: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_v : projected original features [N, hidden_dim]
        h_v : contextualised features [N, out_dim]
        """
        edge_index, edge_type, edge_attr = self._drop_edges(edge_index, edge_type, edge_attr)
        x = x.float()
        edge_attr = edge_attr.float()
        ef = self._edge_features(edge_type, edge_attr)

        x_v = self.input_proj(x)
        h = x_v
        attn_last = None
        for i, layer in enumerate(self.layers):
            want_attn = return_attention and (i == len(self.layers) - 1)
            h, attn = layer(h, edge_index, ef, return_attention=want_attn)
            if want_attn:
                attn_last = attn
        self._last_attention = attn_last
        return x_v, h

    @property
    def last_attention(self) -> Optional[torch.Tensor]:
        """Per-edge attention from the final layer (mean over heads), if requested."""
        return self._last_attention


def mask_panel_features(
    x: torch.Tensor,
    *,
    text_dim: int = 1024,
    panel_vocab_size: int = 6,
) -> torch.Tensor:
    """Zero the panel one-hot slice so panel identity must come via message passing."""
    out = x.clone()
    out[:, text_dim : text_dim + panel_vocab_size] = 0.0
    return out


def panel_labels_from_x(
    x: torch.Tensor,
    *,
    text_dim: int = 1024,
    panel_vocab_size: int = 6,
    n_segments: Optional[int] = None,
) -> torch.Tensor:
    """Argmax over panel one-hot; panel abstract nodes use the same slice."""
    panel = x[:, text_dim : text_dim + panel_vocab_size]
    labels = panel.argmax(dim=-1)
    return labels


class PanelProbe(nn.Module):
    """Linear panel classifier on h_v (throwaway M4 sanity head)."""

    def __init__(self, in_dim: int, n_classes: int = 6) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)
