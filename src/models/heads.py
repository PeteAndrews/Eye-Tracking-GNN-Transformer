"""Prediction heads for the three M6 losses."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NextPanelHead(nn.Module):
    def __init__(self, d_model: int, n_panels: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, n_panels)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.fc(y)


class NextRelationHead(nn.Module):
    """Independent logits per active relation label (sigmoid + BCE outside)."""

    def __init__(self, d_model: int, n_labels: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, n_labels)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.fc(y)


class CandidateRankingHead(nn.Module):
    """Score (behaviour state, candidate x_v+h_v) for softmax-over-candidates."""

    def __init__(self, d_model: int, node_dim: int = 128) -> None:
        super().__init__()
        # candidate = concat(x_v, h_v)
        cand_in = 2 * node_dim
        self.y_proj = nn.Linear(d_model, d_model)
        self.c_proj = nn.Linear(cand_in, d_model)
        self.scorer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        y: torch.Tensor,
        cand_x: torch.Tensor,
        cand_h: torch.Tensor,
        cand_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        y: [B, T, D]
        cand_x/h: [B, T, C, node_dim] gathered candidate node features
        cand_mask: [B, T, C]
        returns scores [B, T, C] (masked positions → -inf)
        """
        b, t, c, _ = cand_x.shape
        y_e = self.y_proj(y).unsqueeze(2).expand(b, t, c, -1)
        ch = self.c_proj(torch.cat([cand_x, cand_h], dim=-1))
        scores = self.scorer(torch.cat([y_e, ch], dim=-1)).squeeze(-1)
        # Large negative (not -inf) — safer backward on Windows CPU
        neg = torch.finfo(scores.dtype).min / 2
        scores = scores.masked_fill(~cand_mask, neg)
        return scores


class ReturnHead(nn.Module):
    """Binary logit: return-to-current-segment within horizon (M7 aux)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, 1)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.fc(y).squeeze(-1)


class LoopRoleHead(nn.Module):
    """Softmax over {none, origin, pivot, closure} (M7 aux)."""

    def __init__(self, d_model: int, n_roles: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, n_roles)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.fc(y)


class BehaviourModel(nn.Module):
    """Transformer + three heads; empty-space emb applied here (not in Dataset)."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        n_panels: int,
        n_relation_labels: int,
        d_model: int,
        node_dim: int = 128,
        empty_mode: str = "panel_specific",
    ) -> None:
        super().__init__()
        from src.models.tokens import EmptySpaceEmbedding

        self.transformer = transformer
        self.node_dim = node_dim
        self.empty_mode = empty_mode
        self.empty_emb = EmptySpaceEmbedding(
            mode=empty_mode, dim=node_dim, n_panels=n_panels
        )
        self.panel_head = NextPanelHead(d_model, n_panels)
        self.relation_head = NextRelationHead(d_model, n_relation_labels)
        self.ranking_head = CandidateRankingHead(d_model, node_dim=node_dim)
        # M7 aux heads (always present; losses gated by config)
        self.return_head = ReturnHead(d_model)
        self.loop_role_head = LoopRoleHead(d_model, n_roles=4)

    def _inject_empty_embeddings(self, tokens: torch.Tensor, batch: dict) -> torch.Tensor:
        """Replace zeroed x_v/h_v slots on empty-space tokens with learned emb."""
        if self.empty_mode == "drop":
            return tokens
        is_empty = batch["is_empty"]
        if not is_empty.any():
            return tokens
        out = tokens.clone()
        idx = is_empty.nonzero(as_tuple=False)  # [N, 2] -> (b, t)
        panels = batch["panel_id"][idx[:, 0], idx[:, 1]]
        ev = self.empty_emb(panels)  # [N, node_dim]
        d = self.node_dim
        out[idx[:, 0], idx[:, 1], :d] = ev
        out[idx[:, 0], idx[:, 1], d : 2 * d] = ev
        return out

    def encode(self, batch: dict) -> torch.Tensor:
        tokens = self._inject_empty_embeddings(batch["tokens"], batch)
        return self.transformer(
            tokens,
            batch["mask"],
            pair_relations=batch.get("pair_relations"),
            node_index=batch.get("node_index"),
            loop_origin_index=batch.get("loop_origin_index"),
        )

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        y = self.encode(batch)
        panel_logits = self.panel_head(y)
        rel_logits = self.relation_head(y)
        cand_idx = batch["rank_candidates"]
        b, t, c = cand_idx.shape
        n_nodes = batch["node_x_v"].size(1)
        safe = cand_idx.clamp(min=0)
        batch_ix = torch.arange(b, device=y.device).view(b, 1, 1).expand(b, t, c)
        cand_x = batch["node_x_v"][batch_ix, safe]
        cand_h = batch["node_h_v"][batch_ix, safe]
        valid = batch["rank_mask"] & (cand_idx >= 0) & (cand_idx < n_nodes)
        cand_x = cand_x * valid.unsqueeze(-1).float()
        cand_h = cand_h * valid.unsqueeze(-1).float()
        rank_scores = self.ranking_head(y, cand_x, cand_h, valid)
        return {
            "y": y,
            "panel_logits": panel_logits,
            "relation_logits": rel_logits,
            "rank_scores": rank_scores,
            "return_logits": self.return_head(y),
            "loop_role_logits": self.loop_role_head(y),
        }
