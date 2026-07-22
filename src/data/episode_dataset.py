"""M5 Episode Dataset: fixation tokens + per-step targets for one gaze episode."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from src.data.loops import annotate_loops
from src.data.targets import (
    RELATION_VOCAB,
    build_edge_relation_lookup,
    next_relation_multihot,
    sample_ranking_candidates,
)
from src.models.gnn import CompactGNN
from src.models.tokens import (
    EmptySpaceEmbedding,
    assemble_token,
    fixation_side_features,
    flatten_fixation_row,
    panel_id_for_row,
)
from src.utils import io as uio
from src.utils.arrow_cuda import read_parquet


def _is_empty_segment(row: dict[str, Any]) -> bool:
    sid = row.get("segment_id")
    if sid is None:
        return True
    try:
        if pd.isna(sid):
            return True
    except (TypeError, ValueError):
        pass
    s = str(sid).strip()
    return s in ("", "None", "nan", "<NA>")


def _as_fixation_dicts(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        return [flatten_fixation_row(r) for r in obj]
    if isinstance(obj, pd.DataFrame):
        return [flatten_fixation_row(r) for r in obj.to_dict(orient="records")]
    raise TypeError(type(obj))


def encode_graph_nodes(
    graph: dict[str, Any],
    gnn: CompactGNN,
) -> tuple[np.ndarray, np.ndarray]:
    """Run CompactGNN → (x_v, h_v) numpy [N, d]."""
    gnn.eval()
    with torch.no_grad():
        x = graph["x"].float()
        xv, hv = gnn(
            x,
            graph["edge_index"].long(),
            graph["edge_type"].long(),
            graph["edge_attr"].float(),
        )
    return xv.cpu().numpy().astype(np.float32), hv.cpu().numpy().astype(np.float32)


class EpisodeDataset(Dataset):
    """One item = one (participant, trial, star) episode with tokens + targets."""

    def __init__(
        self,
        episodes: Sequence[dict[str, Any]],
        *,
        dataset_cfg: Any,
        gnn: Optional[CompactGNN] = None,
        empty_emb: Optional[EmptySpaceEmbedding] = None,
        seed: int = 13,
    ) -> None:
        """
        episodes: list of dicts with keys:
          participant_id, trial_id, star_condition,
          fixations (list[dict] or DataFrame),
          graph (torch-saved dict)
        """
        self.cfg = dataset_cfg
        self.episodes = list(episodes)
        self.gnn = gnn
        self.empty_emb = empty_emb or EmptySpaceEmbedding(
            mode=str(dataset_cfg.empty_space.mode),
            dim=int(dataset_cfg.empty_space.embedding_dim),
            n_panels=len(list(dataset_cfg.panel_classes)),
        )
        self.rng = np.random.default_rng(seed)
        self.panel_classes = list(dataset_cfg.panel_classes)
        self.relation_vocab = list(RELATION_VOCAB)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ep = self.episodes[idx]
        fixes = _as_fixation_dicts(ep["fixations"])
        if fixes and "loop_role" not in fixes[0]:
            # Fallback only when parquet lacks P6 annotations. Always use the
            # configured template set — never templates=[] (would zero D2 labels
            # and mute loop_origin_index / attention bias).
            from omegaconf import OmegaConf
            from pathlib import Path as _Path

            pre = OmegaConf.load(
                _Path(__file__).resolve().parents[2] / "configs" / "preprocessing.yaml"
            )
            fixes, _ = annotate_loops(
                fixes,
                templates=[list(t) for t in pre.loops.templates],
                max_loop_gap_events=int(pre.loops.max_loop_gap_events),
                star_condition=str(ep.get("star_condition") or "not_eligible"),
            )

        graph = ep["graph"]
        node_ids: list[str] = list(graph["node_ids"])
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        n_seg = int(graph["n_segments"])
        text_dim = int(graph.get("text_embedding_dim") or 1024)
        text_emb = graph["x"][:n_seg, :text_dim].float().cpu().numpy()

        if self.gnn is not None:
            x_v, h_v = encode_graph_nodes(graph, self.gnn)
        else:
            d = int(self.cfg.gnn_out_dim)
            x_raw = graph["x"].float().cpu().numpy()
            if x_raw.shape[1] >= d:
                x_v = x_raw[:, :d].astype(np.float32)
            else:
                pad = np.zeros((x_raw.shape[0], d - x_raw.shape[1]), dtype=np.float32)
                x_v = np.concatenate([x_raw, pad], axis=1).astype(np.float32)
            h_v = x_v.copy()

        edge_lookup = build_edge_relation_lookup(graph["edge_index"], graph["edge_type"])
        ep_dur = (
            float(fixes[-1]["t_start_ms"]) + float(fixes[-1].get("duration_ms") or 0)
            if fixes
            else 1.0
        )

        tokens: list[np.ndarray] = []
        panel_ids: list[int] = []
        node_indices: list[int] = []
        is_empty_flags: list[bool] = []
        loop_origins: list[int] = []
        max_rank = int(self.cfg.ranking.n_easy) + int(self.cfg.ranking.n_hard) + 1

        for row in fixes:
            empty = _is_empty_segment(row)
            sid = None if empty else str(row["segment_id"]).strip()
            pid = panel_id_for_row(row, self.panel_classes)
            panel_ids.append(pid)
            is_empty_flags.append(bool(empty))
            side = fixation_side_features(row, episode_duration_ms=ep_dur)
            loi = row.get("loop_origin_index")
            try:
                if loi is None or (isinstance(loi, float) and np.isnan(loi)):
                    loop_origins.append(-1)
                else:
                    loop_origins.append(int(loi))
            except (TypeError, ValueError):
                loop_origins.append(-1)

            if empty or sid not in id_to_idx:
                node_indices.append(-1)
                d = int(self.cfg.gnn_out_dim)
                # Placeholder zeros — learned EmptySpaceEmbedding is applied in BehaviourModel
                tok = assemble_token(
                    x_v=np.zeros(d, np.float32),
                    h_v=np.zeros(d, np.float32),
                    side=side,
                    is_empty=True,
                    empty_vec=None,
                )
            else:
                ni = id_to_idx[sid]
                node_indices.append(ni)
                tok = assemble_token(
                    x_v=x_v[ni],
                    h_v=h_v[ni],
                    side=side,
                    is_empty=False,
                )
            tokens.append(tok)

        T = len(fixes)
        n_rel = len(RELATION_VOCAB)
        next_panel = np.full(T, -100, dtype=np.int64)
        next_rel = np.zeros((T, n_rel), dtype=np.float32)
        rank_pos = np.full(T, -1, dtype=np.int64)
        rank_cands = np.full((T, max_rank), -1, dtype=np.int64)
        rank_labs = np.zeros((T, max_rank), dtype=np.float32)
        rank_mask = np.zeros((T, max_rank), dtype=np.bool_)

        visited: set[int] = set()
        for t in range(T):
            if node_indices[t] >= 0:
                visited.add(node_indices[t])
            if t >= T - 1:
                break
            next_panel[t] = panel_ids[t + 1]
            src_e, dst_e = is_empty_flags[t], is_empty_flags[t + 1]
            src_n = None if src_e else node_indices[t]
            dst_n = None if dst_e else node_indices[t + 1]
            next_rel[t] = next_relation_multihot(
                src_n,
                dst_n,
                edge_lookup=edge_lookup,
                src_is_empty=src_e,
                dst_is_empty=dst_e,
                include_no_direct=bool(self.cfg.targets.include_no_direct_relation),
                empty_label=str(self.cfg.targets.empty_space_transition_label),
            )
            pos = dst_n if not dst_e else None
            q = text_emb[src_n] if (src_n is not None and src_n < n_seg) else None
            cands, labels = sample_ranking_candidates(
                positive_node=pos,
                n_segments=n_seg,
                visited=set(visited),
                text_emb=text_emb,
                query_emb=q,
                n_easy=int(self.cfg.ranking.n_easy),
                n_hard=int(self.cfg.ranking.n_hard),
                rng=self.rng,
            )
            rank_pos[t] = int(pos) if pos is not None else -1
            for j, (c, lab) in enumerate(zip(cands, labels)):
                if j >= max_rank:
                    break
                rank_cands[t, j] = int(c)
                rank_labs[t, j] = float(lab)
                rank_mask[t, j] = True

        token_mat = np.stack(tokens, axis=0) if tokens else np.zeros((0, 1), np.float32)
        # Token-pair graph relations (skip when disabled — large and unused)
        from src.graph.build import RELATION_TO_ID
        from collections import defaultdict

        n_graph_rel = len(RELATION_TO_ID)
        build_pair = bool(getattr(self.cfg, "build_pair_relations", True))
        if build_pair:
            pair_rel = np.zeros((T, T, n_graph_rel), dtype=np.float32)
            tokens_by_node: dict[int, list[int]] = defaultdict(list)
            for ti, ni in enumerate(node_indices):
                if ni >= 0:
                    tokens_by_node[ni].append(ti)
            for (si, sj), rels in edge_lookup.items():
                src_toks = tokens_by_node.get(si)
                dst_toks = tokens_by_node.get(sj)
                if not src_toks or not dst_toks:
                    continue
                for rid in rels:
                    if 0 <= rid < n_graph_rel:
                        for i in src_toks:
                            for j in dst_toks:
                                pair_rel[i, j, rid] = 1.0
        else:
            # Tiny placeholder — collate must not allocate T×T when bias unused
            pair_rel = np.zeros((1, 1, n_graph_rel), dtype=np.float32)

        # Optional truncation (configs/model_transformer.yaml → dataset_cfg.max_seq_len)
        max_len = 0
        if hasattr(self.cfg, "max_seq_len") and self.cfg.max_seq_len is not None:
            max_len = int(self.cfg.max_seq_len)
        if max_len > 0 and T > max_len:
            sl = slice(0, max_len)
            token_mat = token_mat[sl]
            next_panel = next_panel[sl]
            next_rel = next_rel[sl]
            rank_pos = rank_pos[sl]
            rank_cands = rank_cands[sl]
            rank_labs = rank_labs[sl]
            rank_mask = rank_mask[sl]
            node_indices = node_indices[sl]
            panel_ids = panel_ids[sl]
            is_empty_flags = is_empty_flags[sl]
            loop_origins = [
                (o if isinstance(o, int) and 0 <= o < max_len else -1) for o in loop_origins[sl]
            ]
            if pair_rel.shape[0] == T:  # full matrix was built
                pair_rel = pair_rel[sl, sl]
            T = max_len
            # last step has no next-* targets
            next_panel[-1] = -100
            next_rel[-1] = 0.0
            rank_mask[-1] = False

        return {
            "participant_id": ep.get("participant_id"),
            "trial_id": ep.get("trial_id"),
            "star_condition": ep.get("star_condition"),
            "tokens": torch.from_numpy(token_mat),
            "next_panel": torch.from_numpy(next_panel),
            "next_relation": torch.from_numpy(next_rel),
            "rank_positive": torch.from_numpy(rank_pos),
            "rank_candidates": torch.from_numpy(rank_cands),
            "rank_labels": torch.from_numpy(rank_labs),
            "rank_mask": torch.from_numpy(rank_mask),
            "node_index": torch.tensor(node_indices, dtype=torch.long),
            "panel_id": torch.tensor(panel_ids, dtype=torch.long),
            "is_empty": torch.tensor(is_empty_flags, dtype=torch.bool),
            "loop_origin_index": torch.tensor(loop_origins, dtype=torch.long),
            "pair_relations": torch.from_numpy(pair_rel),
            "node_x_v": torch.from_numpy(x_v.astype(np.float32)),
            "node_h_v": torch.from_numpy(h_v.astype(np.float32)),
            "n_nodes": int(x_v.shape[0]),
            "n_segments": n_seg,
            "length": T,
            "relation_vocab": self.relation_vocab,
        }


def collate_episodes(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length episodes (and variable node counts)."""
    lengths = [int(b["length"]) for b in batch]
    max_t = max(lengths) if lengths else 0
    d = int(batch[0]["tokens"].shape[-1]) if batch and batch[0]["tokens"].ndim == 2 else 1
    n_rel = int(batch[0]["next_relation"].shape[-1]) if batch else len(RELATION_VOCAB)
    max_c = int(batch[0]["rank_candidates"].shape[-1]) if batch else 13
    n_graph_rel = int(batch[0]["pair_relations"].shape[-1]) if batch else 5
    # If pair_relations are placeholders (1×1), keep collate cheap
    pair_t = int(batch[0]["pair_relations"].shape[0]) if batch else 1
    use_full_pair = pair_t > 1
    node_dims = [int(b["n_nodes"]) for b in batch]
    max_n = max(node_dims) if node_dims else 0
    xv_d = int(batch[0]["node_x_v"].shape[-1]) if batch else 128
    B = len(batch)
    tokens = torch.zeros(B, max_t, d)
    mask = torch.zeros(B, max_t, dtype=torch.bool)
    next_panel = torch.full((B, max_t), -100, dtype=torch.long)
    next_rel = torch.zeros(B, max_t, n_rel)
    rank_pos = torch.full((B, max_t), -1, dtype=torch.long)
    rank_cands = torch.full((B, max_t, max_c), -1, dtype=torch.long)
    rank_labs = torch.zeros(B, max_t, max_c)
    rank_mask = torch.zeros(B, max_t, max_c, dtype=torch.bool)
    node_index = torch.full((B, max_t), -1, dtype=torch.long)
    panel_id = torch.zeros(B, max_t, dtype=torch.long)
    is_empty = torch.zeros(B, max_t, dtype=torch.bool)
    loop_origin = torch.full((B, max_t), -1, dtype=torch.long)
    if use_full_pair:
        pair_rel = torch.zeros(B, max_t, max_t, n_graph_rel)
    else:
        pair_rel = torch.zeros(B, 1, 1, n_graph_rel)
    node_x = torch.zeros(B, max_n, xv_d)
    node_h = torch.zeros(B, max_n, xv_d)
    node_mask = torch.zeros(B, max_n, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = lengths[i]
        tokens[i, :L] = b["tokens"]
        mask[i, :L] = True
        next_panel[i, :L] = b["next_panel"]
        next_rel[i, :L] = b["next_relation"]
        rank_pos[i, :L] = b["rank_positive"]
        rank_cands[i, :L] = b["rank_candidates"]
        rank_labs[i, :L] = b["rank_labels"]
        rank_mask[i, :L] = b["rank_mask"]
        node_index[i, :L] = b["node_index"]
        panel_id[i, :L] = b["panel_id"]
        is_empty[i, :L] = b["is_empty"]
        loop_origin[i, :L] = b["loop_origin_index"]
        if use_full_pair:
            pair_rel[i, :L, :L] = b["pair_relations"]
        n = int(b["n_nodes"])
        node_x[i, :n] = b["node_x_v"]
        node_h[i, :n] = b["node_h_v"]
        node_mask[i, :n] = True
    return {
        "tokens": tokens,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "next_panel": next_panel,
        "next_relation": next_rel,
        "rank_positive": rank_pos,
        "rank_candidates": rank_cands,
        "rank_labels": rank_labs,
        "rank_mask": rank_mask,
        "node_index": node_index,
        "panel_id": panel_id,
        "is_empty": is_empty,
        "loop_origin_index": loop_origin,
        "pair_relations": pair_rel,
        "node_x_v": node_x,
        "node_h_v": node_h,
        "node_mask": node_mask,
        "participant_id": [b["participant_id"] for b in batch],
        "trial_id": [b["trial_id"] for b in batch],
        "star_condition": [b.get("star_condition") for b in batch],
    }


def load_fixture_episode(repo_root: Path, trial_dir: str) -> dict[str, Any]:
    """Build a minimal episode from fixtures/trials/{trial_dir} + synthetic graph."""
    from src.graph.build import build_graph_dict

    root = Path(repo_root)
    tdir = root / "fixtures" / "trials" / trial_dir
    segs = uio.read_json(tdir / "segments.json")
    fixes_raw = uio.read_json(tdir / "fixations.json")
    cfg = OmegaConf.load(root / "configs" / "graph.yaml")
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(segs), int(cfg.text_embedding_dim))).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    star = str(fixes_raw[0].get("star_condition") or "not_eligible")
    graph = build_graph_dict(
        segs,
        emb,
        trial_id=str(segs[0]["trial_id"]),
        star_condition=star,
        graph_cfg=cfg,
        doc_w=800,
        doc_h=600,
    )
    pre = OmegaConf.load(root / "configs" / "preprocessing.yaml")
    templates = [list(t) for t in pre.loops.templates]
    fixes, _ = annotate_loops(
        [flatten_fixation_row(f) for f in fixes_raw],
        templates=templates,
        max_loop_gap_events=int(pre.loops.max_loop_gap_events),
        star_condition=star,
    )
    return {
        "participant_id": fixes[0]["participant_id"],
        "trial_id": fixes[0]["trial_id"],
        "star_condition": star,
        "fixations": fixes,
        "graph": graph,
    }


def load_real_episode(
    *,
    fixations_root: Path,
    graphs_root: Path,
    graph_version: str,
    participant_id: str,
    trial_id: str,
    star_condition: str,
) -> dict[str, Any]:
    """Load one real episode (parquet fixations + .pt graph)."""
    fix_path = Path(fixations_root) / participant_id / f"{trial_id}__{star_condition}.parquet"
    graph_path = Path(graphs_root) / graph_version / f"{trial_id}__{star_condition}.pt"
    if not fix_path.is_file():
        raise FileNotFoundError(fix_path)
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    df = read_parquet(fix_path)
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    return {
        "participant_id": participant_id,
        "trial_id": trial_id,
        "star_condition": star_condition,
        "fixations": df,
        "graph": graph,
    }


def discover_real_episodes(
    fixations_root: Path,
    *,
    max_episodes: Optional[int] = None,
) -> list[tuple[str, str, str]]:
    """Return (participant, trial, star) triples from fixation parquet paths."""
    root = Path(fixations_root)
    out: list[tuple[str, str, str]] = []
    for p in sorted(root.glob("P*/*.parquet")):
        pid = p.parent.name
        stem = p.stem  # T01__not_eligible
        if "__" not in stem:
            continue
        tid, sc = stem.split("__", 1)
        out.append((pid, tid, sc))
        if max_episodes is not None and len(out) >= max_episodes:
            break
    return out


class LazyRealEpisodeDataset(EpisodeDataset):
    """Load fixations/graphs from disk per ``__getitem__`` (shared graph cache).

    Avoids the OOM from eagerly holding 750 episode dicts each with a full
    duplicated ``.pt`` graph in RAM.
    """

    def __init__(
        self,
        keys: Sequence[tuple[str, str, str]],
        *,
        dataset_cfg: Any,
        fixations_root: Path,
        graphs_root: Path,
        graph_version: str,
        gnn: Optional[CompactGNN] = None,
        empty_emb: Optional[EmptySpaceEmbedding] = None,
        seed: int = 13,
    ) -> None:
        # Parent expects episodes list; we keep keys instead and override getitem.
        super().__init__(
            [],
            dataset_cfg=dataset_cfg,
            gnn=gnn,
            empty_emb=empty_emb,
            seed=seed,
        )
        self.keys = list(keys)
        self.fixations_root = Path(fixations_root)
        self.graphs_root = Path(graphs_root)
        self.graph_version = str(graph_version)
        self._graph_cache: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.keys)

    def _graph(self, trial_id: str, star_condition: str) -> dict[str, Any]:
        key = f"{trial_id}__{star_condition}"
        if key not in self._graph_cache:
            path = self.graphs_root / self.graph_version / f"{key}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            self._graph_cache[key] = torch.load(path, map_location="cpu", weights_only=False)
        return self._graph_cache[key]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pid, tid, sc = self.keys[idx]
        fix_path = self.fixations_root / pid / f"{tid}__{sc}.parquet"
        if not fix_path.is_file():
            raise FileNotFoundError(fix_path)
        df = read_parquet(fix_path)
        ep = {
            "participant_id": pid,
            "trial_id": tid,
            "star_condition": sc,
            "fixations": df,
            "graph": self._graph(tid, sc),
        }
        # Temporarily park as sole episode and reuse parent builder
        self.episodes = [ep]
        try:
            return super().__getitem__(0)
        finally:
            self.episodes = []
