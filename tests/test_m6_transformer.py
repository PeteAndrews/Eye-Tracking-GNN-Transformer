"""M6 tests: causal leakage, three losses, fixture overfit, splits, B=2 padding safety."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.episode_dataset import EpisodeDataset, collate_episodes, load_fixture_episode
from src.data.splits import grouped_participant_folds
from src.data.targets import RELATION_VOCAB
from src.models.heads import BehaviourModel
from src.models.transformer import CausalBehaviourTransformer
from src.train.losses import compute_three_losses
from src.train.loop import build_behaviour_model, build_scheduler
from src.train.relation_weights import resolve_clipped_from_train_cfg
from src.train.sampling import apply_scroll_dropout, set_seed

ROOT = Path(__file__).resolve().parents[1]


def _clipped(train_cfg) -> dict[str, float]:
    return resolve_clipped_from_train_cfg(train_cfg, ROOT)


def _tiny_model(device=None):
    device = device or torch.device("cpu")
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    train_cfg = OmegaConf.load(ROOT / "configs" / "train.yaml")
    tr = CausalBehaviourTransformer(
        token_dim=284,
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )
    model = BehaviourModel(
        tr,
        n_panels=len(list(dcfg.panel_classes)),
        n_relation_labels=len(list(train_cfg.relation_weights.active_labels)),
        d_model=64,
        node_dim=int(dcfg.gnn_out_dim),
        empty_mode=str(dcfg.empty_space.mode),
    )
    return model.to(device), dcfg, train_cfg


def _fixture_episodes():
    return [
        load_fixture_episode(ROOT, "fx01_T99"),
        load_fixture_episode(ROOT, "fx02_T98_star_on"),
    ]


def _fixture_loader(batch_size: int = 1):
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    ds = EpisodeDataset(_fixture_episodes(), dataset_cfg=dcfg, gnn=None, seed=13)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_episodes), dcfg


def _truncate_item(item: dict[str, Any], length: int) -> dict[str, Any]:
    """Truncate a single-episode item to ``length`` (forces padding when batched)."""
    assert length >= 2
    out = dict(item)
    out["length"] = length
    t_keys = (
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
    )
    for k in t_keys:
        out[k] = item[k][:length].clone()
    # last step has no next-* target
    out["next_panel"][-1] = -100
    out["next_relation"][-1] = 0.0
    out["rank_mask"][-1] = False
    pr = item["pair_relations"]
    if pr.ndim == 3 and pr.shape[0] == item["length"]:
        out["pair_relations"] = pr[:length, :length].clone()
    else:
        out["pair_relations"] = pr.clone()
    # clamp loop origins that point past the new end
    loi = out["loop_origin_index"].clone()
    loi[(loi < 0) | (loi >= length)] = -1
    out["loop_origin_index"] = loi
    return out


def _slice_batch_row(batch: dict[str, Any], row: int) -> dict[str, Any]:
    """Keep one batch row (including any pad to the right)."""
    bsz = int(batch["tokens"].size(0))
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.size(0) == bsz:
            out[k] = v[row : row + 1].contiguous()
        elif isinstance(v, list) and len(v) == bsz:
            out[k] = [v[row]]
        else:
            out[k] = v
    return out


def _losses_dict(model, batch, train_cfg) -> dict[str, float]:
    active = list(train_cfg.relation_weights.active_labels)
    clipped = _clipped(train_cfg)
    with torch.no_grad():
        out = model(batch)
        losses = compute_three_losses(
            out, batch, active_labels=active, resolved_clipped=clipped
        )
    return {k: float(v) for k, v in losses.items()}


def test_grouped_folds_no_participant_leak():
    keys = [(f"P{i:02d}", f"T{j}") for i in range(1, 11) for j in range(3)]
    folds = grouped_participant_folds(keys, n_folds=5, seed=13)
    assert len(folds) == 5
    for f in folds:
        assert set(f["train_participants"]).isdisjoint(f["val_participants"])


def test_belongs_to_excluded_from_active_labels():
    train_cfg = OmegaConf.load(ROOT / "configs" / "train.yaml")
    active = list(train_cfg.relation_weights.active_labels)
    assert "BELONGS_TO" not in active
    assert "BELONGS_TO" in RELATION_VOCAB
    assert len(active) == 6


def test_scroll_dropout_zeros_scroll_slice():
    tokens = torch.ones(2, 5, 284)
    out = apply_scroll_dropout(tokens, gnn_out_dim=128, p=1.0)
    assert torch.all(out[:, :, 256 + 15 : 256 + 28] == 0)
    assert torch.all(out[:, :, :256] == 1)


def test_three_losses_backward():
    loader, _ = _fixture_loader()
    model, _, train_cfg = _tiny_model()
    batch = next(iter(loader))
    out = model(batch)
    losses = compute_three_losses(
        out,
        batch,
        active_labels=list(train_cfg.relation_weights.active_labels),
        resolved_clipped=_clipped(train_cfg),
    )
    assert losses["loss_total"].ndim == 0
    losses["loss_total"].backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert any(grads)


def test_causal_mask_no_future_leakage():
    """Perturbing future tokens must leave past outputs unchanged (B=1)."""
    set_seed(0)
    loader, _ = _fixture_loader()
    model, _, _ = _tiny_model()
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        y0 = model.encode(batch).clone()
        lengths = batch["lengths"]
        b = 0
        t_last = int(lengths[b].item()) - 1
        if t_last < 1:
            return
        batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        batch2["tokens"][b, t_last] = batch2["tokens"][b, t_last] + 10.0
        y1 = model.encode(batch2)
        diff = (y0[b, :t_last] - y1[b, :t_last]).abs().max().item()
        assert diff < 1e-5, f"causal leak detected: max past diff={diff}"


def test_causal_mask_no_future_leakage_batched():
    """B=2 different lengths: perturb future+pad of one row; other row + past unchanged."""
    set_seed(0)
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    ds = EpisodeDataset(_fixture_episodes(), dataset_cfg=dcfg, gnn=None, seed=13)
    short = _truncate_item(ds[0], 12)  # row 0
    long = ds[1]  # row 1, length 40
    assert short["length"] < long["length"]
    batch = collate_episodes([short, long])
    model, _, _ = _tiny_model()
    model.eval()

    with torch.no_grad():
        y0 = model.encode(batch).clone()
        L0 = int(batch["lengths"][0].item())
        L1 = int(batch["lengths"][1].item())
        max_t = batch["tokens"].size(1)
        assert L0 < max_t  # real padding present

        batch2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        # Perturb last real token (future relative to earlier steps) on row 0
        batch2["tokens"][0, L0 - 1] = batch2["tokens"][0, L0 - 1] + 10.0
        # Perturb pad positions of row 0
        if L0 < max_t:
            batch2["tokens"][0, L0:] = batch2["tokens"][0, L0:] + 7.0
        y1 = model.encode(batch2)

        past_diff = (y0[0, : L0 - 1] - y1[0, : L0 - 1]).abs().max().item()
        other_diff = (y0[1, :L1] - y1[1, :L1]).abs().max().item()
        assert past_diff < 1e-5, f"row0 past leak: {past_diff}"
        assert other_diff < 1e-5, f"row1 contaminated by row0 pad/future: {other_diff}"


def test_loss_padding_invariance_and_composition():
    """Per-episode losses ignore pads and do not depend on batch partner."""
    set_seed(1)
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    model, _, train_cfg = _tiny_model()
    model.eval()
    ds = EpisodeDataset(_fixture_episodes(), dataset_cfg=dcfg, gnn=None, seed=13)
    item_a = _truncate_item(ds[0], 14)
    item_b = ds[1]  # longer partner
    item_c = _truncate_item(ds[1], 25)  # different partner length

    batch_a = collate_episodes([item_a])
    batch_ab = collate_episodes([item_a, item_b])
    batch_ac = collate_episodes([item_a, item_c])
    assert int(batch_ab["lengths"][0]) < int(batch_ab["tokens"].size(1))

    loss_a = _losses_dict(model, batch_a, train_cfg)
    loss_a_in_ab = _losses_dict(model, _slice_batch_row(batch_ab, 0), train_cfg)
    loss_a_in_ac = _losses_dict(model, _slice_batch_row(batch_ac, 0), train_cfg)

    keys = ("loss_panel", "loss_relation", "loss_ranking")
    for k in keys:
        assert abs(loss_a[k] - loss_a_in_ab[k]) < 1e-5, (
            f"{k}: A alone {loss_a[k]:.6f} vs A in (A,B) {loss_a_in_ab[k]:.6f}"
        )
        assert abs(loss_a[k] - loss_a_in_ac[k]) < 1e-5, (
            f"{k}: A alone {loss_a[k]:.6f} vs A in (A,C) {loss_a_in_ac[k]:.6f}"
        )
        assert abs(loss_a_in_ab[k] - loss_a_in_ac[k]) < 1e-5, (
            f"{k}: A depends on partner — (A,B)={loss_a_in_ab[k]:.6f} (A,C)={loss_a_in_ac[k]:.6f}"
        )


def test_padded_batch_no_nan_inf():
    """No NaN/Inf in outputs, losses, or grads on a B=2 padded batch."""
    set_seed(2)
    dcfg = OmegaConf.load(ROOT / "configs" / "dataset.yaml")
    ds = EpisodeDataset(_fixture_episodes(), dataset_cfg=dcfg, gnn=None, seed=13)
    batch = collate_episodes([_truncate_item(ds[0], 10), ds[1]])
    assert int(batch["lengths"][0]) < int(batch["tokens"].size(1))

    model, _, train_cfg = _tiny_model()
    model.train()
    out = model(batch)
    for name, t in out.items():
        assert torch.isfinite(t).all(), f"non-finite in output[{name}]"

    losses = compute_three_losses(
        out,
        batch,
        active_labels=list(train_cfg.relation_weights.active_labels),
        resolved_clipped=_clipped(train_cfg),
    )
    for name, t in losses.items():
        assert torch.isfinite(t).all(), f"non-finite loss {name}={t}"

    losses["loss_total"].backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"


def test_clip_max_updates_resolved_clipped():
    """Changing clip_max recomputes resolved weights from the M5 table."""
    train_cfg = OmegaConf.load(ROOT / "configs" / "train.yaml")
    train_cfg.relation_weights.clip_max = 10.0
    at_10 = resolve_clipped_from_train_cfg(train_cfg, ROOT)
    assert at_10["SEMANTIC_CANDIDATE"] == 10.0
    assert at_10["PREVIOUS_SEGMENT"] == 10.0
    assert at_10["NEXT_SEGMENT"] == 9.921
    assert at_10["SPATIAL_NEIGHBOUR"] == 4.3165

    train_cfg.relation_weights.clip_max = 8.0
    at_8 = resolve_clipped_from_train_cfg(train_cfg, ROOT)
    assert at_8["SEMANTIC_CANDIDATE"] == 8.0
    assert at_8["PREVIOUS_SEGMENT"] == 8.0
    assert at_8["NEXT_SEGMENT"] == 8.0  # 9.921 also clipped
    assert at_8["EMPTY_SPACE_TRANSITION"] == 8.0
    assert at_8["SPATIAL_NEIGHBOUR"] == 4.3165  # below ceiling
    assert at_8["NO_DIRECT_RELATION"] == 1.619


def test_reduce_lr_on_plateau_steps_down():
    """ReduceLROnPlateau halves LR after ``patience`` non-improving val steps."""
    train_cfg = OmegaConf.create(
        {
            "optim": {
                "lr": 1.0e-3,
                "scheduler": {
                    "name": "reduce_on_plateau",
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 2,
                    "threshold": 0.0,
                    "min_lr": 1.0e-8,
                    "cooldown": 0,
                },
            }
        }
    )
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=1.0e-3)
    sch = build_scheduler(opt, train_cfg)
    assert sch is not None
    assert abs(opt.param_groups[0]["lr"] - 1.0e-3) < 1e-12
    # patience=2 → reduce once num_bad_epochs > 2 (best + 3 non-improving steps)
    sch.step(1.0)  # best
    sch.step(1.0)  # bad=1
    sch.step(1.0)  # bad=2
    assert abs(opt.param_groups[0]["lr"] - 1.0e-3) < 1e-12
    sch.step(1.0)  # bad=3 > patience → factor 0.5
    assert abs(opt.param_groups[0]["lr"] - 5.0e-4) < 1e-12


def test_fixture_overfit_loss_drops(capsys):
    """Two fixture episodes: each loss head drops ≥90% from its start value."""
    set_seed(13)
    loader, _ = _fixture_loader(batch_size=2)
    device = torch.device("cpu")
    _ = build_behaviour_model(ROOT, device)
    model, _, train_cfg = _tiny_model(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    batch = next(iter(loader))
    active = list(train_cfg.relation_weights.active_labels)
    clipped = _clipped(train_cfg)
    heads = ("loss_panel", "loss_relation", "loss_ranking")

    def head_losses() -> dict[str, float]:
        model.eval()
        with torch.no_grad():
            out = model(batch)
            losses = compute_three_losses(
                out, batch, active_labels=active, resolved_clipped=clipped
            )
        return {k: float(losses[k]) for k in heads}

    start = head_losses()
    model.train()
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        losses = compute_three_losses(
            out, batch, active_labels=active, resolved_clipped=clipped
        )
        losses["loss_total"].backward()
        opt.step()
    final = head_losses()

    report = ", ".join(
        f"{k}: {start[k]:.4f} -> {final[k]:.4f} "
        f"({100.0 * (1.0 - final[k] / max(start[k], 1e-12)):.1f}% drop)"
        for k in heads
    )
    # Always emit on pass and fail (pytest otherwise hides stdout on pass)
    with capsys.disabled():
        print(f"overfit per-head: {report}", flush=True)

    failures = []
    for k in heads:
        if start[k] < 1e-8:
            failures.append(f"{k} start already ~0 ({start[k]})")
            continue
        drop = 1.0 - final[k] / start[k]
        if drop < 0.90:
            failures.append(f"{k} drop {drop:.1%} < 90% ({start[k]:.4f} → {final[k]:.4f})")
    assert not failures, " | ".join(failures) + f" || {report}"
