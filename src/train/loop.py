"""M6 training loop: AdamW, ReduceLROnPlateau, early stop, checkpointing."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.train.losses import compute_three_losses
from src.train.relation_weights import resolve_clipped_from_train_cfg
from src.train.sampling import apply_scroll_dropout, set_seed
from src.utils import io as uio
from src.utils.tracking import Tracker, flatten_config


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: Any
) -> Optional[ReduceLROnPlateau]:
    """Build LR scheduler from ``train_cfg.optim.scheduler`` (or None)."""
    sch = getattr(train_cfg.optim, "scheduler", None)
    if sch is None:
        return None
    name = str(getattr(sch, "name", "none")).strip().lower()
    if name in ("", "none", "null"):
        return None
    if name not in ("reduce_on_plateau", "reducelronplateau"):
        raise ValueError(f"Unknown optim.scheduler.name: {name!r}")
    return ReduceLROnPlateau(
        optimizer,
        mode=str(getattr(sch, "mode", "min")),
        factor=float(getattr(sch, "factor", 0.5)),
        patience=int(getattr(sch, "patience", 3)),
        threshold=float(getattr(sch, "threshold", 1e-4)),
        min_lr=float(getattr(sch, "min_lr", 1e-6)),
        cooldown=int(getattr(sch, "cooldown", 0)),
    )


def _git_hash(repo: Path) -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo),
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def build_behaviour_model(repo: Path, device: torch.device) -> torch.nn.Module:
    """Return BehaviourModel (includes EmptySpaceEmbedding) on ``device``."""
    from src.models.heads import BehaviourModel
    from src.models.transformer import CausalBehaviourTransformer

    tcfg = OmegaConf.load(repo / "configs" / "model_transformer.yaml")
    dcfg = OmegaConf.load(repo / "configs" / "dataset.yaml")
    train_cfg = OmegaConf.load(repo / "configs" / "train.yaml")
    active = list(train_cfg.relation_weights.active_labels)
    tr = CausalBehaviourTransformer(
        token_dim=int(tcfg.token_dim),
        d_model=int(tcfg.d_model),
        n_layers=int(tcfg.n_layers),
        n_heads=int(tcfg.n_heads),
        ff_mult=int(tcfg.ff_mult),
        dropout=float(tcfg.dropout),
        use_temporal_bias=bool(tcfg.biases.temporal.enabled),
        use_graph_relation_bias=bool(tcfg.biases.graph_relation.enabled),
        use_loop_return_bias=bool(tcfg.biases.loop_return.enabled),
        n_temporal_buckets=int(tcfg.biases.temporal.n_buckets),
    )
    model = BehaviourModel(
        tr,
        n_panels=len(list(dcfg.panel_classes)),
        n_relation_labels=len(active),
        d_model=int(tcfg.d_model),
        node_dim=int(dcfg.gnn_out_dim),
        empty_mode=str(dcfg.empty_space.mode),
    )
    return model.to(device)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    train_cfg: Any,
    dataset_cfg: Any,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
    resolved_clipped: dict[str, float],
    max_steps: Optional[int] = None,
    start_step: int = 0,
    checkpoint_every: int = 0,
    checkpoint_fn: Optional[Any] = None,
) -> dict[str, float]:
    model.train(train)
    active = list(train_cfg.relation_weights.active_labels)
    clipped = resolved_clipped
    lw = {
        "next_panel": float(train_cfg.losses.next_panel.weight),
        "next_relation": float(train_cfg.losses.next_relation.weight),
        "next_node_ranking": float(train_cfg.losses.next_node_ranking.weight),
    }
    totals = {
        "loss_total": 0.0,
        "loss_panel": 0.0,
        "loss_relation": 0.0,
        "loss_ranking": 0.0,
    }
    if bool(train_cfg.losses.return_aux.enabled):
        totals["loss_return_aux"] = 0.0
    if bool(train_cfg.losses.loop_aux.enabled):
        totals["loss_loop_aux"] = 0.0
    n = 0
    trained = 0
    n_batches = len(loader)
    if train and start_step > 0:
        print(
            f"  resuming mid-epoch: skipping batches 0..{start_step - 1} "
            f"of {n_batches}",
            flush=True,
        )
    for step, batch in enumerate(loader):
        # Resume mid-epoch: skip already-consumed batches
        if step < start_step:
            if train and step > 0 and step % 25 == 0:
                print(f"  skip {step}/{n_batches} ...", flush=True)
            continue
        if max_steps is not None and trained >= max_steps:
            break
        batch = _batch_to_device(batch, device)
        if train:
            batch["tokens"] = apply_scroll_dropout(
                batch["tokens"],
                gnn_out_dim=int(dataset_cfg.gnn_out_dim),
                p=float(train_cfg.scroll_feature_dropout_p),
            )
        if train and trained % 25 == 0:
            print(
                f"  train step {step}/{n_batches}  "
                f"T={int(batch['lengths'].max())}  "
                f"B={int(batch['tokens'].size(0))}",
                flush=True,
            )
        outputs = model(batch)
        losses = compute_three_losses(
            outputs,
            batch,
            active_labels=active,
            resolved_clipped=clipped,
            loss_weights=lw,
            train_cfg=train_cfg,
            gnn_out_dim=int(dataset_cfg.gnn_out_dim),
        )
        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg.optim.grad_clip_norm)
            )
            optimizer.step()
            if (
                checkpoint_every > 0
                and checkpoint_fn is not None
                and (trained + 1) % checkpoint_every == 0
            ):
                checkpoint_fn(step + 1)
        for k in totals:
            totals[k] += float(losses[k].detach())
        n += 1
        trained += 1
        if train and trained % 50 == 0:
            import gc

            gc.collect()
    out = {k: (v / n if n else 0.0) for k, v in totals.items()}
    out["_steps_done"] = start_step + trained
    out["_trained_this_call"] = trained
    out["_n_batches"] = float(n_batches)
    return out


def train_run(
    *,
    repo: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    seed: int,
    fold: int,
    max_epochs: int,
    device: Optional[torch.device] = None,
    tracker_backend: Optional[str] = None,
    max_steps: Optional[int] = None,
    checkpoint_every: int = 100,
    resume: bool = True,
    batch_size: Optional[int] = None,
    truncation_stats: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Train with early stopping on grouped-val total loss; write run artefacts.

    On Windows CPU (CUDA torch wheel without GPU), full epochs can AV after
    ~400–500 steps. Pass ``max_steps`` (e.g. 200) and re-invoke to resume from
    ``checkpoint_last.pt``.

    Mid-epoch ``global_step`` skipping is only used in chunked ``max_steps``
    mode. Full-epoch runs restore weights/optim but always start batches at 0
    (avoids a silent hang when an old B=1 checkpoint's step exceeds B=8
    batches/epoch).
    """
    repo = Path(repo)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    set_seed(seed)
    train_cfg = OmegaConf.load(repo / "configs" / "train.yaml")
    dataset_cfg = OmegaConf.load(repo / "configs" / "dataset.yaml")
    tcfg = OmegaConf.load(repo / "configs" / "model_transformer.yaml")
    resolved_clipped = resolve_clipped_from_train_cfg(train_cfg, repo)
    # Snapshot resolved weights into the saved config for run reproducibility.
    train_cfg.relation_weights.resolved_clipped = OmegaConf.create(resolved_clipped)
    print(
        f"Relation weights (clip_max={float(train_cfg.relation_weights.clip_max)}): "
        + ", ".join(f"{k}={v:g}" for k, v in resolved_clipped.items()),
        flush=True,
    )

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "viz").mkdir(exist_ok=True)

    inferred_bs = batch_size
    if inferred_bs is None:
        inferred_bs = int(getattr(train_loader, "batch_size", None) or 1)

    meta = {
        "seed": seed,
        "fold": fold,
        "git_hash": _git_hash(repo),
        "device": str(device),
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "batch_size": inferred_bs,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "relation_weights_resolved": resolved_clipped,
        "max_seq_len": int(tcfg.max_seq_len),
        "graph_relation_bias": bool(tcfg.biases.graph_relation.enabled),
        "truncation": truncation_stats,
    }
    uio.write_json(run_dir / "run_meta.json", meta)
    OmegaConf.save(train_cfg, run_dir / "train_config.yaml")
    OmegaConf.save(tcfg, run_dir / "model_config.yaml")

    backend = tracker_backend if tracker_backend is not None else str(train_cfg.tracking.backend)
    tracker = Tracker(backend, run_dir=run_dir)
    tag_cfg = dict(getattr(train_cfg.tracking, "tags", {}) or {})
    tracker.set_tags(
        {
            "milestone": str(tag_cfg.get("milestone", "M6")),
            "ablation_id": str(tag_cfg.get("ablation_id", "baseline")),
            "fold": str(fold),
            "seed": str(seed),
        }
    )
    tracker.log_params(flatten_config(train_cfg))

    model = build_behaviour_model(repo, device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.optim.lr),
        weight_decay=float(train_cfg.optim.weight_decay),
    )
    scheduler = build_scheduler(opt, train_cfg)

    last_path = run_dir / "checkpoint_last.pt"
    start_step = 0
    start_epoch = 1
    n_train_batches = len(train_loader)
    if resume and last_path.is_file():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                opt.load_state_dict(ckpt["optimizer"])
            except (ValueError, KeyError) as exc:
                print(f"Optimizer state not restored ({exc}); using fresh AdamW.", flush=True)
        if scheduler is not None and "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except (ValueError, KeyError) as exc:
                print(f"Scheduler state not restored ({exc}).", flush=True)
        ckpt_bs = ckpt.get("batch_size")
        ckpt_step = int(ckpt.get("global_step", 0))
        ckpt_epoch = int(ckpt.get("epoch", 0) or 0)
        legacy_ckpt = ckpt_bs is None
        # Default: continue after the checkpoint's epoch when it looks trustworthy.
        if legacy_ckpt:
            # Pre-batch_size checkpoints (B=1 AV chunks) — don't trust step/epoch.
            start_epoch = 1
            start_step = 0
            print(
                "Legacy checkpoint (no batch_size): restoring weights only; "
                "starting at epoch 1 / batch 0.",
                flush=True,
            )
        else:
            start_epoch = max(1, ckpt_epoch + 1) if ckpt_epoch > 0 else 1
            start_step = 0
            if max_steps is not None:
                if int(ckpt_bs) != int(inferred_bs):
                    print(
                        f"Checkpoint batch_size={ckpt_bs} != current {inferred_bs}; "
                        "starting mid-epoch skip at 0 (weights still loaded).",
                        flush=True,
                    )
                    start_step = 0
                elif ckpt_step >= n_train_batches:
                    print(
                        f"Checkpoint global_step={ckpt_step} >= batches/epoch "
                        f"{n_train_batches}; starting next epoch at batch 0.",
                        flush=True,
                    )
                    start_step = 0
                else:
                    # Still inside the checkpoint epoch — finish it.
                    start_epoch = max(1, ckpt_epoch) if ckpt_epoch > 0 else 1
                    start_step = ckpt_step
            elif ckpt_step > 0:
                print(
                    f"Full-epoch mode: ignoring mid-epoch global_step={ckpt_step} "
                    "(weights/optim restored; training from batch 0).",
                    flush=True,
                )
        print(
            f"Resumed from {last_path} "
            f"(epoch={start_epoch}, start_step={start_step}, "
            f"batches/epoch={n_train_batches}, batch_size={inferred_bs})",
            flush=True,
        )
    elif not resume and last_path.is_file():
        print(f"--fresh: ignoring existing {last_path}", flush=True)

    def _save_last(global_step: int, epoch: int = 0) -> None:
        payload: dict[str, Any] = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "batch_size": inferred_bs,
        }
        if scheduler is not None:
            payload["scheduler"] = scheduler.state_dict()
        torch.save(payload, last_path)
        print(f"  checkpoint_last @ step {global_step} (epoch {epoch})", flush=True)

    metrics_path = run_dir / "metrics.jsonl"
    best_val = float("inf")
    best_path = run_dir / "checkpoint_best.pt"
    patience = int(train_cfg.early_stopping.patience)
    bad = 0
    history: list[dict[str, Any]] = []
    mode = "a" if metrics_path.is_file() and resume else "w"

    if start_epoch > max_epochs:
        print(
            f"Nothing to do: resume epoch {start_epoch} > max_epochs {max_epochs}. "
            "Pass a higher --epochs or --fresh.",
            flush=True,
        )
        tracker.close()
        return {
            "best_val": best_val,
            "epochs": 0,
            "run_dir": str(run_dir),
            "history": history,
        }

    with metrics_path.open(mode, encoding="utf-8", newline="\n") as mf:
        for epoch in range(start_epoch, max_epochs + 1):
            print(
                f"Epoch {epoch}/{max_epochs}  "
                f"train_batches={n_train_batches}  val_batches={len(val_loader)}",
                flush=True,
            )
            t0 = time.perf_counter()
            tr = run_epoch(
                model,
                train_loader,
                train_cfg=train_cfg,
                dataset_cfg=dataset_cfg,
                optimizer=opt,
                device=device,
                train=True,
                resolved_clipped=resolved_clipped,
                max_steps=max_steps,
                start_step=start_step,
                checkpoint_every=checkpoint_every,
                checkpoint_fn=lambda gs, ep=epoch: _save_last(gs, ep),
            )
            # If max_steps chunk finished mid-epoch, save and return for resume
            if max_steps is not None:
                new_step = int(tr.get("_steps_done", start_step + max_steps))
                _save_last(new_step, epoch)
                print(
                    f"Chunk done through step {new_step}/{n_train_batches}. "
                    "Re-run the same command to continue.",
                    flush=True,
                )
                tracker.close()
                return {
                    "best_val": best_val,
                    "epochs": len(history),
                    "run_dir": str(run_dir),
                    "history": history,
                    "global_step": new_step,
                    "chunked": True,
                }

            va = run_epoch(
                model,
                val_loader,
                train_cfg=train_cfg,
                dataset_cfg=dataset_cfg,
                optimizer=None,
                device=device,
                train=False,
                resolved_clipped=resolved_clipped,
            )
            val_loss = va["loss_total"]
            lr_before = float(opt.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step(val_loss)
            lr_after = float(opt.param_groups[0]["lr"])
            if lr_after < lr_before:
                print(
                    f"  LR reduced: {lr_before:.6g} -> {lr_after:.6g} (epoch {epoch})",
                    flush=True,
                )

            row = {
                "epoch": epoch,
                "train": {k: v for k, v in tr.items() if not str(k).startswith("_")},
                "val": {k: v for k, v in va.items() if not str(k).startswith("_")},
                "lr": lr_after,
                "elapsed_s": round(time.perf_counter() - t0, 3),
            }
            history.append(row)
            mf.write(json.dumps(row) + "\n")
            mf.flush()
            flat = {f"train_{k}": v for k, v in row["train"].items()}
            flat.update({f"val_{k}": v for k, v in row["val"].items()})
            flat["lr"] = lr_after
            tracker.log_metrics(flat, step=epoch)
            print(
                f"  train_loss={row['train']['loss_total']:.4f}  "
                f"val_loss={val_loss:.4f}  lr={lr_after:.6g}  "
                f"({row['elapsed_s']}s)",
                flush=True,
            )

            if val_loss < best_val - float(train_cfg.early_stopping.min_delta):
                best_val = val_loss
                bad = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": va}, best_path)
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stop at epoch {epoch} (patience={patience})", flush=True)
                    break
            start_step = 0  # only first resumed epoch may skip mid-loader
            _save_last(0, epoch)

    from src.eval.viz.training import write_v1_v2_report

    report_path = write_v1_v2_report(run_dir, history)
    tracker.log_artifact(report_path)
    tracker.close()
    return {"best_val": best_val, "epochs": len(history), "run_dir": str(run_dir), "history": history}
