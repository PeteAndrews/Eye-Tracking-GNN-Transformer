"""Experiment tracking wrapper — tracker failure never aborts a run."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf

log = logging.getLogger(__name__)


class Tracker:
    """Thin wrapper: mlflow | wandb | none. All methods are best-effort."""

    def __init__(self, backend: str = "none", *, run_dir: Optional[Path] = None) -> None:
        self.backend = (backend or "none").lower()
        self.run_dir = Path(run_dir) if run_dir else None
        self._active = False
        self._warned = False
        self._run = None
        if self.backend == "none":
            return
        try:
            if self.backend == "mlflow":
                import mlflow

                if self.run_dir is not None:
                    root = self.run_dir.parent.resolve()
                    root.mkdir(parents=True, exist_ok=True)
                    # MLflow 3.x: prefer local sqlite over deprecated bare file store
                    db = root / "mlflow.db"
                    mlflow.set_tracking_uri(f"sqlite:///{db.as_posix()}")
                    # Artifacts still land under runs/m6/mlartifacts
                    os.environ.setdefault(
                        "MLFLOW_ARTIFACTS_DESTINATION",
                        str((root / "mlartifacts").resolve()),
                    )
                mlflow.set_experiment("gnn-gaze-phase1")
                self._run = mlflow.start_run()
                self._active = True
            elif self.backend == "wandb":
                import wandb

                self._run = wandb.init(project="gnn-gaze-phase1", dir=str(self.run_dir or "."))
                self._active = True
            else:
                log.warning("Unknown tracker backend %r — using none", backend)
                self.backend = "none"
        except Exception as exc:  # noqa: BLE001
            self._warn(f"tracker init failed ({exc}); continuing without tracker")
            self.backend = "none"
            self._active = False

    def _warn(self, msg: str) -> None:
        if not self._warned:
            log.warning(msg)
            self._warned = True

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        try:
            if self.backend == "mlflow":
                import mlflow

                flat = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v) for k, v in params.items()}
                mlflow.log_params({k: flat[k] for k in list(flat)[:100]})
            elif self.backend == "wandb":
                import wandb

                wandb.config.update(params, allow_val_change=True)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"tracker log_params failed ({exc})")

    def log_metrics(self, metrics: dict[str, float], *, step: int) -> None:
        if not self._active:
            return
        try:
            if self.backend == "mlflow":
                import mlflow

                mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
            elif self.backend == "wandb":
                import wandb

                wandb.log({**metrics, "step": step})
        except Exception as exc:  # noqa: BLE001
            self._warn(f"tracker log_metrics failed ({exc})")

    def set_tags(self, tags: dict[str, str]) -> None:
        if not self._active:
            return
        try:
            if self.backend == "mlflow":
                import mlflow

                mlflow.set_tags({k: str(v) for k, v in tags.items()})
            elif self.backend == "wandb":
                import wandb

                wandb.run.tags = list(tags.values())  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            self._warn(f"tracker set_tags failed ({exc})")

    def log_artifact(self, path: Path) -> None:
        if not self._active:
            return
        try:
            if self.backend == "mlflow":
                import mlflow

                mlflow.log_artifact(str(path))
            elif self.backend == "wandb":
                import wandb

                wandb.save(str(path))
        except Exception as exc:  # noqa: BLE001
            self._warn(f"tracker log_artifact failed ({exc})")

    def close(self) -> None:
        if not self._active:
            return
        try:
            if self.backend == "mlflow":
                import mlflow

                mlflow.end_run()
            elif self.backend == "wandb":
                import wandb

                wandb.finish()
        except Exception as exc:  # noqa: BLE001
            self._warn(f"tracker close failed ({exc})")
        self._active = False


def flatten_config(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten OmegaConf / dict for tracker params."""
    if OmegaConf.is_config(cfg):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    out: dict[str, Any] = {}
    if not isinstance(cfg, dict):
        return {prefix or "value": cfg}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_config(v, key))
        else:
            out[key] = v
    return out
