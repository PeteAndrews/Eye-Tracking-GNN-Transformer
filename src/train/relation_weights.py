"""Resolve next-relation BCE weights from the frozen M5 frequency table.

``clip_max`` in ``configs/train.yaml`` is the only knob for the weight ceiling;
``resolved_clipped`` is computed at load time (never hand-edited).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.utils import io as uio


def load_raw_inv_freq_weights(source: Path) -> dict[str, Optional[float]]:
    """Map relation name → raw ``inv_freq_weight`` (None if count==0)."""
    report = uio.read_json(source)
    out: dict[str, Optional[float]] = {}
    for row in report["labels"]:
        name = str(row["relation"])
        raw = row.get("inv_freq_weight")
        out[name] = None if raw is None else float(raw)
    return out


def resolve_clipped_weights(
    *,
    source: Path,
    active_labels: Sequence[str],
    clip_max: float,
    raw_weights: Optional[Mapping[str, Optional[float]]] = None,
) -> dict[str, float]:
    """Clip raw inverse-frequency weights at ``clip_max`` for active labels."""
    raw = dict(raw_weights) if raw_weights is not None else load_raw_inv_freq_weights(source)
    ceiling = float(clip_max)
    if ceiling <= 0:
        raise ValueError(f"clip_max must be > 0, got {ceiling}")
    resolved: dict[str, float] = {}
    for name in active_labels:
        if name not in raw:
            raise KeyError(f"Label {name!r} missing from {source}")
        w = raw[name]
        if w is None:
            raise ValueError(
                f"Label {name!r} has null inv_freq_weight (zero count); "
                "exclude it from relation_weights.active_labels"
            )
        resolved[str(name)] = min(float(w), ceiling)
    return resolved


def resolve_clipped_from_train_cfg(
    train_cfg: Any, repo: Path
) -> dict[str, float]:
    """Convenience: read ``relation_weights`` block and resolve against repo root."""
    rw = train_cfg.relation_weights
    source = Path(repo) / str(rw.source)
    return resolve_clipped_weights(
        source=source,
        active_labels=list(rw.active_labels),
        clip_max=float(rw.clip_max),
    )
