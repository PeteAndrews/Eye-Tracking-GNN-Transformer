"""P6 ε derivation for gaze→segment assignment (document px).

ε_px = tan(θ°) × distance_mm × px_per_mm
where distance_mm = median eye-position Z (DACSmm) and px_per_mm = 1 / mm_per_px
from the P1 DACSmm↔px regression stored in epsilon_inputs.parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils import io as uio


def epsilon_from_row(
    eye_z_median_mm: float,
    mm_per_px_x: float,
    mm_per_px_y: float,
    *,
    visual_angle_deg: float = 0.5,
) -> float:
    mm_per_px = 0.5 * (float(mm_per_px_x) + float(mm_per_px_y))
    if mm_per_px <= 0 or not np.isfinite(mm_per_px):
        raise ValueError(f"invalid mm_per_px={mm_per_px}")
    px_per_mm = 1.0 / mm_per_px
    return float(np.tan(np.deg2rad(visual_angle_deg)) * float(eye_z_median_mm) * px_per_mm)


def derive_epsilon(
    repo_root: Path,
    *,
    write_config_comment: bool = True,
) -> dict[str, Any]:
    """Derive corpus ε from epsilon_inputs; write value into preprocessing.yaml."""
    data_cfg = OmegaConf.load(repo_root / "configs" / "data.yaml")
    pre_path = repo_root / "configs" / "preprocessing.yaml"
    pre_cfg = OmegaConf.load(pre_path)
    data_version = str(data_cfg.data_version)
    eps_path = (
        repo_root
        / str(data_cfg.paths.processed_root)
        / data_version
        / "gaze_pruned"
        / "epsilon_inputs.parquet"
    )
    df = pd.read_parquet(eps_path)
    angle = float(pre_cfg.gaze_assignment.epsilon_visual_angle_deg)
    rows = []
    for r in df.itertuples(index=False):
        try:
            e = epsilon_from_row(
                float(r.eye_z_median_mm),
                float(r.mm_per_px_x),
                float(r.mm_per_px_y),
                visual_angle_deg=angle,
            )
            ok = True
            err = None
        except Exception as ex:  # noqa: BLE001
            e = float("nan")
            ok = False
            err = str(ex)
        rows.append(
            {
                "participant_id": str(r.participant_id),
                "eye_z_median_mm": float(r.eye_z_median_mm),
                "mm_per_px_x": float(r.mm_per_px_x),
                "mm_per_px_y": float(r.mm_per_px_y),
                "epsilon_doc_px": e,
                "ok": ok,
                "error": err,
            }
        )
    out = pd.DataFrame(rows)
    valid = out.loc[out["ok"], "epsilon_doc_px"]
    if len(valid) == 0:
        raise RuntimeError("ε derivation failed for all participants")

    # Plausibility: ε should be a few–tens of document px (not <1 or >200)
    median_eps = float(valid.median())
    mean_eps = float(valid.mean())
    used_fallback = False
    fallback_reason = None
    if not (1.0 <= median_eps <= 200.0) or not np.isfinite(median_eps):
        used_fallback = True
        fallback_reason = (
            f"derived median ε={median_eps} outside [1,200] px; "
            f"using epsilon_fallback_doc_px"
        )
        median_eps = float(pre_cfg.gaze_assignment.epsilon_fallback_doc_px)

    # Persist numeric ε into config without rewriting the whole YAML (keeps comments).
    if write_config_comment:
        text = pre_path.read_text(encoding="utf-8")
        old = "epsilon_doc_px: null  # filled after P6 derivation"
        new = (
            f"epsilon_doc_px: {float(round(median_eps, 3))}  "
            f"# corpus median from P1 epsilon_inputs; "
            f"tan({angle}°)×Z_mm×px_per_mm"
        )
        if old in text:
            pre_path.write_text(text.replace(old, new, 1), encoding="utf-8")
        else:
            # Already filled or format drifted — still update via OmegaConf key only
            import re

            text2, n = re.subn(
                r"epsilon_doc_px:\s*[^\n#]+",
                f"epsilon_doc_px: {float(round(median_eps, 3))}  ",
                text,
                count=1,
            )
            if n:
                pre_path.write_text(text2, encoding="utf-8")
    else:
        pre_cfg.gaze_assignment.epsilon_doc_px = float(round(median_eps, 3))

    out_dir = (
        repo_root
        / str(data_cfg.paths.processed_root)
        / data_version
        / "fixations"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "epsilon_per_participant.parquet", index=False)
    summary = {
        "visual_angle_deg": angle,
        "epsilon_doc_px": float(round(median_eps, 3)),
        "median_derived": float(valid.median()),
        "mean_derived": mean_eps,
        "std_derived": float(valid.std(ddof=0)),
        "min_derived": float(valid.min()),
        "max_derived": float(valid.max()),
        "n_participants": int(len(out)),
        "used_fallback": used_fallback,
        "fallback_reason": fallback_reason,
        "formula": "tan(angle_deg) * eye_z_median_mm * (1 / mean(mm_per_px_x,y))",
    }
    uio.write_json(out_dir / "epsilon_derivation.json", summary)
    return summary
