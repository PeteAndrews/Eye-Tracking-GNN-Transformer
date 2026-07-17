"""P5 — coordinate finalisation (DOCnorm + viewport features).

Working space for assignment/geometry remains raw document pixels.
DOCnorm and viewport columns are additive features for P6+ only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.aoi_injection import write_gaze_table
from src.utils import io as uio


def docnorm_xy(
    x_doc: np.ndarray,
    y_doc: np.ndarray,
    *,
    w_doc: float,
    h_doc: float,
    mode: str = "docnorm",
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise document-space coordinates.

    docnorm:   x/W, y/H
    isotropic: x/W, y/W  (both axes by width)
    """
    w = float(w_doc)
    h = float(h_doc)
    if w <= 0:
        raise ValueError(f"W_doc must be positive, got {w_doc}")
    if mode == "docnorm":
        if h <= 0:
            raise ValueError(f"H_doc must be positive, got {h_doc}")
        return x_doc / w, y_doc / h
    if mode == "isotropic":
        return x_doc / w, y_doc / w
    raise ValueError(f"Unknown normalisation mode: {mode!r}")


def viewport_features(
    y_doc: np.ndarray,
    scroll_offset_y: np.ndarray,
    *,
    h_doc: float,
    h_screen: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_screen, viewport_doc_position, gaze_viewport_y).

    y_screen = y_doc - scroll_offset_y
    viewport_doc_position = scroll_offset_y / (H_doc - H_screen)  (0 if no scroll range)
    gaze_viewport_y = y_screen / H_screen  (screen-normalised gaze-in-viewport y)
    """
    y_screen = y_doc - scroll_offset_y
    scrollable = float(h_doc) - float(h_screen)
    if scrollable > 0:
        viewport_doc_position = scroll_offset_y / scrollable
    else:
        viewport_doc_position = np.zeros_like(scroll_offset_y, dtype=float)
    hs = float(h_screen)
    if hs <= 0:
        raise ValueError(f"H_screen must be positive, got {h_screen}")
    gaze_viewport_y = y_screen / hs
    return y_screen, viewport_doc_position, gaze_viewport_y


def add_coordinate_columns(
    df: pd.DataFrame,
    *,
    w_doc: float,
    h_doc: float,
    h_screen: float,
    mode: str = "docnorm",
) -> pd.DataFrame:
    """Add DOCnorm + viewport columns; preserve raw doc coords."""
    out = df.copy()
    x = out["gaze_point_x_doc"].to_numpy(dtype=float)
    y = out["gaze_point_y_doc"].to_numpy(dtype=float)
    scroll = out["scroll_offset_y"].to_numpy(dtype=float)
    xn, yn = docnorm_xy(x, y, w_doc=w_doc, h_doc=h_doc, mode=mode)
    y_screen, vpos, gvy = viewport_features(y, scroll, h_doc=h_doc, h_screen=h_screen)
    out["w_doc"] = float(w_doc)
    out["h_doc"] = float(h_doc)
    out["h_screen"] = float(h_screen)
    out["normalisation_mode"] = mode
    out["x_docnorm"] = xn
    out["y_docnorm"] = yn
    out["y_screen"] = y_screen
    out["viewport_doc_position"] = vpos
    out["gaze_viewport_y"] = gvy
    return out


def _dims_lookup(dim_df: pd.DataFrame) -> dict[tuple[str, str], tuple[int, int]]:
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for r in dim_df.itertuples(index=False):
        out[(str(r.trial_id), str(r.star_condition))] = (int(r.W_doc), int(r.H_doc))
    return out


def run_p5(repo_root: Optional[Path] = None) -> dict[str, Any]:
    """Enrich gaze_canonical → gaze_coords with DOCnorm + viewport columns."""
    repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    data_cfg = OmegaConf.load(repo_root / "configs" / "data.yaml")
    pre_cfg = OmegaConf.load(repo_root / "configs" / "preprocessing.yaml")
    data_version = str(data_cfg.data_version)
    mode = str(pre_cfg.normalisation)
    h_screen = float(pre_cfg.viewport.H_screen_px)

    processed = repo_root / str(data_cfg.paths.processed_root) / data_version
    canon_dir = processed / "gaze_canonical"
    out_dir = processed / "gaze_coords"
    out_dir.mkdir(parents=True, exist_ok=True)

    dim_df = pd.read_parquet(processed / "registry" / "document_dimensions.parquet")
    dims = _dims_lookup(dim_df)
    star_tbl = pd.read_parquet(processed / "registry" / "star_conditions.parquet")
    star_map = {
        (str(r.participant_id), str(r.trial_id)): str(r.star_condition)
        for r in star_tbl.itertuples(index=False)
    }

    errors: list[str] = []
    n_participants = 0
    n_rows = 0
    modes_seen: set[str] = set()

    for path in sorted(canon_dir.glob("p*.parquet")):
        num = path.stem[1:] if path.stem.lower().startswith("p") else path.stem
        pid_key = f"P{int(num):02d}" if str(num).isdigit() else path.stem.upper()
        df = pd.read_parquet(path)
        parts = []
        for trial_id, ep in df.groupby("trial_id", sort=True):
            tid = str(trial_id)
            sc = star_map.get((pid_key, tid))
            if sc is None:
                errors.append(f"missing star_condition for {pid_key}/{tid}")
                continue
            key = (tid, sc)
            if key not in dims:
                errors.append(f"missing dimensions for {tid}/{sc}")
                continue
            w_doc, h_doc = dims[key]
            parts.append(
                add_coordinate_columns(
                    ep, w_doc=w_doc, h_doc=h_doc, h_screen=h_screen, mode=mode
                )
            )
        if not parts:
            errors.append(f"no episodes written for {path.name}")
            continue
        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values(
            ["participant_id", "trial_id", "recording_timestamp"]
        ).reset_index(drop=True)
        write_gaze_table(out_dir / path.stem, out)
        n_participants += 1
        n_rows += len(out)
        modes_seen.add(mode)

    # Spot-check columns on first file
    sample_cols: list[str] = []
    sample_path = out_dir / "p01.parquet"
    if sample_path.is_file():
        sample_cols = list(pd.read_parquet(sample_path, columns=None).columns)

    required = {
        "gaze_point_x_doc",
        "gaze_point_y_doc",
        "x_docnorm",
        "y_docnorm",
        "y_screen",
        "viewport_doc_position",
        "gaze_viewport_y",
        "w_doc",
        "h_doc",
        "h_screen",
        "normalisation_mode",
    }
    summary = {
        "data_version": data_version,
        "out_dir": str(out_dir),
        "n_participants": n_participants,
        "n_rows": n_rows,
        "normalisation_mode": mode,
        "h_screen_px": h_screen,
        "formats": ["parquet", "tsv"],
        "required_columns_present": required.issubset(set(sample_cols)),
        "sample_columns": sample_cols,
        "errors": errors,
        "ok": n_participants == int(data_cfg.expected.n_participants)
        and not errors
        and required.issubset(set(sample_cols)),
        "note": (
            "Raw gaze_point_*_doc remain the working space for assignment. "
            "DOCnorm/viewport columns feed P6 features only."
        ),
    }
    uio.write_json(out_dir / "p5_summary.json", summary)
    return summary
