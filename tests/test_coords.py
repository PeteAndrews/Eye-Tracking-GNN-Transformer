"""Unit tests for P5 coordinate finalisation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.coords import add_coordinate_columns, docnorm_xy, viewport_features

ROOT = Path(__file__).resolve().parents[1]


def test_docnorm_hand_computed() -> None:
    x = np.array([0.0, 960.0, 1920.0])
    y = np.array([0.0, 540.0, 1080.0])
    xn, yn = docnorm_xy(x, y, w_doc=1920, h_doc=1080, mode="docnorm")
    np.testing.assert_allclose(xn, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(yn, [0.0, 0.5, 1.0])


def test_isotropic_uses_width_for_both_axes() -> None:
    x = np.array([960.0])
    y = np.array([960.0])
    xn, yn = docnorm_xy(x, y, w_doc=1920, h_doc=2344, mode="isotropic")
    np.testing.assert_allclose(xn, [0.5])
    np.testing.assert_allclose(yn, [0.5])  # y/W, not y/H


def test_viewport_hand_computed() -> None:
    y_doc = np.array([200.0, 800.0])
    scroll = np.array([0.0, 100.0])
    y_screen, vpos, gvy = viewport_features(
        y_doc, scroll, h_doc=1280, h_screen=1080
    )
    np.testing.assert_allclose(y_screen, [200.0, 700.0])
    np.testing.assert_allclose(vpos, [0.0, 100.0 / 200.0])
    np.testing.assert_allclose(gvy, [200.0 / 1080.0, 700.0 / 1080.0])


def test_viewport_zero_when_no_scroll_range() -> None:
    y_doc = np.array([100.0])
    scroll = np.array([0.0])
    _, vpos, _ = viewport_features(y_doc, scroll, h_doc=1080, h_screen=1080)
    np.testing.assert_allclose(vpos, [0.0])


def test_add_coordinate_columns_preserves_raw() -> None:
    df = pd.DataFrame(
        {
            "gaze_point_x_doc": [192.0, 384.0],
            "gaze_point_y_doc": [108.0, 216.0],
            "scroll_offset_y": [0.0, 50.0],
        }
    )
    out = add_coordinate_columns(df, w_doc=1920, h_doc=1080, h_screen=1080, mode="docnorm")
    assert list(out["gaze_point_x_doc"]) == [192.0, 384.0]
    np.testing.assert_allclose(out["x_docnorm"], [0.1, 0.2])
    np.testing.assert_allclose(out["y_docnorm"], [0.1, 0.2])
    assert out["normalisation_mode"].iloc[0] == "docnorm"
    assert out["h_screen"].iloc[0] == 1080


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown normalisation"):
        docnorm_xy(np.array([1.0]), np.array([1.0]), w_doc=10, h_doc=10, mode="bogus")


@pytest.mark.skipif(
    not (ROOT / "data_processed" / "v0_p0" / "gaze_coords" / "p01.parquet").is_file(),
    reason="P5 outputs not built yet",
)
def test_real_episode_has_coord_columns() -> None:
    path = ROOT / "data_processed" / "v0_p0" / "gaze_coords" / "p01.parquet"
    df = pd.read_parquet(path)
    ep = df[df["trial_id"].astype(str) == "T01"]
    assert len(ep) > 0
    for col in (
        "x_docnorm",
        "y_docnorm",
        "y_screen",
        "viewport_doc_position",
        "gaze_viewport_y",
        "w_doc",
        "h_doc",
    ):
        assert col in ep.columns
    # Coord features may be NaN only where raw doc gaze is missing
    valid = ep["gaze_point_x_doc"].notna() & ep["gaze_point_y_doc"].notna()
    assert valid.any()
    assert bool(ep.loc[valid, "x_docnorm"].notna().all())
    assert bool(ep.loc[valid, "y_docnorm"].notna().all())
    assert bool(ep.loc[valid, "y_screen"].notna().all())
