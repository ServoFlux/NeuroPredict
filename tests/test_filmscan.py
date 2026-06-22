"""Tests for the Archive MRI digitizer film-sheet <-> volume bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.filmscan import (  # noqa: E402
    contact_sheet_from_volume,
    grid_shape_for_depth,
    volume_from_contact_sheet,
)
from wmd.synthetic import make_etiology_volume  # noqa: E402


def test_grid_shape_for_depth() -> None:
    assert grid_shape_for_depth(64, 8) == (8, 8)
    assert grid_shape_for_depth(50, 8) == (7, 8)  # rounds up to hold all slices
    assert grid_shape_for_depth(1, 8) == (1, 8)


def test_contact_sheet_dimensions() -> None:
    volume = make_etiology_volume(0, shape=(16, 32, 32), rng=np.random.default_rng(0))
    sheet = contact_sheet_from_volume(volume, cols=4, cell=64)
    rows, cols = grid_shape_for_depth(16, 4)
    assert sheet.shape == (rows * 64, cols * 64)
    assert sheet.dtype == np.uint8


def test_roundtrip_preserves_depth_and_structure() -> None:
    rng = np.random.default_rng(1)
    volume = make_etiology_volume(1, shape=(32, 48, 48), rng=rng)
    sheet = contact_sheet_from_volume(volume, cols=8, cell=48)
    rows, cols = grid_shape_for_depth(32, 8)

    recon = volume_from_contact_sheet(sheet, rows=rows, cols=cols, depth=32)
    assert recon.shape[0] == 32  # all slices recovered, no blank trailing tiles

    # The reconstructed slices should track the original brightness profile:
    # bright (brain) slices stay brighter than near-empty edge slices.
    orig_profile = volume.mean(axis=(1, 2))
    recon_profile = recon.mean(axis=(1, 2))
    corr = np.corrcoef(orig_profile, recon_profile)[0, 1]
    assert corr > 0.8


def test_auto_crop_handles_dark_border() -> None:
    rng = np.random.default_rng(2)
    volume = make_etiology_volume(0, shape=(16, 32, 32), rng=rng)
    sheet = contact_sheet_from_volume(volume, cols=4, cell=40)
    # Pad a black border around the "photo" as a real camera shot would have.
    padded = np.zeros((sheet.shape[0] + 30, sheet.shape[1] + 30), dtype=np.uint8)
    padded[15 : 15 + sheet.shape[0], 15 : 15 + sheet.shape[1]] = sheet
    rows, cols = grid_shape_for_depth(16, 4)

    recon = volume_from_contact_sheet(padded, rows=rows, cols=cols, depth=16)
    assert recon.shape[0] == 16
    assert recon.mean() > 0  # the border was trimmed, brain content recovered
