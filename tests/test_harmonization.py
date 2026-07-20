import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmd.config import PreprocessConfig
from wmd.harmonization import (
    bias_field_correct,
    otsu_brain_mask,
    white_stripe_normalize,
    zscore_normalize,
)
from wmd.preprocessing import preprocess_volume
from wmd.synthetic import make_volume

def _shaded_volume(shape=(32, 32, 32)) -> np.ndarray:
    vol = make_volume(label=1, shape=shape).astype(np.float32)
    axis = np.linspace(0.5, 1.5, shape[2], dtype=np.float32)
    gradient = np.broadcast_to(axis, shape)
    return vol * gradient

def test_otsu_mask_separates_brain_from_background() -> None:
    vol = make_volume(label=0, shape=(32, 32, 32))
    mask = otsu_brain_mask(vol)
    assert mask.shape == vol.shape
    assert mask.dtype == bool
    frac = float(mask.mean())
    assert 0.05 < frac < 0.95
    assert vol[mask].mean() > vol[~mask].mean()

def test_bias_correction_reduces_shading() -> None:
    shaded = _shaded_volume()
    mask = otsu_brain_mask(shaded)
    corrected = bias_field_correct(shaded, mask=mask)
    assert corrected.shape == shaded.shape
    assert corrected.dtype == np.float32

    def _halves_gap(v: np.ndarray) -> float:
        w = v.shape[2]
        left = v[:, :, : w // 2][mask[:, :, : w // 2]]
        right = v[:, :, w // 2 :][mask[:, :, w // 2 :]]
        return abs(float(left.mean()) - float(right.mean()))

    assert _halves_gap(corrected) < _halves_gap(shaded)

def test_zscore_normalize_stats() -> None:
    vol = make_volume(label=1, shape=(32, 32, 32))
    mask = otsu_brain_mask(vol)
    out = zscore_normalize(vol, mask)
    assert out.shape == vol.shape
    brain = out[mask]
    assert abs(float(brain.mean())) < 1e-4
    assert abs(float(brain.std()) - 1.0) < 1e-4

def test_whitestripe_is_scanner_scale_invariant() -> None:
    vol = make_volume(label=1, shape=(32, 32, 32)).astype(np.float32)
    a = white_stripe_normalize(vol)
    b = white_stripe_normalize(vol * 2.0 + 5.0)
    assert a.shape == b.shape
    assert np.abs(a - b).mean() < 0.1

def test_preprocess_harmonization_modes_shape_and_default_unchanged() -> None:
    vol = _shaded_volume((28, 28, 28))
    target = (24, 24, 24)

    default = preprocess_volume(vol, PreprocessConfig(target_shape=target))
    assert float(default.min()) >= 0.0 and float(default.max()) <= 1.0

    for mode in ("zscore", "whitestripe"):
        out = preprocess_volume(
            vol,
            PreprocessConfig(target_shape=target, intensity_norm=mode, bias_correct=True),
        )
        assert out.shape == (1, *target)
        assert out.dtype == torch.float32
        assert not torch.equal(default, out)

def test_unknown_intensity_norm_raises() -> None:
    vol = make_volume(label=0, shape=(24, 24, 24))
    try:
        preprocess_volume(vol, PreprocessConfig(intensity_norm="bogus"))
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown intensity_norm")
