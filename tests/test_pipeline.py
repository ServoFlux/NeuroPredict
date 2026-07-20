from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.config import CLASS_NAMES, PreprocessConfig, TrainConfig
from wmd.dataset import ManifestDataset
from wmd.explain import grad_cam, overlay_cam_on_slice
from wmd.inference import WMDPredictor
from wmd.model import build_model
from wmd.preprocessing import median_filter_3d, preprocess_volume
from wmd.synthetic import generate_dataset, make_volume
from wmd.train import train

def test_preprocess_output_shape() -> None:
    volume = make_volume(label=1, shape=(40, 50, 45))
    config = PreprocessConfig(target_shape=(32, 32, 32))
    tensor = preprocess_volume(volume, config)
    assert tensor.shape == (1, 32, 32, 32)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0

def test_median_filter_removes_salt_and_pepper() -> None:
    rng = np.random.default_rng(0)
    clean = np.tile(np.linspace(0.0, 1.0, 16), (16, 16, 1)).astype(np.float32)
    noisy = clean.copy()
    mask = rng.random(clean.shape)
    noisy[mask < 0.05] = 0.0
    noisy[mask > 0.95] = 1.0
    denoised = median_filter_3d(noisy, size=3)
    assert denoised.shape == clean.shape
    assert np.abs(denoised - clean).mean() < np.abs(noisy - clean).mean()

def test_median_filter_disabled_via_config() -> None:
    volume = make_volume(label=1, shape=(32, 32, 32))
    plain = preprocess_volume(volume, PreprocessConfig(target_shape=(24, 24, 24)))
    denoised = preprocess_volume(
        volume, PreprocessConfig(target_shape=(24, 24, 24), denoise_median_size=3)
    )
    assert plain.shape == denoised.shape == (1, 24, 24, 24)
    assert not torch.equal(plain, denoised)

def test_model_forward() -> None:
    model = build_model(num_classes=len(CLASS_NAMES))
    x = torch.randn(2, 1, 32, 32, 32)
    out = model(x)
    assert out.shape == (2, len(CLASS_NAMES))

def test_dataset_loads(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path, n_per_class=3, shape=(32, 32, 32))
    ds = ManifestDataset(manifest, preprocess=PreprocessConfig(target_shape=(32, 32, 32)))
    assert len(ds) == 6
    vol, label = ds[0]
    assert vol.shape == (1, 32, 32, 32)
    assert label in (0, 1)

def test_train_and_predict(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path / "data", n_per_class=8, shape=(32, 32, 32))
    model_path = tmp_path / "model.pt"
    config = TrainConfig(
        epochs=3,
        batch_size=4,
        preprocess=PreprocessConfig(target_shape=(32, 32, 32)),
    )
    metrics = train(manifest, config=config, model_path=model_path)
    assert "accuracy" in metrics

    predictor = WMDPredictor(model_path)
    healthy = make_volume(label=0, shape=(40, 40, 40), rng=np.random.default_rng(0))
    pred = predictor.predict_volume(healthy)
    assert pred.label in CLASS_NAMES
    assert 0.0 <= pred.confidence <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4

def test_grad_cam_shape_and_range() -> None:
    model = build_model(num_classes=len(CLASS_NAMES))
    x = torch.randn(1, 1, 32, 32, 32, requires_grad=True)
    cam = grad_cam(model, x, class_idx=1)
    assert cam.shape == (32, 32, 32)
    assert float(cam.min()) >= 0.0
    assert float(cam.max()) <= 1.0

    overlay = overlay_cam_on_slice(np.zeros((32, 32)), cam[16])
    assert overlay.shape == (32, 32, 3)
    assert overlay.dtype == np.uint8
