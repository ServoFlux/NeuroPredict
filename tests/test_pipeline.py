"""Fast smoke tests for the WMD pipeline (no GPU, no real data needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.config import CLASS_NAMES, PreprocessConfig, TrainConfig  # noqa: E402
from wmd.dataset import ManifestDataset  # noqa: E402
from wmd.explain import grad_cam, overlay_cam_on_slice  # noqa: E402
from wmd.inference import WMDPredictor  # noqa: E402
from wmd.model import build_model  # noqa: E402
from wmd.preprocessing import preprocess_volume  # noqa: E402
from wmd.synthetic import generate_dataset, make_volume  # noqa: E402
from wmd.train import train  # noqa: E402


def test_preprocess_output_shape() -> None:
    volume = make_volume(label=1, shape=(40, 50, 45))
    config = PreprocessConfig(target_shape=(32, 32, 32))
    tensor = preprocess_volume(volume, config)
    assert tensor.shape == (1, 32, 32, 32)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= 0.0


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
