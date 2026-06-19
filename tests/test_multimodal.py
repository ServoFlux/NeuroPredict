"""Tests for the multimodal (MRI + clinical) pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.clinical import (  # noqa: E402
    CLINICAL_FIELD_NAMES,
    NUM_CLINICAL_FEATURES,
    encode_clinical,
    make_clinical,
)
from wmd.config import (  # noqa: E402
    CLASS_NAMES,
    ETIOLOGY_CLASS_NAMES,
    PreprocessConfig,
    TrainConfig,
)
from wmd.dataset import MultimodalManifestDataset  # noqa: E402
from wmd.inference import MultimodalWMDPredictor  # noqa: E402
from wmd.model import build_multimodal_model  # noqa: E402
from wmd.synthetic import (  # noqa: E402
    generate_dataset,
    make_etiology_volume,
    make_volume,
)
from wmd.train import train_multimodal  # noqa: E402


def test_encode_clinical_shape_and_age_scaling() -> None:
    answers = {"age": 80, "hypertension": 1, "diabetes": 0}
    vec = encode_clinical(answers)
    assert vec.shape == (NUM_CLINICAL_FEATURES,)
    assert vec.dtype == np.float32
    assert abs(vec[0] - 0.8) < 1e-6  # age normalized by 100
    assert set(np.unique(vec[1:])) <= {0.0, 1.0}


def test_make_clinical_correlates_with_label() -> None:
    rng = np.random.default_rng(0)
    n = 60
    diseased = [make_clinical(1, rng) for _ in range(n)]
    healthy = [make_clinical(0, rng) for _ in range(n)]
    # Diseased subjects should report memory problems more often on average.
    d_mem = np.mean([a["memory_problems"] for a in diseased])
    h_mem = np.mean([a["memory_problems"] for a in healthy])
    assert d_mem > h_mem


def test_multimodal_model_forward() -> None:
    model = build_multimodal_model(NUM_CLINICAL_FEATURES, num_classes=len(CLASS_NAMES))
    vol = torch.randn(2, 1, 32, 32, 32)
    clin = torch.randn(2, NUM_CLINICAL_FEATURES)
    out = model(vol, clin)
    assert out.shape == (2, len(CLASS_NAMES))


def test_multimodal_dataset(tmp_path: Path) -> None:
    manifest = generate_dataset(
        tmp_path, n_per_class=3, shape=(32, 32, 32), with_clinical=True
    )
    ds = MultimodalManifestDataset(
        manifest, preprocess=PreprocessConfig(target_shape=(32, 32, 32))
    )
    assert len(ds) == 6
    vol, clin, label = ds[0]
    assert vol.shape == (1, 32, 32, 32)
    assert clin.shape == (NUM_CLINICAL_FEATURES,)
    assert label in (0, 1)


def test_train_predict_and_attribution(tmp_path: Path) -> None:
    manifest = generate_dataset(
        tmp_path / "data", n_per_class=8, shape=(32, 32, 32), with_clinical=True
    )
    model_path = tmp_path / "mm.pt"
    config = TrainConfig(
        epochs=3,
        batch_size=4,
        preprocess=PreprocessConfig(target_shape=(32, 32, 32)),
    )
    metrics = train_multimodal(manifest, config=config, model_path=model_path)
    assert "accuracy" in metrics

    predictor = MultimodalWMDPredictor(model_path)
    assert predictor.clinical_fields == list(CLINICAL_FIELD_NAMES)

    volume = make_volume(label=1, shape=(40, 40, 40), rng=np.random.default_rng(0))
    answers = make_clinical(1, np.random.default_rng(1))
    pred, attr = predictor.predict(volume, answers)
    assert pred.label in CLASS_NAMES
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4
    assert 0.0 <= attr.combined <= 1.0
    assert 0.0 <= attr.baseline <= 1.0
    assert abs(attr.image_share + attr.clinical_share - 1.0) < 1e-6


def test_make_clinical_etiology_profiles() -> None:
    rng = np.random.default_rng(0)
    n = 80
    genetic = [make_clinical(ETIOLOGY_CLASS_NAMES.index("genetic"), rng) for _ in range(n)]
    vascular = [make_clinical(ETIOLOGY_CLASS_NAMES.index("vascular"), rng) for _ in range(n)]
    # Genetic profile carries NOTCH3 variants far more than vascular does.
    assert np.mean([a["notch3_variant"] for a in genetic]) > np.mean(
        [a["notch3_variant"] for a in vascular]
    )
    # Vascular profile carries hypertension far more than genetic does.
    assert np.mean([a["hypertension"] for a in vascular]) > np.mean(
        [a["hypertension"] for a in genetic]
    )


def test_multiclass_dataset_and_prediction(tmp_path: Path) -> None:
    manifest = generate_dataset(
        tmp_path / "data",
        n_per_class=6,
        shape=(32, 32, 32),
        with_clinical=True,
        multiclass=True,
    )
    ds = MultimodalManifestDataset(
        manifest, preprocess=PreprocessConfig(target_shape=(32, 32, 32))
    )
    assert ds.target_column == "etiology"
    assert len(ds) == 6 * len(ETIOLOGY_CLASS_NAMES)
    assert set(ds.labels()) <= set(range(len(ETIOLOGY_CLASS_NAMES)))

    model_path = tmp_path / "mm.pt"
    config = TrainConfig(
        epochs=2, batch_size=4,
        preprocess=PreprocessConfig(target_shape=(32, 32, 32)),
    )
    train_multimodal(manifest, config=config, model_path=model_path)
    predictor = MultimodalWMDPredictor(model_path)
    assert predictor.class_names == list(ETIOLOGY_CLASS_NAMES)

    volume = make_etiology_volume(
        ETIOLOGY_CLASS_NAMES.index("vascular"), shape=(40, 40, 40),
        rng=np.random.default_rng(0),
    )
    answers = make_clinical(ETIOLOGY_CLASS_NAMES.index("vascular"), np.random.default_rng(1))
    pred, attr = predictor.predict(volume, answers)
    assert pred.label in ETIOLOGY_CLASS_NAMES
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4
    assert 0.0 <= attr.combined <= 1.0


def test_explain_writes_images(tmp_path: Path) -> None:
    manifest = generate_dataset(
        tmp_path / "data", n_per_class=8, shape=(32, 32, 32), with_clinical=True
    )
    model_path = tmp_path / "mm.pt"
    config = TrainConfig(
        epochs=2,
        batch_size=4,
        preprocess=PreprocessConfig(target_shape=(32, 32, 32)),
    )
    train_multimodal(manifest, config=config, model_path=model_path)
    predictor = MultimodalWMDPredictor(model_path)

    import nibabel as nib

    vol = make_volume(label=1, shape=(40, 40, 40))
    scan_path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), str(scan_path))
    answers = make_clinical(1)
    pred, _ = predictor.predict_path(scan_path, answers)

    overlay = tmp_path / "cam.png"
    inp = tmp_path / "in.png"
    exp = predictor.explain_path(scan_path, answers, pred, overlay, inp)
    assert overlay.exists() and inp.exists()
    assert exp.slice_index >= 0
