from __future__ import annotations
import sys
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from wmd.config import CLASS_NAMES, PreprocessConfig
from wmd.dataset import ManifestDataset
from wmd.model import build_model
from wmd.synthetic import generate_dataset
from train_real import AugmentedDataset, pretrain_on_synthetic
def test_strong_aug_changes_volume_and_keeps_shape(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path, n_per_class=2, shape=(24, 24, 24))
    base = ManifestDataset(manifest, preprocess=PreprocessConfig(target_shape=(24, 24, 24)))
    aug = AugmentedDataset(base, seed=0, strong=True)
    vol, label = aug[0]
    assert vol.shape == (1, 24, 24, 24)
    assert label in (0, 1)
    assert float(vol.min()) >= 0.0
    assert float(vol.max()) <= 1.0
def test_pretrain_returns_loadable_state_dict() -> None:
    state = pretrain_on_synthetic(target_shape=(24, 24, 24), n_per_class=2, epochs=1, seed=0)
    model = build_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(state)
    x = torch.randn(1, 1, 24, 24, 24)
    assert model(x).shape == (1, len(CLASS_NAMES))
