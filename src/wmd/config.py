"""Central configuration for the WMD detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project layout
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MODEL_PATH = MODELS_DIR / "wmd_cnn.pt"

# Class labels (index order matters: it maps to model output logits)
CLASS_NAMES: tuple[str, ...] = ("no_wmd", "early_wmd")


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for turning a raw scan into a model-ready tensor."""

    # Volume is resampled to this (depth, height, width) before going to the CNN.
    target_shape: tuple[int, int, int] = (64, 64, 64)
    # Intensity normalization clip percentiles. The upper bound is deliberately
    # high (99.9) so that bright white-matter hyperintensities -- the signal of
    # interest -- are preserved rather than clipped away.
    clip_percentiles: tuple[float, float] = (0.5, 99.9)


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training the 3D CNN."""

    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    seed: int = 42
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


# Disclaimer surfaced everywhere a prediction is shown.
RESEARCH_DISCLAIMER = (
    "This tool is for research and educational purposes only. It is NOT a "
    "medical device and must NOT be used for diagnosis or clinical "
    "decision-making. Always consult a qualified clinician."
)
