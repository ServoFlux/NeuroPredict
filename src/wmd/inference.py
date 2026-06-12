"""Inference: load a trained checkpoint and predict on a single scan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_MODEL_PATH, PreprocessConfig
from .model import build_model
from .preprocessing import load_volume, preprocess_volume


@dataclass
class Prediction:
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]


class WMDPredictor:
    """Loads a checkpoint once and serves predictions for uploaded scans."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.model_path}. "
                "Train one first (see scripts/train_demo.py)."
            )
        # weights_only=False: our checkpoint stores config metadata, not just tensors.
        checkpoint = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        self.class_names: list[str] = checkpoint["class_names"]
        self.preprocess = PreprocessConfig(
            target_shape=tuple(checkpoint["target_shape"]),
            clip_percentiles=tuple(checkpoint["clip_percentiles"]),
        )
        self.val_metrics: dict[str, float] = checkpoint.get("val_metrics", {})
        self.model = build_model(num_classes=checkpoint["num_classes"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_volume(self, volume: np.ndarray) -> Prediction:
        tensor = preprocess_volume(volume, self.preprocess)[None]  # (1, 1, D, H, W)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        idx = int(probs.argmax())
        return Prediction(
            label=self.class_names[idx],
            label_index=idx,
            confidence=float(probs[idx]),
            probabilities={
                name: float(p) for name, p in zip(self.class_names, probs)
            },
        )

    def predict_path(self, path: str | Path) -> Prediction:
        return self.predict_volume(load_volume(path))


def save_preview(path: str | Path, out_png: str | Path) -> Path:
    """Save a mid-axial-slice PNG preview of a scan for display in the UI."""
    from PIL import Image

    volume = load_volume(path)
    mid = volume.shape[0] // 2
    slice2d = volume[mid]
    lo, hi = np.percentile(slice2d, (1, 99))
    if hi <= lo:
        hi, lo = float(slice2d.max()), float(slice2d.min())
    norm = np.clip((slice2d - lo) / (hi - lo + 1e-8), 0, 1)
    img = Image.fromarray((norm * 255).astype(np.uint8))
    img = img.resize((256, 256))
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_png))
    return out_png
