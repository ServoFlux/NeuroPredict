from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from torch import nn

from .clinical import encode_clinical
from .config import DEFAULT_MODEL_PATH, DEFAULT_MULTIMODAL_MODEL_PATH, PreprocessConfig
from .explain import grad_cam, most_salient_axial_index, overlay_cam_on_slice
from .model import build_model, build_multimodal_model
from .preprocessing import load_volume, preprocess_volume

@dataclass
class Prediction:
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]

@dataclass
class ModalityAttribution:

    combined: float
    baseline: float
    image_delta: float
    clinical_delta: float
    image_share: float
    clinical_share: float

@dataclass
class Explanation:

    original_shape: tuple[int, int, int]
    processed_shape: tuple[int, int, int]
    slice_index: int
    attention_fraction: float

def _save_cam_slice(
    processed: np.ndarray,
    cam: np.ndarray,
    input_png: str | Path,
    overlay_png: str | Path,
) -> int:
    from PIL import Image

    z = most_salient_axial_index(cam)
    base = np.clip(processed[z], 0.0, 1.0)
    input_rgb = (np.stack([base, base, base], axis=-1) * 255).astype(np.uint8)
    overlay_rgb = overlay_cam_on_slice(base, cam[z])
    for arr, dest in ((input_rgb, input_png), (overlay_rgb, overlay_png)):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).resize((256, 256), Image.NEAREST).save(str(dest))
    return z

class WMDPredictor:

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.model_path}. "
                "Train one first (see scripts/train_demo.py)."
            )
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
        tensor = preprocess_volume(volume, self.preprocess)[None]
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

    def explain_path(
        self,
        path: str | Path,
        prediction: Prediction,
        overlay_png: str | Path,
        input_png: str | Path,
    ) -> Explanation:
        volume = load_volume(path)
        original_shape = tuple(int(d) for d in volume.shape)

        tensor = preprocess_volume(volume, self.preprocess)[None]
        tensor.requires_grad_(True)
        cam = grad_cam(self.model, tensor, prediction.label_index)

        processed = tensor.detach()[0, 0].numpy()
        z = _save_cam_slice(processed, cam, input_png, overlay_png)
        return Explanation(
            original_shape=original_shape,
            processed_shape=tuple(int(d) for d in processed.shape),
            slice_index=z,
            attention_fraction=float((cam[z] > 0.5).mean()),
        )

class _ImageBranchWrapper(nn.Module):

    def __init__(self, mm_model: nn.Module, clinical: torch.Tensor) -> None:
        super().__init__()
        self.mm_model = mm_model
        self.features = mm_model.features
        self._clinical = clinical

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.mm_model(volume, self._clinical)

class MultimodalWMDPredictor:

    def __init__(
        self, model_path: str | Path = DEFAULT_MULTIMODAL_MODEL_PATH
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Multimodal model checkpoint not found at {self.model_path}. "
                "Train one first (see scripts/train_demo.py)."
            )
        checkpoint = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        self.class_names: list[str] = checkpoint["class_names"]
        self.clinical_fields: list[str] = checkpoint["clinical_fields"]
        self.preprocess = PreprocessConfig(
            target_shape=tuple(checkpoint["target_shape"]),
            clip_percentiles=tuple(checkpoint["clip_percentiles"]),
        )
        self.val_metrics: dict[str, float] = checkpoint.get("val_metrics", {})
        self.model = build_multimodal_model(
            num_clinical_features=checkpoint["num_clinical_features"],
            num_classes=checkpoint["num_classes"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self._healthy_index = (
            self.class_names.index("no_wmd")
            if "no_wmd" in self.class_names
            else None
        )

    def _wmd_signal(self, probs: np.ndarray) -> float:
        if self._healthy_index is not None:
            return float(1.0 - probs[self._healthy_index])
        return float(probs[-1])

    def _clinical_tensor(self, answers: dict[str, float]) -> torch.Tensor:
        return torch.from_numpy(encode_clinical(answers))[None]

    def _reference_clinical(self) -> torch.Tensor:
        return self._clinical_tensor({"age": 55.0})

    @torch.no_grad()
    def predict(
        self, volume: np.ndarray, answers: dict[str, float]
    ) -> tuple[Prediction, ModalityAttribution]:
        tensor = preprocess_volume(volume, self.preprocess)[None]
        clinical = self._clinical_tensor(answers)

        def _prob(vol: torch.Tensor, clin: torch.Tensor) -> np.ndarray:
            return torch.softmax(self.model(vol, clin), dim=1)[0].numpy()

        probs = _prob(tensor, clinical)
        idx = int(probs.argmax())
        prediction = Prediction(
            label=self.class_names[idx],
            label_index=idx,
            confidence=float(probs[idx]),
            probabilities={n: float(p) for n, p in zip(self.class_names, probs)},
        )

        ref_clin = self._reference_clinical()
        neutral_img = torch.zeros_like(tensor)

        baseline = self._wmd_signal(_prob(neutral_img, ref_clin))
        image_alone = self._wmd_signal(_prob(tensor, ref_clin))
        clinical_alone = self._wmd_signal(_prob(neutral_img, clinical))
        combined = self._wmd_signal(probs)

        image_delta = image_alone - baseline
        clinical_delta = clinical_alone - baseline
        total = abs(image_delta) + abs(clinical_delta)
        image_share = abs(image_delta) / total if total > 1e-6 else 0.5

        attribution = ModalityAttribution(
            combined=combined,
            baseline=baseline,
            image_delta=image_delta,
            clinical_delta=clinical_delta,
            image_share=image_share,
            clinical_share=1.0 - image_share,
        )
        return prediction, attribution

    def predict_path(
        self, path: str | Path, answers: dict[str, float]
    ) -> tuple[Prediction, ModalityAttribution]:
        return self.predict(load_volume(path), answers)

    def explain_path(
        self,
        path: str | Path,
        answers: dict[str, float],
        prediction: Prediction,
        overlay_png: str | Path,
        input_png: str | Path,
    ) -> Explanation:
        volume = load_volume(path)
        original_shape = tuple(int(d) for d in volume.shape)

        tensor = preprocess_volume(volume, self.preprocess)[None]
        tensor.requires_grad_(True)
        clinical = self._clinical_tensor(answers)
        wrapper = _ImageBranchWrapper(self.model, clinical)
        cam = grad_cam(wrapper, tensor, prediction.label_index)

        processed = tensor.detach()[0, 0].numpy()
        z = _save_cam_slice(processed, cam, input_png, overlay_png)
        return Explanation(
            original_shape=original_shape,
            processed_shape=tuple(int(d) for d in processed.shape),
            slice_index=z,
            attention_fraction=float((cam[z] > 0.5).mean()),
        )

def save_preview(path: str | Path, out_png: str | Path) -> Path:
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
