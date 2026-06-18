"""Explainability for the 3D CNN via Grad-CAM.

Grad-CAM opens up the "black box": instead of only returning a label, it shows
*which regions of the brain* drove the prediction by weighting the last
convolutional feature maps by the gradient of the target class score.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def grad_cam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    target_layer: nn.Module | None = None,
) -> np.ndarray:
    """Compute a 3D Grad-CAM map for ``class_idx``.

    Args:
        model: The trained classifier.
        input_tensor: A (1, 1, D, H, W) input volume.
        class_idx: Index of the class whose evidence to visualize.
        target_layer: Layer to attach hooks to. Defaults to the model's
            convolutional feature extractor (``model.features``).

    Returns:
        A (D, H, W) float array in [0, 1], upsampled to the input shape.
    """
    if target_layer is None:
        target_layer = model.features  # type: ignore[attr-defined]

    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module: nn.Module, _inp: object, out: torch.Tensor) -> None:
        activations["value"] = out

    def backward_hook(
        _module: nn.Module, _grad_in: object, grad_out: tuple[torch.Tensor, ...]
    ) -> None:
        gradients["value"] = grad_out[0]

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_full_backward_hook(backward_hook)
    try:
        logits = model(input_tensor)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()

        acts = activations["value"][0]  # (C, d, h, w)
        grads = gradients["value"][0]  # (C, d, h, w)
        weights = grads.mean(dim=(1, 2, 3))  # (C,) — importance of each channel
        cam = torch.relu((weights[:, None, None, None] * acts).sum(dim=0))  # (d, h, w)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()

    cam = F.interpolate(
        cam[None, None].detach(),
        size=tuple(input_tensor.shape[2:]),
        mode="trilinear",
        align_corners=False,
    )[0, 0].numpy()

    lo, hi = float(cam.min()), float(cam.max())
    if hi > lo:
        cam = (cam - lo) / (hi - lo)
    else:
        cam = np.zeros_like(cam)
    return cam.astype(np.float32)


def heatmap_rgb(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to a jet-style RGB array (..., 3) in [0, 1]."""
    x = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def overlay_cam_on_slice(
    base_slice: np.ndarray, cam_slice: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """Blend a Grad-CAM heatmap over a grayscale slice. Returns uint8 RGB."""
    base = np.clip(base_slice, 0.0, 1.0)
    base_rgb = np.stack([base, base, base], axis=-1)
    heat = heatmap_rgb(cam_slice)
    weight = (np.clip(cam_slice, 0.0, 1.0) * alpha)[..., None]
    blended = base_rgb * (1.0 - weight) + heat * weight
    return (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)


def most_salient_axial_index(cam: np.ndarray) -> int:
    """Return the axial slice index with the strongest Grad-CAM response."""
    return int(cam.sum(axis=(1, 2)).argmax())
