from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

def otsu_brain_mask(volume: np.ndarray) -> np.ndarray:
    flat = volume[np.isfinite(volume)].astype(np.float64)
    if flat.size == 0 or float(flat.max()) <= float(flat.min()):
        return np.ones_like(volume, dtype=bool)

    lo, hi = float(flat.min()), float(flat.max())
    hist, edges = np.histogram(flat, bins=256, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return np.ones_like(volume, dtype=bool)

    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    valid = (weight_bg > 0) & (weight_fg > 0)
    if not np.any(valid):
        return volume > float(np.mean(flat))

    cumsum_mean = np.cumsum(hist * centers)
    global_mean = cumsum_mean[-1]
    mean_bg = np.divide(cumsum_mean, weight_bg, out=np.zeros_like(cumsum_mean), where=weight_bg > 0)
    mean_fg = np.divide(
        global_mean - cumsum_mean, weight_fg, out=np.zeros_like(cumsum_mean), where=weight_fg > 0
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~valid] = -np.inf
    threshold = centers[int(np.argmax(between))]
    return volume > threshold

def _gaussian_blur_3d(volume: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return volume.astype(np.float32)
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    tensor = torch.from_numpy(np.ascontiguousarray(volume))[None, None].float()
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = kernel_1d.numel()
        k = kernel_1d.view(shape)
        pad = [0, 0, 0, 0, 0, 0]
        pad[(2 - dim) * 2] = radius
        pad[(2 - dim) * 2 + 1] = radius
        tensor = F.pad(tensor, pad, mode="replicate")
        tensor = F.conv3d(tensor, k)
    return tensor[0, 0].numpy().astype(np.float32)

def bias_field_correct(
    volume: np.ndarray,
    sigma: float = 8.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    volume = volume.astype(np.float32)
    if mask is None:
        mask = otsu_brain_mask(volume)

    eps = 1e-6
    positive = np.clip(volume, eps, None)
    log_img = np.log(positive)
    log_fg = np.where(mask, log_img, 0.0).astype(np.float32)
    smooth = _gaussian_blur_3d(log_fg, sigma)
    if np.any(mask):
        smooth = smooth - float(smooth[mask].mean())
    field = np.exp(smooth)
    field = np.where(field < eps, 1.0, field)
    corrected = volume / field
    corrected = np.where(mask, corrected, volume)
    return corrected.astype(np.float32)

def zscore_normalize(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    volume = volume.astype(np.float32)
    if mask is None:
        mask = otsu_brain_mask(volume)
    brain = volume[mask]
    if brain.size == 0:
        brain = volume.ravel()
    mean = float(brain.mean())
    std = float(brain.std())
    if std < 1e-6:
        return np.zeros_like(volume, dtype=np.float32)
    return ((volume - mean) / std).astype(np.float32)

def white_stripe_normalize(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    width: float = 0.05,
) -> np.ndarray:
    volume = volume.astype(np.float32)
    if mask is None:
        mask = otsu_brain_mask(volume)
    brain = volume[mask]
    if brain.size == 0:
        return zscore_normalize(volume, mask)

    lo, hi = float(brain.min()), float(brain.max())
    if hi <= lo:
        return zscore_normalize(volume, mask)

    hist, edges = np.histogram(brain, bins=128, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    upper = centers >= (lo + 0.5 * (hi - lo))
    if not np.any(upper):
        upper = np.ones_like(centers, dtype=bool)
    search = np.where(upper, hist, 0)
    mode_intensity = float(centers[int(np.argmax(search))])

    half = width * (hi - lo)
    stripe = brain[(brain >= mode_intensity - half) & (brain <= mode_intensity + half)]
    if stripe.size < 2:
        stripe = brain
    mean = float(stripe.mean())
    std = float(stripe.std())
    if std < 1e-6:
        return zscore_normalize(volume, mask)
    return ((volume - mean) / std).astype(np.float32)
