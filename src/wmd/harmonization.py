"""Cross-scanner harmonization for MRI volumes.

Archival scans come from different machines (1.5T vs 3T; Philips vs Siemens vs
GE) with different shading and intensity scales. Those differences can bias a
CNN into learning "which scanner" instead of "is there disease". This module
adds two per-scan harmonization steps that reduce that bias:

* ``bias_field_correct`` -- a lightweight retrospective bias-field correction
  (an N4-style low-frequency shading removal). It flattens the slow brightness
  gradient scanners impose across the brain.
* ``white_stripe_normalize`` / ``zscore_normalize`` -- intensity normalization
  so the *same tissue* maps to the *same number* across scanners. WhiteStripe
  anchors on normal-appearing white matter; z-score anchors on the whole brain.

All functions are per-scan and dependency-free (NumPy/PyTorch only, matching the
rest of the pipeline -- no SciPy/SimpleITK). They are honest approximations of
the standard neuroimaging tools (ANTs N4, WhiteStripe, ComBat), suitable for a
research/educational demo, not a validated clinical harmonization pipeline.

Note on ComBat: true ComBat harmonization is a *cohort-level* statistical method
-- it needs a batch of scans with known site labels to estimate and remove
site effects. It cannot run on a single scan at inference time, so it is not
included here; the per-scan steps above are what a live upload can use.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def otsu_brain_mask(volume: np.ndarray) -> np.ndarray:
    """Estimate a foreground (brain) mask via Otsu's threshold.

    Background/air voxels are near zero; Otsu picks the intensity threshold that
    best separates the dark background from the brighter head. Returns a boolean
    array the same shape as ``volume``.
    """
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
    """Separable 3D Gaussian blur using depthwise conv (no SciPy)."""
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
    """Remove low-frequency intensity shading (an N4-style bias correction).

    Scanners impose a smooth multiplicative brightness gradient across the field
    of view (surface coils are brighter near the coil). This estimates that slow
    field as a heavily blurred version of the image in the log domain, then
    divides it out so tissue of one type has a consistent intensity everywhere.

    This is a simplified, dependency-free approximation of ANTs/ITK N4 -- good
    enough to flatten obvious shading for a demo, not a validated correction.

    Args:
        volume: 3D float array.
        sigma: Gaussian sigma (in voxels) for the smooth field estimate. Larger
            = smoother field (removes only very low-frequency shading).
        mask: Optional boolean foreground mask; background is left unchanged.

    Returns:
        The bias-corrected 3D float32 array, same shape as the input.
    """
    volume = volume.astype(np.float32)
    if mask is None:
        mask = otsu_brain_mask(volume)

    eps = 1e-6
    positive = np.clip(volume, eps, None)
    log_img = np.log(positive)
    # Estimate the smooth field only from foreground, then blur.
    log_fg = np.where(mask, log_img, 0.0).astype(np.float32)
    smooth = _gaussian_blur_3d(log_fg, sigma)
    # Normalize the field to zero-mean over the brain so we rescale, not darken.
    if np.any(mask):
        smooth = smooth - float(smooth[mask].mean())
    field = np.exp(smooth)
    field = np.where(field < eps, 1.0, field)
    corrected = volume / field
    corrected = np.where(mask, corrected, volume)
    return corrected.astype(np.float32)


def zscore_normalize(volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score normalize intensities within the brain mask (mean 0, std 1).

    Puts every scan on a common intensity scale so absolute scanner units don't
    matter. Background is scored with the same statistics.
    """
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
    """WhiteStripe intensity normalization anchored on normal-appearing tissue.

    WhiteStripe (Shinohara et al., 2014) finds the intensity of normal-appearing
    white matter -- a tissue that should look the same across scanners -- and
    rescales so that intensity becomes a fixed reference. Here we approximate the
    white-matter mode by the peak of the upper-half brain histogram (on FLAIR/T1
    white matter is a dominant bright-ish mode), take a "stripe" of voxels around
    it, and z-score by that stripe's mean/std. The result is comparable across
    scanners regardless of their raw units.

    Args:
        volume: 3D float array.
        mask: Optional foreground mask.
        width: Half-width of the intensity stripe as a fraction of the intensity
            range around the detected tissue mode.

    Returns:
        The normalized 3D float32 array.
    """
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
    # Search the upper half of intensities for the tissue mode (skip CSF/dark).
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
