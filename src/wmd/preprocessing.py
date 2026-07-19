"""Loading and preprocessing of MRI volumes.

Supports NIfTI (.nii / .nii.gz) and DICOM (single file or a directory/series).
Produces a normalized, fixed-shape single-channel tensor suitable for the 3D CNN.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import PreprocessConfig

NIFTI_SUFFIXES = (".nii", ".nii.gz")
DICOM_SUFFIXES = (".dcm", ".ima")


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def load_volume(path: str | Path) -> np.ndarray:
    """Load a 3D MRI volume from disk as a float32 numpy array.

    Args:
        path: A NIfTI file, a single DICOM file, or a directory of DICOM slices.

    Returns:
        A 3D float32 array with shape (depth, height, width).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such scan path: {path}")

    if path.is_dir():
        return _load_dicom_series(path)
    if _is_nifti(path):
        return _load_nifti(path)
    if path.suffix.lower() in DICOM_SUFFIXES:
        return _load_dicom_series(path.parent)
    # Fall back: try NIfTI, then DICOM.
    try:
        return _load_nifti(path)
    except Exception:  # noqa: BLE001 - last-resort fallback
        return _load_dicom_series(path.parent)


def _load_nifti(path: Path) -> np.ndarray:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim == 4:
        # Take the first volume of a 4D series.
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {data.shape} from {path}")
    return data


def _load_dicom_series(directory: Path) -> np.ndarray:
    import pydicom

    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES + ("",)
    )
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
        except Exception:  # noqa: BLE001 - skip non-DICOM files
            continue
        if not hasattr(ds, "pixel_array"):
            continue
        slices.append(ds)

    if not slices:
        raise ValueError(f"No readable DICOM slices found in {directory}")

    def _sort_key(ds):  # type: ignore[no-untyped-def]
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
        return float(getattr(ds, "InstanceNumber", 0))

    slices.sort(key=_sort_key)
    volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)
    return volume


def normalize_intensity(
    volume: np.ndarray, clip_percentiles: tuple[float, float]
) -> np.ndarray:
    """Robust intensity normalization to roughly [0, 1].

    Clips to the given percentiles (to suppress extreme bright voxels that are
    common in MRI) and rescales to [0, 1].
    """
    lo, hi = np.percentile(volume, clip_percentiles)
    if hi <= lo:
        hi = float(volume.max())
        lo = float(volume.min())
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    clipped = np.clip(volume, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def resample_to_shape(
    volume: np.ndarray, target_shape: tuple[int, int, int]
) -> np.ndarray:
    """Trilinearly resample a 3D volume to a fixed (D, H, W) shape."""
    tensor = torch.from_numpy(volume)[None, None].float()
    resampled = F.interpolate(
        tensor, size=target_shape, mode="trilinear", align_corners=False
    )
    return resampled[0, 0].numpy()


def median_filter_3d(volume: np.ndarray, size: int = 3) -> np.ndarray:
    """Remove salt-and-pepper (impulse) noise with a 3D median filter.

    Salt-and-pepper noise is isolated very-bright or very-dark voxels -- from
    scanner dead/hot pixels, motion, or (for the film digitizer) dust, scratches
    and camera sensor noise. Those specks mimic small white-matter
    hyperintensities and cause false positives. Replacing each voxel with the
    *median* of its neighbourhood erases lone outliers while preserving real
    lesions and edges (a median ignores extreme values, unlike a blur/average).

    Implemented with PyTorch tensor shifts (no SciPy dependency): for a size-3
    kernel it stacks the 27 neighbours of every voxel and takes the per-voxel
    median. Edges use replicate padding so the brain border is not darkened.

    Args:
        volume: 3D float array (D, H, W).
        size: Odd window size (default 3 -> a 3x3x3 neighbourhood).

    Returns:
        The denoised 3D float32 array, same shape as the input.
    """
    if size < 2:
        return volume.astype(np.float32)
    if size % 2 == 0:
        raise ValueError(f"median filter size must be odd, got {size}")

    pad = size // 2
    tensor = torch.from_numpy(np.ascontiguousarray(volume))[None, None].float()
    padded = F.pad(tensor, (pad,) * 6, mode="replicate")[0, 0]

    neighbours = []
    for dz in range(size):
        for dy in range(size):
            for dx in range(size):
                neighbours.append(
                    padded[
                        dz : dz + volume.shape[0],
                        dy : dy + volume.shape[1],
                        dx : dx + volume.shape[2],
                    ]
                )
    stacked = torch.stack(neighbours, dim=0)
    filtered = stacked.median(dim=0).values
    return filtered.numpy().astype(np.float32)


def preprocess_volume(
    volume: np.ndarray, config: PreprocessConfig | None = None
) -> torch.Tensor:
    """Full preprocessing: (harmonize) -> normalize -> denoise -> resample.

    Optional cross-scanner harmonization runs first (bias-field correction +
    the chosen intensity normalization) so scans from different machines look
    comparable to the CNN. Returns a tensor of shape (1, D, H, W).
    """
    config = config or PreprocessConfig()
    volume = volume.astype(np.float32)

    mask = None
    if config.bias_correct:
        from .harmonization import bias_field_correct, otsu_brain_mask

        mask = otsu_brain_mask(volume)
        volume = bias_field_correct(volume, mask=mask)

    normalized = _apply_intensity_norm(volume, config, mask)
    if config.denoise_median_size and config.denoise_median_size >= 2:
        normalized = median_filter_3d(normalized, config.denoise_median_size)
    resampled = resample_to_shape(normalized, config.target_shape)
    return torch.from_numpy(resampled)[None].float()


def _apply_intensity_norm(
    volume: np.ndarray, config: PreprocessConfig, mask: np.ndarray | None
) -> np.ndarray:
    """Dispatch to the configured intensity normalization mode."""
    mode = config.intensity_norm
    if mode == "minmax":
        return normalize_intensity(volume, config.clip_percentiles)
    if mode in ("zscore", "whitestripe"):
        from .harmonization import (
            otsu_brain_mask,
            white_stripe_normalize,
            zscore_normalize,
        )

        if mask is None:
            mask = otsu_brain_mask(volume)
        if mode == "zscore":
            return zscore_normalize(volume, mask)
        return white_stripe_normalize(volume, mask)
    raise ValueError(
        f"Unknown intensity_norm '{mode}' (expected 'minmax', 'zscore', or 'whitestripe')"
    )


def load_and_preprocess(
    path: str | Path, config: PreprocessConfig | None = None
) -> torch.Tensor:
    """Convenience: load a scan from disk and return a model-ready tensor (1, D, H, W)."""
    volume = load_volume(path)
    return preprocess_volume(volume, config)
