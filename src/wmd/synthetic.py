"""Generate synthetic brain-like MRI volumes for end-to-end demos and tests.

This is a stand-in for real FLAIR MRI data (e.g. OASIS-3 or the MICCAI WMH
Segmentation Challenge). It produces:
  - "no_wmd": a smooth ellipsoidal "brain" with mild texture.
  - "early_wmd": the same brain plus a few small bright blobs that mimic
    periventricular / deep white matter hyperintensities.

The goal is ONLY to make the full pipeline runnable and demonstrate the model
learning a signal. It is NOT anatomically realistic and has no clinical meaning.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES


def _ellipsoid_mask(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    d, h, w = shape
    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w].astype(np.float32)
    cz, cy, cx = d / 2, h / 2, w / 2
    rz, ry, rx = d * 0.42, h * 0.40, w * 0.38
    jitter = rng.uniform(0.95, 1.05, size=3)
    val = (
        ((zz - cz) / (rz * jitter[0])) ** 2
        + ((yy - cy) / (ry * jitter[1])) ** 2
        + ((xx - cx) / (rx * jitter[2])) ** 2
    )
    return val <= 1.0


def _add_blob(
    volume: np.ndarray,
    center: tuple[int, int, int],
    radius: float,
    intensity: float,
) -> None:
    d, h, w = volume.shape
    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w].astype(np.float32)
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    blob = np.exp(-dist2 / (2 * radius**2)) * intensity
    volume += blob


def make_volume(
    label: int,
    shape: tuple[int, int, int] = (64, 64, 64),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Create one synthetic volume. label=0 -> healthy, label=1 -> with lesions."""
    rng = rng or np.random.default_rng()
    brain = _ellipsoid_mask(shape, rng).astype(np.float32)

    # Base tissue intensity + smooth low-frequency variation + noise.
    volume = brain * rng.uniform(0.55, 0.7)
    texture = rng.normal(0, 0.03, size=shape).astype(np.float32)
    volume = volume + brain * texture

    if label == 1:
        d, h, w = shape
        n_lesions = int(rng.integers(2, 6))
        for _ in range(n_lesions):
            # Place lesions in a periventricular-ish central band.
            center = (
                int(rng.integers(int(d * 0.35), int(d * 0.65))),
                int(rng.integers(int(h * 0.35), int(h * 0.65))),
                int(rng.integers(int(w * 0.30), int(w * 0.70))),
            )
            radius = float(rng.uniform(1.5, 3.0))
            intensity = float(rng.uniform(0.45, 0.7))
            _add_blob(volume, center, radius, intensity)

    volume = np.clip(volume, 0.0, None)
    return volume.astype(np.float32)


def generate_dataset(
    out_dir: str | Path,
    n_per_class: int = 40,
    shape: tuple[int, int, int] = (64, 64, 64),
    seed: int = 42,
) -> Path:
    """Generate a balanced synthetic dataset and a manifest CSV.

    Returns the path to the manifest CSV.
    """
    import nibabel as nib

    out_dir = Path(out_dir)
    vol_dir = out_dir / "volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    rows: list[tuple[str, int]] = []
    for label in range(len(CLASS_NAMES)):
        for i in range(n_per_class):
            volume = make_volume(label, shape=shape, rng=rng)
            fname = f"{CLASS_NAMES[label]}_{i:03d}.nii.gz"
            fpath = vol_dir / fname
            nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), str(fpath))
            rows.append((str(Path("volumes") / fname), label))

    rng.shuffle(rows)
    manifest = out_dir / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "label"])
        writer.writerows(rows)

    return manifest
