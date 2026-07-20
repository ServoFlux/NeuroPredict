from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .clinical import CLINICAL_FIELD_NAMES, make_clinical
from .config import ETIOLOGY_CLASS_NAMES
def _ellipsoid_mask(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    d, h, w = shape
    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w].astype(np.float32)
    cz, cy, cx = (d / 2, h / 2, w / 2)
    rz, ry, rx = (d * 0.42, h * 0.4, w * 0.38)
    jitter = rng.uniform(0.95, 1.05, size=3)
    val = ((zz - cz) / (rz * jitter[0])) ** 2 + ((yy - cy) / (ry * jitter[1])) ** 2 + ((xx - cx) / (rx * jitter[2])) ** 2
    return val <= 1.0
def _add_blob(volume: np.ndarray, center: tuple[int, int, int], radius: float, intensity: float) -> None:
    d, h, w = volume.shape
    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w].astype(np.float32)
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    blob = np.exp(-dist2 / (2 * radius ** 2)) * intensity
    volume += blob
def make_volume(label: int, shape: tuple[int, int, int]=(64, 64, 64), rng: np.random.Generator | None=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    brain = _ellipsoid_mask(shape, rng).astype(np.float32)
    volume = brain * rng.uniform(0.55, 0.7)
    texture = rng.normal(0, 0.03, size=shape).astype(np.float32)
    volume = volume + brain * texture
    if label == 1:
        _add_etiology_lesions(volume, 'vascular', rng)
    volume = np.clip(volume, 0.0, None)
    return volume.astype(np.float32)
def _add_etiology_lesions(volume: np.ndarray, etiology: str, rng: np.random.Generator) -> None:
    d, h, w = volume.shape
    def band(lo: float, hi: float, axis: int) -> int:
        size = volume.shape[axis]
        return int(rng.integers(int(size * lo), int(size * hi)))
    if etiology == 'vascular':
        for _ in range(int(rng.integers(3, 7))):
            center = (band(0.35, 0.65, 0), band(0.35, 0.65, 1), band(0.3, 0.7, 2))
            _add_blob(volume, center, float(rng.uniform(1.5, 2.6)), float(rng.uniform(0.45, 0.65)))
    elif etiology == 'autoimmune':
        for _ in range(int(rng.integers(2, 5))):
            center = (band(0.4, 0.6, 0), band(0.42, 0.58, 1), band(0.45, 0.55, 2))
            _add_blob(volume, center, float(rng.uniform(2.0, 3.2)), float(rng.uniform(0.5, 0.7)))
    elif etiology == 'genetic':
        for _ in range(int(rng.integers(1, 3))):
            z = band(0.4, 0.6, 0)
            y = band(0.3, 0.45, 1)
            for x in (band(0.25, 0.38, 2), w - band(0.25, 0.38, 2)):
                _add_blob(volume, (z, y, x), float(rng.uniform(1.8, 2.8)), float(rng.uniform(0.5, 0.68)))
    elif etiology == 'metabolic':
        for _ in range(int(rng.integers(6, 11))):
            center = (band(0.3, 0.7, 0), band(0.3, 0.7, 1), band(0.3, 0.7, 2))
            _add_blob(volume, center, float(rng.uniform(1.2, 2.0)), float(rng.uniform(0.35, 0.5)))
    elif etiology == 'infectious':
        for _ in range(int(rng.integers(1, 4))):
            center = (band(0.3, 0.7, 0), band(0.3, 0.7, 1), band(0.25, 0.75, 2))
            _add_blob(volume, center, float(rng.uniform(2.6, 4.0)), float(rng.uniform(0.5, 0.72)))
def make_etiology_volume(etiology: int, shape: tuple[int, int, int]=(64, 64, 64), rng: np.random.Generator | None=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    brain = _ellipsoid_mask(shape, rng).astype(np.float32)
    volume = brain * rng.uniform(0.55, 0.7)
    texture = rng.normal(0, 0.03, size=shape).astype(np.float32)
    volume = volume + brain * texture
    name = ETIOLOGY_CLASS_NAMES[etiology]
    if name != 'no_wmd':
        _add_etiology_lesions(volume, name, rng)
    volume = np.clip(volume, 0.0, None)
    return volume.astype(np.float32)
def generate_dataset(out_dir: str | Path, n_per_class: int=40, shape: tuple[int, int, int]=(64, 64, 64), seed: int=42, with_clinical: bool=False, clinical_noise: float=0.3, multiclass: bool=False) -> Path:
    import nibabel as nib
    out_dir = Path(out_dir)
    vol_dir = out_dir / 'volumes'
    vol_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if multiclass:
        etiologies = list(range(len(ETIOLOGY_CLASS_NAMES)))
    else:
        etiologies = [0, 1]
    n_etiologies = len(ETIOLOGY_CLASS_NAMES)
    rows: list[list[object]] = []
    for etiology in etiologies:
        for i in range(n_per_class):
            volume = make_etiology_volume(etiology, shape=shape, rng=rng)
            label = 0 if etiology == 0 else 1
            fname = f'{ETIOLOGY_CLASS_NAMES[etiology]}_{i:03d}.nii.gz'
            fpath = vol_dir / fname
            nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), str(fpath))
            row: list[object] = [str(Path('volumes') / fname), label]
            if multiclass:
                row.append(etiology)
            if with_clinical:
                clinical_etiology = etiology
                if rng.random() < clinical_noise:
                    clinical_etiology = int(rng.integers(0, n_etiologies))
                answers = make_clinical(clinical_etiology, rng=rng)
                row.extend((answers[name] for name in CLINICAL_FIELD_NAMES))
            rows.append(row)
    rng.shuffle(rows)
    manifest = out_dir / 'manifest.csv'
    header = ['path', 'label']
    if multiclass:
        header.append('etiology')
    if with_clinical:
        header.extend(CLINICAL_FIELD_NAMES)
    with manifest.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return manifest
