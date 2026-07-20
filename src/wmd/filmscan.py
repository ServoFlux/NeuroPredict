from __future__ import annotations

import math
from pathlib import Path

import numpy as np

def _volume_norm_bounds(volume: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(volume, (1.0, 99.9))
    if hi <= lo:
        hi, lo = float(volume.max()), float(volume.min())
    return float(lo), float(hi)

def _to_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    norm = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    return (norm * 255).astype(np.uint8)

def grid_shape_for_depth(depth: int, cols: int) -> tuple[int, int]:
    cols = max(1, int(cols))
    rows = max(1, math.ceil(depth / cols))
    return rows, cols

def contact_sheet_from_volume(
    volume: np.ndarray, cols: int = 8, cell: int = 64
) -> np.ndarray:
    from PIL import Image

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    depth = volume.shape[0]
    rows, cols = grid_shape_for_depth(depth, cols)
    sheet = np.zeros((rows * cell, cols * cell), dtype=np.uint8)
    lo, hi = _volume_norm_bounds(volume)

    for idx in range(depth):
        r, c = divmod(idx, cols)
        tile = Image.fromarray(_to_uint8(volume[idx], lo, hi)).resize(
            (cell, cell), Image.BILINEAR
        )
        sheet[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell] = np.asarray(tile)
    return sheet

def save_contact_sheet(
    volume: np.ndarray, out_path: str | Path, cols: int = 8, cell: int = 64
) -> Path:
    from PIL import Image

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet = contact_sheet_from_volume(volume, cols=cols, cell=cell)
    Image.fromarray(sheet).save(str(out_path))
    return out_path

def _auto_crop(gray: np.ndarray, threshold: float = 0.06) -> np.ndarray:
    norm = gray.astype(np.float32) / 255.0
    mask = norm > threshold
    if not mask.any():
        return gray
    ys, xs = np.where(mask)
    return gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

def _load_grayscale(image: str | Path | np.ndarray) -> np.ndarray:
    from PIL import Image

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        return arr.astype(np.uint8)
    return np.asarray(Image.open(str(image)).convert("L"), dtype=np.uint8)

def volume_from_contact_sheet(
    image: str | Path | np.ndarray,
    rows: int,
    cols: int,
    depth: int | None = None,
    auto_crop: bool = True,
) -> np.ndarray:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    gray = _load_grayscale(image)
    if auto_crop:
        gray = _auto_crop(gray)

    h, w = gray.shape
    cell_h, cell_w = h // rows, w // cols
    if cell_h < 2 or cell_w < 2:
        raise ValueError(
            f"Image {w}x{h} too small to split into a {rows}x{cols} grid"
        )

    slices: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            cell = gray[
                r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w
            ]
            slices.append(cell.astype(np.float32))

    if depth is not None:
        slices = slices[: max(1, depth)]
    return np.stack(slices, axis=0).astype(np.float32)
