"""PyTorch dataset for MRI volumes described by a manifest CSV.

The manifest is a CSV with at least two columns: `path,label`.
- `path`: path to a NIfTI file, DICOM file, or DICOM directory (absolute, or
  relative to the manifest's location).
- `label`: integer class index (0-based) matching `config.CLASS_NAMES`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import PreprocessConfig
from .preprocessing import load_and_preprocess


class ManifestDataset(Dataset):
    """Loads (volume_tensor, label) pairs from a manifest CSV."""

    def __init__(
        self,
        manifest_path: str | Path,
        preprocess: PreprocessConfig | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.preprocess = preprocess or PreprocessConfig()
        self.samples: list[tuple[Path, int]] = []

        with self.manifest_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "path" not in reader.fieldnames:
                raise ValueError("Manifest must have a header with a 'path' column")
            for row in reader:
                raw_path = row["path"].strip()
                path = Path(raw_path)
                if not path.is_absolute():
                    path = (self.base_dir / path).resolve()
                label = int(row["label"])
                self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No samples found in manifest {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def labels(self) -> list[int]:
        return [label for _, label in self.samples]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        volume = load_and_preprocess(path, self.preprocess)
        return volume, label
