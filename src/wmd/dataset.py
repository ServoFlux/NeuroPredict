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

from .clinical import CLINICAL_FIELD_NAMES, encode_clinical
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


class MultimodalManifestDataset(Dataset):
    """Loads (volume_tensor, clinical_tensor, label) triples from a manifest CSV.

    The manifest must contain `path`, the clinical questionnaire columns (see
    ``clinical.CLINICAL_FIELD_NAMES``), and a target column. The target defaults
    to ``etiology`` (multi-class cause) when present, otherwise ``label``
    (binary healthy/diseased).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        preprocess: PreprocessConfig | None = None,
        target_column: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.base_dir = self.manifest_path.parent
        self.preprocess = preprocess or PreprocessConfig()
        self.samples: list[tuple[Path, int, dict[str, float]]] = []

        with self.manifest_path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            if "path" not in fields:
                raise ValueError("Manifest must have a 'path' column")
            if target_column is None:
                target_column = "etiology" if "etiology" in fields else "label"
            if target_column not in fields:
                raise ValueError(f"Manifest missing target column: {target_column}")
            self.target_column = target_column
            missing = [c for c in CLINICAL_FIELD_NAMES if c not in fields]
            if missing:
                raise ValueError(f"Manifest missing clinical columns: {missing}")
            for row in reader:
                raw_path = row["path"].strip()
                path = Path(raw_path)
                if not path.is_absolute():
                    path = (self.base_dir / path).resolve()
                label = int(row[target_column])
                answers = {name: float(row[name]) for name in CLINICAL_FIELD_NAMES}
                self.samples.append((path, label, answers))

        if not self.samples:
            raise ValueError(f"No samples found in manifest {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def labels(self) -> list[int]:
        return [label for _, label, _ in self.samples]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        path, label, answers = self.samples[index]
        volume = load_and_preprocess(path, self.preprocess)
        clinical = torch.from_numpy(encode_clinical(answers))
        return volume, clinical, label
