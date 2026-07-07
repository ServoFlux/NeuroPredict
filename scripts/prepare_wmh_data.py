"""Prepare a MICCAI WMH Challenge dataset for NeuroPredict training.

This script reads the WMH Segmentation Challenge directory structure and
produces a manifest CSV that plugs directly into the existing training code
(``ManifestDataset`` / ``MultimodalManifestDataset``).

Expected input layout (the standard WMH challenge training set)::

    <wmh_root>/
        Amsterdam/ (or GE3T/)
            0/
                pre/
                    FLAIR.nii.gz
                    T1.nii.gz
                wmh.nii.gz          # radiologist WMH mask
            1/ ...
        Singapore/ (or GE15T/)
            ...
        Utrecht/ (or UMC/)
            ...

Each subject gets a binary detection label (0 = low WMH burden, 1 = significant
WMH) derived by thresholding the total WMH lesion volume. The threshold is
configurable; the default (``--threshold-ml 1.0``) labels subjects with > 1 mL
of WMH as *early_wmd*. Subjects below the threshold are labeled *no_wmd*.

Usage::

    python scripts/prepare_wmh_data.py \\
        --wmh-root /path/to/WMH/training \\
        --out-dir data/wmh_real \\
        --threshold-ml 1.0

The output manifest (``data/wmh_real/manifest.csv``) is ready for::

    python -m wmd.train --manifest data/wmh_real/manifest.csv

Note: the WMH challenge data does NOT include clinical questionnaire answers, so
the manifest has dummy zero values for the clinical columns. The detection
(image-only) model can train directly; the multimodal cause model will only see
imaging signal.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

# Allow running from the repo root (``python scripts/prepare_wmh_data.py``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from wmd.clinical import CLINICAL_FIELD_NAMES  # noqa: E402

# ---- Helpers ----------------------------------------------------------------

# Known site directory names in the WMH challenge (training set).
_SITE_ALIASES: dict[str, str] = {
    "amsterdam": "Amsterdam",
    "ge3t": "Amsterdam",
    "singapore": "Singapore",
    "ge15t": "Singapore",
    "utrecht": "Utrecht",
    "umc": "Utrecht",
}


def _site_of(subject_dir: Path, wmh_root: Path) -> str:
    """Infer the acquisition site from the path (first component under root)."""
    try:
        rel = subject_dir.relative_to(wmh_root)
    except ValueError:
        rel = subject_dir
    for part in rel.parts:
        key = part.lower()
        if key in _SITE_ALIASES:
            return _SITE_ALIASES[key]
    # Fall back to the first path component.
    return rel.parts[0] if rel.parts else "unknown"


def _find_subjects(wmh_root: Path) -> list[dict[str, Path]]:
    """Recursively walk the WMH directory tree and return one dict per subject.

    Robust to the different challenge layouts, e.g. both
    ``Utrecht/11/pre/FLAIR.nii.gz`` and the extra scanner level in
    ``Amsterdam/GE3T/137/pre/FLAIR.nii.gz``. A subject is any folder that
    contains ``pre/FLAIR.nii(.gz)`` and a sibling ``wmh.nii(.gz)`` mask.

    Each dict has keys: ``flair``, ``t1`` (optional), ``mask``, ``site``,
    ``subject_dir``.
    """
    subjects: list[dict[str, Path]] = []
    seen: set[Path] = set()
    flair_files = sorted(wmh_root.rglob("FLAIR.nii.gz")) + sorted(
        wmh_root.rglob("FLAIR.nii")
    )
    for flair in flair_files:
        # Only take the pre-processed FLAIR (skip the raw orig/ copy).
        if flair.parent.name != "pre":
            continue
        subj_dir = flair.parent.parent
        if subj_dir in seen:
            continue
        mask = subj_dir / "wmh.nii.gz"
        if not mask.exists():
            mask = subj_dir / "wmh.nii"
        if not mask.exists():
            print(f"  [skip] no WMH mask found for {subj_dir}")
            continue
        t1 = subj_dir / "pre" / "T1.nii.gz"
        if not t1.exists():
            t1 = subj_dir / "pre" / "T1.nii"
        seen.add(subj_dir)
        subjects.append({
            "flair": flair,
            "t1": t1 if t1.exists() else None,
            "mask": mask,
            "site": _site_of(subj_dir, wmh_root),
            "subject_dir": subj_dir,
        })
    return subjects


def _wmh_volume_ml(mask_path: Path) -> float:
    """Compute total WMH volume in millilitres from a NIfTI mask.

    WMH voxels are labeled 1 in the challenge masks (label 2 = other pathology,
    which we ignore).
    """
    img = nib.load(str(mask_path))
    mask = np.asarray(img.dataobj)
    wmh_voxels = int((mask == 1).sum())
    # Voxel volume from the affine (product of absolute values of the diagonal).
    voxel_vol_mm3 = float(np.abs(np.linalg.det(img.affine[:3, :3])))
    return wmh_voxels * voxel_vol_mm3 / 1000.0  # mm^3 -> mL


# ---- Main -------------------------------------------------------------------


def prepare(
    wmh_root: Path,
    out_dir: Path,
    threshold_ml: float = 1.0,
    copy_scans: bool = False,
) -> Path:
    """Build a manifest CSV from the WMH challenge directory.

    Args:
        wmh_root: Path to the WMH challenge training directory.
        out_dir: Where to write the manifest (and optionally symlink/copy scans).
        threshold_ml: WMH volume threshold in mL. Subjects above this are
            labeled ``early_wmd`` (1); at or below are ``no_wmd`` (0).
        copy_scans: If True, copy FLAIR files into ``out_dir/scans/``. If False
            (default), the manifest references the original paths.

    Returns:
        Path to the written manifest CSV.
    """
    subjects = _find_subjects(wmh_root)
    if not subjects:
        raise SystemExit(
            f"No subjects found under {wmh_root}. "
            "Expected site subdirectories (Amsterdam/, Singapore/, Utrecht/) "
            "each containing numbered subject folders with pre/FLAIR.nii.gz "
            "and wmh.nii.gz."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    scan_dir = out_dir / "scans"
    if copy_scans:
        scan_dir.mkdir(exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    fieldnames = ["path", "label", "wmh_volume_ml", "site"] + list(CLINICAL_FIELD_NAMES)

    stats = {"no_wmd": 0, "early_wmd": 0}
    rows: list[dict[str, object]] = []

    print(f"Found {len(subjects)} subjects in {wmh_root}")
    for subj in subjects:
        vol_ml = _wmh_volume_ml(subj["mask"])
        label = 1 if vol_ml > threshold_ml else 0
        label_name = "early_wmd" if label == 1 else "no_wmd"
        stats[label_name] += 1

        if copy_scans:
            dest = scan_dir / f"{subj['site']}_{subj['subject_dir'].name}_FLAIR.nii.gz"
            if not dest.exists():
                shutil.copy2(str(subj["flair"]), str(dest))
            scan_path = str(dest)
        else:
            scan_path = str(subj["flair"].resolve())

        row: dict[str, object] = {
            "path": scan_path,
            "label": label,
            "wmh_volume_ml": round(vol_ml, 3),
            "site": subj["site"],
        }
        # Fill clinical columns with zeros (not available in the WMH dataset).
        for col in CLINICAL_FIELD_NAMES:
            row[col] = 0.0
        rows.append(row)
        print(f"  {subj['site']}/{subj['subject_dir'].name}: "
              f"WMH={vol_ml:.2f} mL -> {label_name}")

    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest written to {manifest_path}")
    print(f"  no_wmd: {stats['no_wmd']}  |  early_wmd: {stats['early_wmd']}")
    print(f"  threshold: {threshold_ml} mL")
    if stats["no_wmd"] == 0 or stats["early_wmd"] == 0:
        print(
            "  WARNING: all subjects ended up in one class. Consider adjusting "
            "--threshold-ml so there is a meaningful split."
        )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a MICCAI WMH Challenge dataset for NeuroPredict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--wmh-root", type=Path, required=True,
        help="Path to the WMH challenge training directory (contains site folders).",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/wmh_real"),
        help="Output directory for the manifest CSV (default: data/wmh_real).",
    )
    parser.add_argument(
        "--threshold-ml", type=float, default=1.0,
        help="WMH volume threshold in mL for binary labeling (default: 1.0).",
    )
    parser.add_argument(
        "--copy-scans", action="store_true",
        help="Copy FLAIR files into the output directory instead of referencing originals.",
    )
    args = parser.parse_args()
    prepare(args.wmh_root, args.out_dir, args.threshold_ml, args.copy_scans)


if __name__ == "__main__":
    main()
