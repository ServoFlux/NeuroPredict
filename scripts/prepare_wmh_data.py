from __future__ import annotations
import argparse
import csv
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import nibabel as nib
import numpy as np
from wmd.clinical import CLINICAL_FIELD_NAMES
_SITE_ALIASES: dict[str, str] = {'amsterdam': 'Amsterdam', 'ge3t': 'Amsterdam', 'singapore': 'Singapore', 'ge15t': 'Singapore', 'utrecht': 'Utrecht', 'umc': 'Utrecht'}
def _site_of(subject_dir: Path, wmh_root: Path) -> str:
    try:
        rel = subject_dir.relative_to(wmh_root)
    except ValueError:
        rel = subject_dir
    for part in rel.parts:
        key = part.lower()
        if key in _SITE_ALIASES:
            return _SITE_ALIASES[key]
    return rel.parts[0] if rel.parts else 'unknown'
def _find_subjects(wmh_root: Path) -> list[dict[str, Path]]:
    subjects: list[dict[str, Path]] = []
    seen: set[Path] = set()
    flair_files = sorted(wmh_root.rglob('FLAIR.nii.gz')) + sorted(wmh_root.rglob('FLAIR.nii'))
    for flair in flair_files:
        if flair.parent.name != 'pre':
            continue
        subj_dir = flair.parent.parent
        if subj_dir in seen:
            continue
        mask = subj_dir / 'wmh.nii.gz'
        if not mask.exists():
            mask = subj_dir / 'wmh.nii'
        if not mask.exists():
            print(f'  [skip] no WMH mask found for {subj_dir}')
            continue
        t1 = subj_dir / 'pre' / 'T1.nii.gz'
        if not t1.exists():
            t1 = subj_dir / 'pre' / 'T1.nii'
        seen.add(subj_dir)
        subjects.append({'flair': flair, 't1': t1 if t1.exists() else None, 'mask': mask, 'site': _site_of(subj_dir, wmh_root), 'subject_dir': subj_dir})
    return subjects
def _wmh_volume_ml(mask_path: Path) -> float:
    img = nib.load(str(mask_path))
    mask = np.asarray(img.dataobj)
    wmh_voxels = int((mask == 1).sum())
    voxel_vol_mm3 = float(np.abs(np.linalg.det(img.affine[:3, :3])))
    return wmh_voxels * voxel_vol_mm3 / 1000.0
def prepare(wmh_root: Path, out_dir: Path, threshold_ml: float=1.0, copy_scans: bool=False) -> Path:
    subjects = _find_subjects(wmh_root)
    if not subjects:
        raise SystemExit(f'No subjects found under {wmh_root}. Expected site subdirectories (Amsterdam/, Singapore/, Utrecht/) each containing numbered subject folders with pre/FLAIR.nii.gz and wmh.nii.gz.')
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_dir = out_dir / 'scans'
    if copy_scans:
        scan_dir.mkdir(exist_ok=True)
    manifest_path = out_dir / 'manifest.csv'
    fieldnames = ['path', 'label', 'wmh_volume_ml', 'site'] + list(CLINICAL_FIELD_NAMES)
    stats = {'no_wmd': 0, 'early_wmd': 0}
    rows: list[dict[str, object]] = []
    print(f'Found {len(subjects)} subjects in {wmh_root}')
    for subj in subjects:
        vol_ml = _wmh_volume_ml(subj['mask'])
        label = 1 if vol_ml > threshold_ml else 0
        label_name = 'early_wmd' if label == 1 else 'no_wmd'
        stats[label_name] += 1
        if copy_scans:
            dest = scan_dir / f'{subj['site']}_{subj['subject_dir'].name}_FLAIR.nii.gz'
            if not dest.exists():
                shutil.copy2(str(subj['flair']), str(dest))
            scan_path = str(dest)
        else:
            scan_path = str(subj['flair'].resolve())
        row: dict[str, object] = {'path': scan_path, 'label': label, 'wmh_volume_ml': round(vol_ml, 3), 'site': subj['site']}
        for col in CLINICAL_FIELD_NAMES:
            row[col] = 0.0
        rows.append(row)
        print(f'  {subj['site']}/{subj['subject_dir'].name}: WMH={vol_ml:.2f} mL -> {label_name}')
    with manifest_path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nManifest written to {manifest_path}')
    print(f'  no_wmd: {stats['no_wmd']}  |  early_wmd: {stats['early_wmd']}')
    print(f'  threshold: {threshold_ml} mL')
    if stats['no_wmd'] == 0 or stats['early_wmd'] == 0:
        print('  WARNING: all subjects ended up in one class. Consider adjusting --threshold-ml so there is a meaningful split.')
    return manifest_path
def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a MICCAI WMH Challenge dataset for NeuroPredict.', formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--wmh-root', type=Path, required=True, help='Path to the WMH challenge training directory (contains site folders).')
    parser.add_argument('--out-dir', type=Path, default=Path('data/wmh_real'), help='Output directory for the manifest CSV (default: data/wmh_real).')
    parser.add_argument('--threshold-ml', type=float, default=1.0, help='WMH volume threshold in mL for binary labeling (default: 1.0).')
    parser.add_argument('--copy-scans', action='store_true', help='Copy FLAIR files into the output directory instead of referencing originals.')
    args = parser.parse_args()
    prepare(args.wmh_root, args.out_dir, args.threshold_ml, args.copy_scans)
if __name__ == '__main__':
    main()
