from __future__ import annotations
import csv
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
from leakage_audit import audit

def _write_manifest(path: Path, rows: list[tuple[str, int]]) -> Path:
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['path', 'site', 'label'])
        writer.writeheader()
        for i, (site, label) in enumerate(rows):
            writer.writerow({'path': f'/x/training/{site}/{i}/pre/FLAIR.nii.gz', 'site': site, 'label': label})
    return path

def test_audit_flags_perfect_site_leak(tmp_path: Path) -> None:
    rows = [('A', 1)] * 8 + [('B', 0)] * 8
    train = _write_manifest(tmp_path / 'train.csv', rows)
    test = _write_manifest(tmp_path / 'test.csv', rows)
    report = audit(train, test, image_auc=0.77)
    assert report['baselines']['site_only']['test_auc'] > 0.9
    assert 'POSSIBLE LEAKAGE' in _verdict(report)

def test_audit_reports_no_leak_when_site_uninformative(tmp_path: Path) -> None:
    rows = [('A', 1), ('A', 0), ('B', 1), ('B', 0)] * 4
    train = _write_manifest(tmp_path / 'train.csv', rows)
    test = _write_manifest(tmp_path / 'test.csv', rows)
    report = audit(train, test, image_auc=0.77)
    assert abs(report['baselines']['site_only']['test_auc'] - 0.5) < 0.1

def _verdict(report: dict) -> str:
    worst = max((r['test_auc'] or 0.0 for r in report['baselines'].values()), default=0.0)
    return 'POSSIBLE LEAKAGE' if worst >= 0.65 else 'NO MEANINGFUL LEAKAGE'
