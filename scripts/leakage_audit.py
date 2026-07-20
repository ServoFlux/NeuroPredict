from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from wmd.clinical import CLINICAL_FIELD_NAMES
from wmd.config import MODELS_DIR

def _load(manifest: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    sites, clinical, labels = ([], [], [])
    with manifest.open() as fh:
        for row in csv.DictReader(fh):
            sites.append(row['site'])
            clinical.append([float(row.get(name, 0.0) or 0.0) for name in CLINICAL_FIELD_NAMES])
            labels.append(int(row['label']))
    return (sites, np.array(clinical, dtype=np.float32), np.array(labels))

def _onehot(values: list[str], vocab: list[str]) -> np.ndarray:
    return np.array([[1.0 if v == k else 0.0 for k in vocab] for v in values], dtype=np.float32)

def _safe_auc(y: np.ndarray, p: np.ndarray) -> float | None:
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else None

def _baseline(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, y_te: np.ndarray, seed: int) -> dict:
    if x_tr.std() == 0.0:
        return {'test_auc': _safe_auc(y_te, np.full(len(y_te), y_tr.mean())), 'train_cv_auc': None, 'note': 'features are constant (not collected in this dataset)'}
    clf = LogisticRegression(max_iter=1000).fit(x_tr, y_tr)
    n_splits = min(5, int(y_tr.sum()), int((y_tr == 0).sum()))
    cv = cross_val_predict(LogisticRegression(max_iter=1000), x_tr, y_tr, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed), method='predict_proba')[:, 1]
    return {'test_auc': _safe_auc(y_te, clf.predict_proba(x_te)[:, 1]), 'train_cv_auc': _safe_auc(y_tr, cv)}

def audit(train_manifest: Path, test_manifest: Path, image_auc: float | None=None, seed: int=0) -> dict:
    tr_site, tr_clin, tr_y = _load(train_manifest)
    te_site, te_clin, te_y = _load(test_manifest)
    vocab = sorted(set(tr_site))
    return {'n_train': int(len(tr_y)), 'n_test': int(len(te_y)), 'image_model_test_auc': image_auc, 'sites': vocab, 'baselines': {'site_only': _baseline(_onehot(tr_site, vocab), tr_y, _onehot(te_site, vocab), te_y, seed), 'clinical_only': _baseline(tr_clin, tr_y, te_clin, te_y, seed)}}

def _fmt(v: float | None) -> str:
    return f'{v:.3f}' if v is not None else 'n/a'

def main() -> None:
    parser = argparse.ArgumentParser(description='Scanner/site leakage audit (per Dr. Tohka): check whether MRI labels can be predicted from scanner-site metadata alone. If a metadata-only baseline rivals the image model, the reported AUC is partly leaked, not learned from anatomy.')
    parser.add_argument('--train-manifest', type=Path, required=True, help='Training manifest CSV (from prepare_wmh_data.py).')
    parser.add_argument('--test-manifest', type=Path, required=True, help='Held-out test manifest CSV.')
    parser.add_argument('--image-auc', type=float, default=None, help='Image model test ROC-AUC for comparison. Defaults to models/performance_real.json if present.')
    parser.add_argument('--out', type=Path, default=MODELS_DIR / 'leakage_audit.json', help='Where to write the JSON report.')
    args = parser.parse_args()
    image_auc = args.image_auc
    perf_path = MODELS_DIR / 'performance_real.json'
    if image_auc is None and perf_path.exists():
        image_auc = json.loads(perf_path.read_text())['detection']['metrics'].get('roc_auc')
    report = audit(args.train_manifest, args.test_manifest, image_auc)
    report['image_model_test_auc'] = image_auc
    print(f'Leakage audit  (train n={report['n_train']}, test n={report['n_test']}, sites={report['sites']})')
    print(f'  IMAGE model     test AUC = {_fmt(image_auc)}')
    for name, res in report['baselines'].items():
        print(f'  {name:15s} test AUC = {_fmt(res['test_auc'])}  (train CV = {_fmt(res['train_cv_auc'])})')
    worst = max((r['test_auc'] or 0.0 for r in report['baselines'].values()), default=0.0)
    report['verdict'] = 'NO MEANINGFUL LEAKAGE: metadata-only baselines are near chance (0.5), so the image model is not relying on a scanner/site shortcut.' if worst < 0.65 else 'POSSIBLE LEAKAGE: a metadata-only baseline is well above chance; investigate before trusting the image-model AUC.'
    print(f'  Verdict: {report['verdict']}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f'Wrote report to {args.out}')
if __name__ == '__main__':
    main()
