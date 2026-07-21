from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wmd.config import DATA_DIR, DEFAULT_MODEL_PATH, DEFAULT_MULTIMODAL_MODEL_PATH, MODELS_DIR
from wmd.evaluate import build_performance_report
from wmd.synthetic import generate_dataset
TEST_SEED = 1234

def main() -> None:
    for path in (DEFAULT_MODEL_PATH, DEFAULT_MULTIMODAL_MODEL_PATH):
        if not Path(path).exists():
            raise SystemExit(f'Missing model checkpoint: {path}. Run `python scripts/train_demo.py` first.')
    print(f'Generating held-out synthetic test set (seed={TEST_SEED})...')
    manifest = generate_dataset(DATA_DIR / 'synthetic_test', n_per_class=40, seed=TEST_SEED, with_clinical=True, multiclass=True)
    print('Evaluating detection + etiology models on the held-out set...')
    report = build_performance_report(manifest, manifest)
    report['test_seed'] = TEST_SEED
    out_path = MODELS_DIR / 'performance.json'
    out_path.write_text(json.dumps(report, indent=2))
    print(f'Wrote performance report to {out_path}')
    print(f'  Detection: {report['detection']['metrics']}')
    print(f'  Etiology:  {report['etiology']['metrics']}')
if __name__ == '__main__':
    main()
