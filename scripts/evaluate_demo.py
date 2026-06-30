"""Evaluate the trained demo models on a fresh held-out synthetic test set.

Generates a brand-new synthetic dataset with a *different* random seed than the
one used for training, so the models have genuinely never seen it. It then
computes a confusion matrix and metrics for both the detection model and the
cause (etiology) model, and writes the result to ``models/performance.json``
for the web app's /performance page.

Run from the project root:
    python scripts/evaluate_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmd.config import (  # noqa: E402
    DATA_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_MULTIMODAL_MODEL_PATH,
    MODELS_DIR,
)
from wmd.evaluate import build_performance_report  # noqa: E402
from wmd.synthetic import generate_dataset  # noqa: E402

# Seed for the held-out test set. Must differ from the training seed (42) so the
# models are evaluated on data they never saw.
TEST_SEED = 1234


def main() -> None:
    for path in (DEFAULT_MODEL_PATH, DEFAULT_MULTIMODAL_MODEL_PATH):
        if not Path(path).exists():
            raise SystemExit(
                f"Missing model checkpoint: {path}. "
                "Run `python scripts/train_demo.py` first."
            )

    test_dir = DATA_DIR / "synthetic_test"
    print(f"Generating held-out synthetic test set (seed={TEST_SEED})...")
    manifest = generate_dataset(
        test_dir, n_per_class=40, seed=TEST_SEED, with_clinical=True, multiclass=True
    )

    print("Evaluating detection + etiology models on the held-out set...")
    report = build_performance_report(manifest, manifest)
    report["test_seed"] = TEST_SEED

    out_path = MODELS_DIR / "performance.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote performance report to {out_path}")
    det = report["detection"]["metrics"]
    eti = report["etiology"]["metrics"]
    print(f"  Detection: {det}")
    print(f"  Etiology:  {eti}")


if __name__ == "__main__":
    main()
