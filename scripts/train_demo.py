"""End-to-end demo: generate synthetic data (if needed) and train the models.

Trains both the image-only 3D CNN and the multimodal (MRI + clinical) model on
synthetic data so the web app works end-to-end out of the box.

Run from the project root:
    python scripts/train_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmd.config import (  # noqa: E402
    DATA_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_MULTIMODAL_MODEL_PATH,
    TrainConfig,
)
from wmd.synthetic import generate_dataset  # noqa: E402
from wmd.train import train, train_multimodal  # noqa: E402

from evaluate_demo import main as evaluate_demo  # noqa: E402


def main() -> None:
    synthetic_dir = DATA_DIR / "synthetic"
    manifest = synthetic_dir / "manifest.csv"
    # Always (re)generate with clinical columns + per-etiology classes so the
    # multimodal model can learn the cause (vascular/autoimmune/genetic/...).
    print("Generating synthetic multi-etiology data with clinical columns...")
    manifest = generate_dataset(
        synthetic_dir, n_per_class=80, with_clinical=True, multiclass=True
    )

    config = TrainConfig(epochs=20, batch_size=8)
    print("\n== Training image-only 3D CNN ==")
    img_metrics = train(manifest, config=config, model_path=DEFAULT_MODEL_PATH)
    print(f"Image-only validation metrics: {img_metrics}")

    print("\n== Training multimodal (MRI + clinical) model ==")
    mm_metrics = train_multimodal(
        manifest, config=config, model_path=DEFAULT_MULTIMODAL_MODEL_PATH
    )
    print(f"Multimodal validation metrics: {mm_metrics}")

    print("\n== Evaluating on a held-out test set (for the /performance page) ==")
    evaluate_demo()


if __name__ == "__main__":
    main()
