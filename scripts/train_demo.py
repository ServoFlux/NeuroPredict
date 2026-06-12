"""End-to-end demo: generate synthetic data (if needed) and train the model.

Run from the project root:
    python scripts/train_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wmd.config import DATA_DIR, DEFAULT_MODEL_PATH, TrainConfig  # noqa: E402
from wmd.synthetic import generate_dataset  # noqa: E402
from wmd.train import train  # noqa: E402


def main() -> None:
    synthetic_dir = DATA_DIR / "synthetic"
    manifest = synthetic_dir / "manifest.csv"
    if not manifest.exists():
        print("No synthetic data found; generating it now...")
        manifest = generate_dataset(synthetic_dir, n_per_class=100)

    config = TrainConfig(epochs=20, batch_size=8)
    metrics = train(manifest, config=config, model_path=DEFAULT_MODEL_PATH)
    print(f"Demo training complete. Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
