from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wmd.config import DATA_DIR, DEFAULT_MODEL_PATH, DEFAULT_MULTIMODAL_MODEL_PATH, TrainConfig
from wmd.synthetic import generate_dataset
from wmd.train import train, train_multimodal
def main() -> None:
    synthetic_dir = DATA_DIR / 'synthetic'
    manifest = synthetic_dir / 'manifest.csv'
    print('Generating synthetic multi-etiology data with clinical columns...')
    manifest = generate_dataset(synthetic_dir, n_per_class=80, with_clinical=True, multiclass=True)
    config = TrainConfig(epochs=20, batch_size=8)
    print('\n== Training image-only 3D CNN ==')
    img_metrics = train(manifest, config=config, model_path=DEFAULT_MODEL_PATH)
    print(f'Image-only validation metrics: {img_metrics}')
    print('\n== Training multimodal (MRI + clinical) model ==')
    mm_metrics = train_multimodal(manifest, config=config, model_path=DEFAULT_MULTIMODAL_MODEL_PATH)
    print(f'Multimodal validation metrics: {mm_metrics}')
    print('\n== Evaluating on a held-out synthetic test set ==')
    import evaluate_demo
    evaluate_demo.main()
if __name__ == '__main__':
    main()
