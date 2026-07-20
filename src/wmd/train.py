from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from .clinical import CLINICAL_FIELD_NAMES, NUM_CLINICAL_FEATURES
from .config import CLASS_NAMES, DEFAULT_MODEL_PATH, DEFAULT_MULTIMODAL_MODEL_PATH, ETIOLOGY_CLASS_NAMES, PreprocessConfig, TrainConfig
from .dataset import ManifestDataset, MultimodalManifestDataset
from .model import build_model, build_multimodal_model
def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
def _clone_state(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
def _fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, n_train: int, config: TrainConfig, device: torch.device, evaluate_fn, multimodal: bool) -> dict:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_metric = -1.0
    best_state = _clone_state(model)
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            if multimodal:
                volumes, clinical, labels = batch
                volumes, clinical, labels = (volumes.to(device), clinical.to(device), labels.to(device))
                logits = model(volumes, clinical)
            else:
                volumes, labels = batch
                volumes, labels = (volumes.to(device), labels.to(device))
                logits = model(volumes)
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * volumes.size(0)
        train_loss = running_loss / n_train
        metrics = evaluate_fn(model, val_loader, device)
        score = metrics.get('roc_auc', metrics['accuracy'])
        msg = ' '.join((f'{k}={v:.3f}' for k, v in metrics.items()))
        print(f'epoch {epoch:02d} | train_loss={train_loss:.4f} | val {msg}')
        if score >= best_metric:
            best_metric = score
            best_state = _clone_state(model)
    model.load_state_dict(best_state)
    return best_state
def _split_dataset(dataset: ManifestDataset, val_fraction: float, seed: int) -> tuple[Subset, Subset]:
    indices = np.arange(len(dataset))
    labels = dataset.labels()
    stratify = labels if len(set(labels)) > 1 else None
    train_idx, val_idx = train_test_split(indices, test_size=val_fraction, random_state=seed, stratify=stratify)
    return (Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist()))
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[float] = []
    for volumes, labels in loader:
        volumes = volumes.to(device)
        logits = model(volumes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs[:, 1].tolist())
    metrics = {'accuracy': float(accuracy_score(all_labels, all_preds))}
    if len(set(all_labels)) > 1:
        metrics['roc_auc'] = float(roc_auc_score(all_labels, all_probs))
    return metrics
def train(manifest_path: str | Path, config: TrainConfig | None=None, model_path: str | Path=DEFAULT_MODEL_PATH) -> dict[str, float]:
    config = config or TrainConfig()
    _set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ManifestDataset(manifest_path, preprocess=config.preprocess)
    train_ds, val_ds = _split_dataset(dataset, config.val_fraction, config.seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    best_state = _fit(model, train_loader, val_loader, len(train_ds), config, device, evaluate, multimodal=False)
    final_metrics = evaluate(model, val_loader, device)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'class_names': list(CLASS_NAMES), 'num_classes': len(CLASS_NAMES), 'target_shape': config.preprocess.target_shape, 'clip_percentiles': config.preprocess.clip_percentiles, 'val_metrics': final_metrics}, str(model_path))
    print(f'Saved model to {model_path} | final val {final_metrics}')
    return final_metrics
@torch.no_grad()
def evaluate_multimodal(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: list[np.ndarray] = []
    for volumes, clinical, labels in loader:
        volumes = volumes.to(device)
        clinical = clinical.to(device)
        logits = model(volumes, clinical)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(probs.argmax(axis=1).tolist())
        all_probs.extend(probs)
    metrics = {'accuracy': float(accuracy_score(all_labels, all_preds))}
    n_classes = int(all_probs[0].shape[0]) if all_probs else 0
    if len(set(all_labels)) > 1:
        prob_arr = np.vstack(all_probs)
        if n_classes <= 2:
            metrics['roc_auc'] = float(roc_auc_score(all_labels, prob_arr[:, 1]))
        else:
            metrics['roc_auc'] = float(roc_auc_score(all_labels, prob_arr, multi_class='ovr', average='macro', labels=list(range(n_classes))))
    return metrics
def train_multimodal(manifest_path: str | Path, config: TrainConfig | None=None, model_path: str | Path=DEFAULT_MULTIMODAL_MODEL_PATH) -> dict[str, float]:
    config = config or TrainConfig()
    _set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MultimodalManifestDataset(manifest_path, preprocess=config.preprocess)
    train_ds, val_ds = _split_dataset(dataset, config.val_fraction, config.seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    class_names = ETIOLOGY_CLASS_NAMES if dataset.target_column == 'etiology' else CLASS_NAMES
    model = build_multimodal_model(num_clinical_features=NUM_CLINICAL_FEATURES, num_classes=len(class_names)).to(device)
    best_state = _fit(model, train_loader, val_loader, len(train_ds), config, device, evaluate_multimodal, multimodal=True)
    final_metrics = evaluate_multimodal(model, val_loader, device)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': best_state, 'class_names': list(class_names), 'num_classes': len(class_names), 'multimodal': True, 'clinical_fields': list(CLINICAL_FIELD_NAMES), 'num_clinical_features': NUM_CLINICAL_FEATURES, 'target_shape': config.preprocess.target_shape, 'clip_percentiles': config.preprocess.clip_percentiles, 'val_metrics': final_metrics}, str(model_path))
    print(f'Saved multimodal model to {model_path} | final val {final_metrics}')
    return final_metrics
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train the WMD 3D CNN')
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--epochs', type=int, default=TrainConfig.epochs)
    parser.add_argument('--batch-size', type=int, default=TrainConfig.batch_size)
    parser.add_argument('--lr', type=float, default=TrainConfig.learning_rate)
    parser.add_argument('--multimodal', action='store_true', help='Train the MRI + clinical fusion model (manifest needs clinical columns)')
    return parser.parse_args()
def main() -> None:
    args = _parse_args()
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, preprocess=PreprocessConfig())
    if args.multimodal:
        model_path = args.model_path or str(DEFAULT_MULTIMODAL_MODEL_PATH)
        train_multimodal(args.manifest, config=config, model_path=model_path)
    else:
        model_path = args.model_path or str(DEFAULT_MODEL_PATH)
        train(args.manifest, config=config, model_path=model_path)
if __name__ == '__main__':
    main()
