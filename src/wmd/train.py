"""Training and evaluation for the WMD 3D CNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset

from .config import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    PreprocessConfig,
    TrainConfig,
)
from .dataset import ManifestDataset
from .model import build_model


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split_dataset(
    dataset: ManifestDataset, val_fraction: float, seed: int
) -> tuple[Subset, Subset]:
    indices = np.arange(len(dataset))
    labels = dataset.labels()
    stratify = labels if len(set(labels)) > 1 else None
    train_idx, val_idx = train_test_split(
        indices, test_size=val_fraction, random_state=seed, stratify=stratify
    )
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, float]:
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

    metrics = {"accuracy": float(accuracy_score(all_labels, all_preds))}
    if len(set(all_labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs))
    return metrics


def train(
    manifest_path: str | Path,
    config: TrainConfig | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, float]:
    """Train the 3D CNN on a manifest and save a checkpoint.

    Returns the final validation metrics.
    """
    config = config or TrainConfig()
    _set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ManifestDataset(manifest_path, preprocess=config.preprocess)
    train_ds, val_ds = _split_dataset(dataset, config.val_fraction, config.seed)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = build_model(num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_metric = -1.0
    best_state = model.state_dict()
    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        for volumes, labels in train_loader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(volumes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * volumes.size(0)

        train_loss = running_loss / len(train_ds)
        metrics = evaluate(model, val_loader, device)
        score = metrics.get("roc_auc", metrics["accuracy"])
        msg = " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val {msg}")

        if score >= best_metric:
            best_metric = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, device)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "class_names": list(CLASS_NAMES),
            "num_classes": len(CLASS_NAMES),
            "target_shape": config.preprocess.target_shape,
            "clip_percentiles": config.preprocess.clip_percentiles,
            "val_metrics": final_metrics,
        },
        str(model_path),
    )
    print(f"Saved model to {model_path} | final val {final_metrics}")
    return final_metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the WMD 3D CNN")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        preprocess=PreprocessConfig(),
    )
    train(args.manifest, config=config, model_path=args.model_path)


if __name__ == "__main__":
    main()
