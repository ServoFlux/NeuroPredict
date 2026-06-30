"""Evaluate trained checkpoints on a held-out test set.

Produces an honest performance report: a confusion matrix plus the metrics
derived from it (accuracy, ROC-AUC, and -- for the binary detection model --
sensitivity and specificity). The report is JSON-serializable so the web app
can render it on the /performance page without re-running inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from .config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_MULTIMODAL_MODEL_PATH,
    PreprocessConfig,
)
from .dataset import ManifestDataset, MultimodalManifestDataset
from .model import build_model, build_multimodal_model


def _preprocess_from_checkpoint(checkpoint: dict) -> PreprocessConfig:
    return PreprocessConfig(
        target_shape=tuple(checkpoint["target_shape"]),
        clip_percentiles=tuple(checkpoint["clip_percentiles"]),
    )


@torch.no_grad()
def evaluate_detection(
    manifest_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> dict[str, object]:
    """Evaluate the image-only binary detection model on a held-out manifest."""
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    class_names: list[str] = list(checkpoint["class_names"])
    preprocess = _preprocess_from_checkpoint(checkpoint)
    model = build_model(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = ManifestDataset(manifest_path, preprocess=preprocess)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    labels: list[int] = []
    preds: list[int] = []
    pos_probs: list[float] = []
    for volumes, batch_labels in loader:
        probs = torch.softmax(model(volumes), dim=1).numpy()
        labels.extend(batch_labels.numpy().tolist())
        preds.extend(probs.argmax(axis=1).tolist())
        pos_probs.extend(probs[:, 1].tolist())

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    report: dict[str, object] = {
        "task": "Detection (white matter disease: yes / no)",
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(labels),
        "metrics": _binary_metrics(cm, labels, pos_probs),
    }
    return report


@torch.no_grad()
def evaluate_etiology(
    manifest_path: str | Path,
    model_path: str | Path = DEFAULT_MULTIMODAL_MODEL_PATH,
) -> dict[str, object]:
    """Evaluate the multimodal cause-prediction model on a held-out manifest."""
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    class_names: list[str] = list(checkpoint["class_names"])
    preprocess = _preprocess_from_checkpoint(checkpoint)
    model = build_multimodal_model(
        num_clinical_features=checkpoint["num_clinical_features"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    dataset = MultimodalManifestDataset(
        manifest_path, preprocess=preprocess, target_column="etiology"
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    labels: list[int] = []
    preds: list[int] = []
    prob_rows: list[np.ndarray] = []
    for volumes, clinical, batch_labels in loader:
        probs = torch.softmax(model(volumes, clinical), dim=1).numpy()
        labels.extend(batch_labels.numpy().tolist())
        preds.extend(probs.argmax(axis=1).tolist())
        prob_rows.extend(probs)

    n_classes = len(class_names)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    metrics: dict[str, float] = {"accuracy": float(accuracy_score(labels, preds))}
    if len(set(labels)) > 1:
        metrics["roc_auc_macro"] = float(
            roc_auc_score(
                labels,
                np.vstack(prob_rows),
                multi_class="ovr",
                average="macro",
                labels=list(range(n_classes)),
            )
        )
    report: dict[str, object] = {
        "task": "Cause / etiology (5 causes + healthy)",
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(labels),
        "metrics": metrics,
    }
    return report


def _binary_metrics(
    cm: np.ndarray, labels: list[int], pos_probs: list[float]
) -> dict[str, float]:
    """Accuracy, ROC-AUC, sensitivity, specificity from a 2x2 matrix."""
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    total = tn + fp + fn + tp
    metrics: dict[str, float] = {
        "accuracy": (tp + tn) / total if total else 0.0,
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
    }
    if len(set(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, pos_probs))
    return metrics


def build_performance_report(
    detection_manifest: str | Path,
    etiology_manifest: str | Path,
    detection_model: str | Path = DEFAULT_MODEL_PATH,
    etiology_model: str | Path = DEFAULT_MULTIMODAL_MODEL_PATH,
) -> dict[str, object]:
    """Full performance report for both models on a held-out test set."""
    return {
        "note": (
            "Metrics computed on a freshly generated synthetic test set the "
            "models never saw during training. Synthetic data is a stand-in "
            "for real MRI, so these numbers demonstrate the pipeline rather "
            "than clinical performance."
        ),
        "detection": evaluate_detection(detection_manifest, detection_model),
        "etiology": evaluate_etiology(etiology_manifest, etiology_model),
    }
