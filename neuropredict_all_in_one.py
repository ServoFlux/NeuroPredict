"""NeuroPredict — the entire project condensed into one readable file.

This single file reproduces the whole NeuroPredict *software* pipeline so it can
be read and explained end to end:

  1. Synthetic brain-MRI generator (a stand-in for real FLAIR MRI).
  2. Clinical + genomic questionnaire (encodes patient answers into numbers).
  3. A 3D CNN that detects white matter disease from the MRI.
  4. A multimodal model that fuses the MRI with the questionnaire to predict the
     *cause* (vascular / autoimmune / genetic / metabolic / infectious).
  5. Grad-CAM (a heatmap of where the CNN looked).
  6. Training on a held-out split, and evaluation with a **confusion matrix**.
  7. A FastAPI web app: a prediction page and a Model-Performance page that draws
     both confusion matrices — deployable to Hugging Face Spaces as-is.

The full multi-file project still lives in ``src/`` / ``webapp/``; this file is
an added, self-contained "read the whole thing in one place" version. It is a
research/education demo trained on synthetic data — NOT a medical device.

Run it:
    python neuropredict_all_in_one.py train     # train + print confusion matrix
    python neuropredict_all_in_one.py serve      # launch the web app on :8000
    python neuropredict_all_in_one.py            # train (if needed) then serve
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

# Image-only model: is white matter disease present?
CLASS_NAMES = ("no_wmd", "early_wmd")

# Multimodal model: which *cause*? Index 0 is healthy.
ETIOLOGY_CLASS_NAMES = (
    "no_wmd", "vascular", "autoimmune", "genetic", "metabolic", "infectious",
)
ETIOLOGY_LABELS = {
    "no_wmd": "No white matter disease",
    "vascular": "Vascular (small-vessel disease)",
    "autoimmune": "Autoimmune (e.g. multiple sclerosis)",
    "genetic": "Genetic (e.g. CADASIL / CARASIL)",
    "metabolic": "Metabolic (e.g. leukodystrophy, B12 deficiency)",
    "infectious": "Infectious (e.g. HIV, Lyme, PML)",
}
# Educational next-steps per cause (NOT medical advice).
ETIOLOGY_NEXT_STEPS = {
    "no_wmd": ["No disease flagged — not a diagnosis; see a doctor if you have symptoms.",
               "Protect brain health: exercise, good diet, control BP/sugar/cholesterol."],
    "vascular": ["Share with a doctor or neurologist.",
                 "Ask about controlling blood pressure, diabetes and cholesterol.",
                 "Exercise, heart-healthy diet, stop smoking."],
    "autoimmune": ["Ask for a neurology referral to evaluate for MS.",
                   "May need a contrast MRI of brain/spine and a lumbar puncture."],
    "genetic": ["Consider genetic counseling (CADASIL/NOTCH3, CARASIL/HTRA1, COL4A1).",
                "Family members may be screened; manage stroke risk factors."],
    "metabolic": ["See a physician for a metabolic workup (e.g. vitamin B12, thyroid).",
                  "Some metabolic causes are treatable — early evaluation matters."],
    "infectious": ["See a doctor promptly for an infection workup (HIV, Lyme, etc.).",
                   "Mention recent infections, travel, tick exposure or fevers."],
}

RESEARCH_DISCLAIMER = (
    "This tool is for research and educational purposes only. It is NOT a "
    "medical device and must NOT be used for diagnosis or clinical "
    "decision-making. Always consult a qualified clinician."
)

# Where the trained weights + metrics are cached.
ARTIFACT_PATH = Path(__file__).resolve().parent / "models" / "all_in_one.pt"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    seed: int = 42
    shape: tuple[int, int, int] = (48, 48, 48)
    n_per_class: int = 40           # training samples per etiology
    test_n_per_class: int = 40      # held-out test samples per etiology
    test_seed: int = 1234           # different seed => models never see the test set


# ---------------------------------------------------------------------------
# 2. CLINICAL / GENOMIC QUESTIONNAIRE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClinicalField:
    name: str
    label: str
    kind: str            # "age" or "binary"
    category: str = "History"


# Order defines the feature-vector layout and must stay stable.
CLINICAL_FIELDS = (
    ClinicalField("age", "Age (years)", "age", "Demographics"),
    ClinicalField("hypertension", "High blood pressure", "binary", "History"),
    ClinicalField("diabetes", "Diabetes", "binary", "History"),
    ClinicalField("prior_stroke", "Prior stroke or TIA", "binary", "History"),
    ClinicalField("smoking", "Current or former smoker", "binary", "History"),
    ClinicalField("high_cholesterol", "High cholesterol", "binary", "History"),
    ClinicalField("autoimmune_history", "Autoimmune disease (e.g. MS, lupus)", "binary", "History"),
    ClinicalField("recent_cns_infection", "Recent/chronic CNS infection (HIV, Lyme)", "binary", "History"),
    ClinicalField("metabolic_disorder", "Metabolic disorder (B12 deficiency, leukodystrophy)", "binary", "History"),
    ClinicalField("memory_problems", "Memory problems", "binary", "Symptoms"),
    ClinicalField("slow_gait", "Slow walking / gait changes", "binary", "Symptoms"),
    ClinicalField("balance_problems", "Balance problems / falls", "binary", "Symptoms"),
    ClinicalField("poor_concentration", "Reduced concentration", "binary", "Symptoms"),
    ClinicalField("low_mood", "Low mood / depression", "binary", "Symptoms"),
    ClinicalField("urinary_incontinence", "Urinary incontinence", "binary", "Symptoms"),
    ClinicalField("apoe4_carrier", "APOE \u03b54 carrier", "binary", "Genomic"),
    ClinicalField("notch3_variant", "NOTCH3 variant (CADASIL)", "binary", "Genomic"),
    ClinicalField("htra1_variant", "HTRA1 variant (CARASIL)", "binary", "Genomic"),
    ClinicalField("col4a1_variant", "COL4A1 / COL4A2 variant", "binary", "Genomic"),
    ClinicalField("mthfr_677tt", "MTHFR C677T (TT genotype)", "binary", "Genomic"),
    ClinicalField("family_history_stroke", "Family history of stroke / vascular dementia", "binary", "Genomic"),
    ClinicalField("high_wmh_prs", "Elevated WMH polygenic risk score", "binary", "Genomic"),
)
CLINICAL_FIELD_NAMES = tuple(f.name for f in CLINICAL_FIELDS)
NUM_CLINICAL_FEATURES = len(CLINICAL_FIELDS)
_AGE_SCALE = 100.0
_BASELINE_P = 0.07

# Per-cause synthetic profile: an age range and prevalence of each binary field.
_ETIOLOGY_PROFILES: dict[str, dict[str, object]] = {
    "no_wmd": {"age": (40, 66), "fields": {}, "baseline": 0.04},
    "vascular": {"age": (62, 86), "fields": {
        "hypertension": 0.80, "diabetes": 0.50, "high_cholesterol": 0.60, "smoking": 0.50,
        "prior_stroke": 0.40, "slow_gait": 0.55, "balance_problems": 0.45,
        "urinary_incontinence": 0.45, "memory_problems": 0.45, "poor_concentration": 0.40,
        "high_wmh_prs": 0.50, "family_history_stroke": 0.40, "apoe4_carrier": 0.35}},
    "autoimmune": {"age": (25, 50), "fields": {
        "autoimmune_history": 0.85, "balance_problems": 0.55, "poor_concentration": 0.55,
        "low_mood": 0.50, "memory_problems": 0.40, "urinary_incontinence": 0.40, "slow_gait": 0.35}},
    "genetic": {"age": (35, 60), "fields": {
        "notch3_variant": 0.60, "htra1_variant": 0.30, "col4a1_variant": 0.25,
        "family_history_stroke": 0.80, "apoe4_carrier": 0.50, "prior_stroke": 0.45,
        "memory_problems": 0.45, "slow_gait": 0.45, "high_wmh_prs": 0.55, "balance_problems": 0.40}},
    "metabolic": {"age": (30, 65), "fields": {
        "metabolic_disorder": 0.85, "diabetes": 0.55, "mthfr_677tt": 0.55,
        "poor_concentration": 0.55, "memory_problems": 0.45, "low_mood": 0.35, "balance_problems": 0.35}},
    "infectious": {"age": (28, 62), "fields": {
        "recent_cns_infection": 0.85, "poor_concentration": 0.55, "memory_problems": 0.45,
        "low_mood": 0.35, "balance_problems": 0.35, "slow_gait": 0.30}},
}


def encode_clinical(answers: dict[str, float]) -> np.ndarray:
    """Questionnaire answers -> fixed-length float32 vector (age scaled, others 0/1)."""
    vec = np.zeros(NUM_CLINICAL_FEATURES, dtype=np.float32)
    for i, f in enumerate(CLINICAL_FIELDS):
        raw = answers.get(f.name)
        if raw is None:
            continue
        vec[i] = float(raw) / _AGE_SCALE if f.kind == "age" else (1.0 if float(raw) >= 0.5 else 0.0)
    return vec


def make_clinical(etiology: int, rng: np.random.Generator) -> dict[str, float]:
    """Draw a synthetic questionnaire whose stats depend on the cause index."""
    profile = _ETIOLOGY_PROFILES[ETIOLOGY_CLASS_NAMES[etiology]]
    age_lo, age_hi = profile["age"]                          # type: ignore[misc]
    field_p: dict[str, float] = profile["fields"]            # type: ignore[assignment]
    baseline = float(profile.get("baseline", _BASELINE_P))
    answers: dict[str, float] = {}
    for f in CLINICAL_FIELDS:
        if f.kind == "age":
            answers[f.name] = float(rng.integers(age_lo, age_hi))
        else:
            answers[f.name] = float(rng.random() < field_p.get(f.name, baseline))
    return answers


# ---------------------------------------------------------------------------
# 3. SYNTHETIC MRI GENERATOR (stand-in for real FLAIR MRI)
# ---------------------------------------------------------------------------


def _ellipsoid(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    d, h, w = shape
    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w].astype(np.float32)
    j = rng.uniform(0.95, 1.05, size=3)
    val = (((zz - d / 2) / (d * 0.42 * j[0])) ** 2
           + ((yy - h / 2) / (h * 0.40 * j[1])) ** 2
           + ((xx - w / 2) / (w * 0.38 * j[2])) ** 2)
    return (val <= 1.0).astype(np.float32)


def _add_blob(vol: np.ndarray, center, radius: float, intensity: float) -> None:
    d, h, w = vol.shape
    zz, yy, xx = np.mgrid[0:d, 0:h, 0:w].astype(np.float32)
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    vol += np.exp(-dist2 / (2 * radius ** 2)) * intensity


def _add_lesions(vol: np.ndarray, etiology: str, rng: np.random.Generator) -> None:
    """Add lesions whose spatial pattern loosely reflects the cause.

    Imaging alone is rarely cause-specific, so these patterns overlap on purpose
    — the questionnaire is what mainly pins down the cause after fusion.
    """
    d, h, w = vol.shape

    def band(lo, hi, axis):
        return int(rng.integers(int(vol.shape[axis] * lo), int(vol.shape[axis] * hi)))

    if etiology == "vascular":
        for _ in range(int(rng.integers(3, 7))):
            _add_blob(vol, (band(.35, .65, 0), band(.35, .65, 1), band(.30, .70, 2)),
                      float(rng.uniform(1.5, 2.6)), float(rng.uniform(.45, .65)))
    elif etiology == "autoimmune":
        for _ in range(int(rng.integers(2, 5))):
            _add_blob(vol, (band(.40, .60, 0), band(.42, .58, 1), band(.45, .55, 2)),
                      float(rng.uniform(2.0, 3.2)), float(rng.uniform(.5, .7)))
    elif etiology == "genetic":
        for _ in range(int(rng.integers(1, 3))):
            z, y = band(.40, .60, 0), band(.30, .45, 1)
            for x in (band(.25, .38, 2), w - band(.25, .38, 2)):
                _add_blob(vol, (z, y, x), float(rng.uniform(1.8, 2.8)), float(rng.uniform(.5, .68)))
    elif etiology == "metabolic":
        for _ in range(int(rng.integers(6, 11))):
            _add_blob(vol, (band(.30, .70, 0), band(.30, .70, 1), band(.30, .70, 2)),
                      float(rng.uniform(1.2, 2.0)), float(rng.uniform(.35, .5)))
    elif etiology == "infectious":
        for _ in range(int(rng.integers(1, 4))):
            _add_blob(vol, (band(.30, .70, 0), band(.30, .70, 1), band(.25, .75, 2)),
                      float(rng.uniform(2.6, 4.0)), float(rng.uniform(.5, .72)))


def make_volume(etiology: int, shape, rng: np.random.Generator) -> np.ndarray:
    """One synthetic MRI volume for a cause index (0 = healthy)."""
    brain = _ellipsoid(shape, rng)
    vol = brain * rng.uniform(0.55, 0.7) + brain * rng.normal(0, 0.03, size=shape).astype(np.float32)
    name = ETIOLOGY_CLASS_NAMES[etiology]
    if name != "no_wmd":
        _add_lesions(vol, name, rng)
    return np.clip(vol, 0.0, None).astype(np.float32)


def normalize(vol: np.ndarray) -> np.ndarray:
    """Percentile-clip + scale to [0, 1] (upper bound high so bright lesions survive)."""
    lo, hi = np.percentile(vol, 0.5), np.percentile(vol, 99.9)
    if hi <= lo:
        hi = lo + 1e-5
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def make_dataset(cfg: TrainConfig, n_per_class: int, seed: int):
    """Return (volumes[N,1,D,H,W], clinical[N,F], binary_labels[N], etiology[N])."""
    rng = np.random.default_rng(seed)
    vols, clins, bins, etis = [], [], [], []
    for etiology in range(len(ETIOLOGY_CLASS_NAMES)):
        for _ in range(n_per_class):
            vols.append(normalize(make_volume(etiology, cfg.shape, rng))[None])  # add channel
            clins.append(encode_clinical(make_clinical(etiology, rng)))
            bins.append(0 if etiology == 0 else 1)
            etis.append(etiology)
    return (torch.tensor(np.stack(vols)), torch.tensor(np.stack(clins)),
            torch.tensor(bins), torch.tensor(etis))


# ---------------------------------------------------------------------------
# 4. MODELS — 3D CNN and the multimodal (MRI + questionnaire) fusion model
# ---------------------------------------------------------------------------

IMAGE_EMBED_DIM = 64


class ConvBlock(nn.Module):
    """Conv3d -> GroupNorm -> ReLU -> MaxPool (GroupNorm suits tiny batches)."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, cout), cout),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

    def forward(self, x):
        return self.block(x)


class WMDClassifier3D(nn.Module):
    """Compact 3D CNN for the image-only detection task."""

    def __init__(self, num_classes: int = 2, in_channels: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            ConvBlock(in_channels, 8), ConvBlock(8, 16), ConvBlock(16, 32), ConvBlock(32, 64))
        self.pool = nn.AdaptiveMaxPool3d(1)  # max pool keeps small bright lesions
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 32),
            nn.ReLU(inplace=True), nn.Linear(32, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class MultimodalWMDClassifier(nn.Module):
    """Late-fusion: 3D-CNN image embedding + questionnaire MLP -> shared head."""

    def __init__(self, num_clinical_features: int, num_classes: int = 2,
                 in_channels: int = 1, clinical_embed_dim: int = 16) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_clinical_features = num_clinical_features
        self.features = nn.Sequential(
            ConvBlock(in_channels, 8), ConvBlock(8, 16), ConvBlock(16, 32), ConvBlock(32, 64))
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(num_clinical_features, clinical_embed_dim), nn.ReLU(inplace=True),
            nn.Linear(clinical_embed_dim, clinical_embed_dim), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(IMAGE_EMBED_DIM + clinical_embed_dim, 32),
            nn.ReLU(inplace=True), nn.Linear(32, num_classes))

    def image_embedding(self, volume):
        return torch.flatten(self.pool(self.features(volume)), 1)

    def forward(self, volume, clinical):
        return self.head(torch.cat([self.image_embedding(volume), self.clinical_encoder(clinical)], dim=1))


# ---------------------------------------------------------------------------
# 5. GRAD-CAM — where did the CNN look?
# ---------------------------------------------------------------------------


def grad_cam(model: MultimodalWMDClassifier, volume: torch.Tensor,
             clinical: torch.Tensor, target: int) -> np.ndarray:
    """Return a [0,1] heatmap over the volume for the target class.

    Hooks the last conv block: CAM = ReLU(sum_c mean(grad_c) * activation_c).
    """
    model.eval()
    acts: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    last_conv = model.features[-1].block[0]

    def fwd(_m, _i, out):
        acts["v"] = out.detach()

    def bwd(_m, _gi, gout):
        grads["v"] = gout[0].detach()

    h1 = last_conv.register_forward_hook(fwd)
    h2 = last_conv.register_full_backward_hook(bwd)
    try:
        logits = model(volume, clinical)
        model.zero_grad()
        logits[0, target].backward()
        weights = grads["v"].mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * acts["v"]).sum(dim=1)).squeeze(0).cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam


# ---------------------------------------------------------------------------
# 6. TRAINING + EVALUATION (with confusion matrix)
# ---------------------------------------------------------------------------


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _split(n: int, labels, val_fraction: float, seed: int):
    idx = np.arange(n)
    strat = labels if len(set(labels.tolist())) > 1 else None
    tr, va = train_test_split(idx, test_size=val_fraction, random_state=seed, stratify=strat)
    return tr.tolist(), va.tolist()


def _run_epoch(model, loader, device, optimizer, criterion, multimodal: bool):
    model.train()
    total = 0.0
    for batch in loader:
        if multimodal:
            vol, clin, y = batch
            out = model(vol.to(device), clin.to(device))
        else:
            vol, y = batch
            out = model(vol.to(device))
        y = y.to(device)
        optimizer.zero_grad()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * y.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def _predict_all(model, loader, device, multimodal: bool):
    model.eval()
    labels, preds, probs = [], [], []
    for batch in loader:
        if multimodal:
            vol, clin, y = batch
            p = torch.softmax(model(vol.to(device), clin.to(device)), dim=1).cpu().numpy()
        else:
            vol, y = batch
            p = torch.softmax(model(vol.to(device)), dim=1).cpu().numpy()
        labels.extend(y.numpy().tolist())
        preds.extend(p.argmax(axis=1).tolist())
        probs.extend(p)
    return labels, preds, np.array(probs)


def _report(labels, preds, probs, class_names) -> dict:
    """Confusion matrix + metrics derived from it."""
    n = len(class_names)
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    metrics: dict[str, float] = {"accuracy": float(accuracy_score(labels, preds))}
    if len(set(labels)) > 1:
        if n == 2:
            tn, fp, fn, tp = cm.ravel()
            metrics["roc_auc"] = float(roc_auc_score(labels, probs[:, 1]))
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else 0.0
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
        else:
            metrics["roc_auc_macro"] = float(roc_auc_score(
                labels, probs, multi_class="ovr", average="macro", labels=list(range(n))))
    return {"class_names": list(class_names), "confusion_matrix": cm.tolist(),
            "n_samples": len(labels), "metrics": metrics}


def _train_one(model, train_loader, val_loader, device, cfg, multimodal: bool):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_score, best_state = -1.0, {k: v.clone() for k, v in model.state_dict().items()}
    for epoch in range(1, cfg.epochs + 1):
        loss = _run_epoch(model, train_loader, device, optimizer, criterion, multimodal)
        labels, preds, probs = _predict_all(model, val_loader, device, multimodal)
        rep = _report(labels, preds, probs, [str(i) for i in range(model.num_classes)])
        score = rep["metrics"].get("roc_auc", rep["metrics"].get("roc_auc_macro", rep["metrics"]["accuracy"]))
        print(f"  epoch {epoch:02d} | loss={loss:.4f} | val_acc={rep['metrics']['accuracy']:.3f}")
        if score >= best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_everything(cfg: TrainConfig | None = None, save: bool = True) -> dict:
    """Train both models, evaluate on a held-out test set, return + cache artifacts."""
    cfg = cfg or TrainConfig()
    _set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vols, clins, bins, etis = make_dataset(cfg, cfg.n_per_class, cfg.seed)
    tr, va = _split(len(vols), etis, cfg.val_fraction, cfg.seed)

    # --- image-only detection model ---
    print("== Training image-only detection CNN ==")
    det = WMDClassifier3D(num_classes=len(CLASS_NAMES)).to(device)
    det_train = DataLoader(TensorDataset(vols[tr], bins[tr]), batch_size=cfg.batch_size, shuffle=True)
    det_val = DataLoader(TensorDataset(vols[va], bins[va]), batch_size=cfg.batch_size)
    det = _train_one(det, det_train, det_val, device, cfg, multimodal=False)

    # --- multimodal cause model ---
    print("== Training multimodal cause model ==")
    mm = MultimodalWMDClassifier(NUM_CLINICAL_FEATURES, num_classes=len(ETIOLOGY_CLASS_NAMES)).to(device)
    mm_train = DataLoader(TensorDataset(vols[tr], clins[tr], etis[tr]), batch_size=cfg.batch_size, shuffle=True)
    mm_val = DataLoader(TensorDataset(vols[va], clins[va], etis[va]), batch_size=cfg.batch_size)
    mm = _train_one(mm, mm_train, mm_val, device, cfg, multimodal=True)

    # --- held-out test set (fresh seed => never seen) ---
    print("== Evaluating on held-out test set ==")
    tv, tc, tb, te = make_dataset(cfg, cfg.test_n_per_class, cfg.test_seed)
    det_labels, det_preds, det_probs = _predict_all(
        det, DataLoader(TensorDataset(tv, tb), batch_size=cfg.batch_size), device, False)
    mm_labels, mm_preds, mm_probs = _predict_all(
        mm, DataLoader(TensorDataset(tv, tc, te), batch_size=cfg.batch_size), device, True)
    performance = {
        "note": ("Metrics on a fresh synthetic test set the models never trained on. "
                 "Synthetic data demonstrates the pipeline, not clinical performance."),
        "detection": {"task": "Detection (white matter disease: yes / no)",
                      **_report(det_labels, det_preds, det_probs, CLASS_NAMES)},
        "etiology": {"task": "Cause / etiology (5 causes + healthy)",
                     **_report(mm_labels, mm_preds, mm_probs, ETIOLOGY_CLASS_NAMES)},
    }
    artifacts = {"det_state": det.state_dict(), "mm_state": mm.state_dict(),
                 "performance": performance, "shape": cfg.shape}
    if save:
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifacts, str(ARTIFACT_PATH))
        print(f"Saved artifacts to {ARTIFACT_PATH}")
    _print_confusion(performance)
    return artifacts


def _print_confusion(performance: dict) -> None:
    for key in ("detection", "etiology"):
        rep = performance[key]
        print(f"\n{rep['task']}  (n={rep['n_samples']})")
        print("  metrics:", {k: round(v, 3) for k, v in rep["metrics"].items()})
        names = [n[:9] for n in rep["class_names"]]
        print("  confusion matrix (rows=true, cols=pred):")
        print("      " + " ".join(f"{n:>9}" for n in names))
        for name, row in zip(names, rep["confusion_matrix"]):
            print(f"  {name:>9} " + " ".join(f"{c:>9}" for c in row))


# ---------------------------------------------------------------------------
# 7. INFERENCE — load the trained models and predict on one patient
# ---------------------------------------------------------------------------


class Predictor:
    """Loads cached artifacts and predicts detection + cause for one scan."""

    def __init__(self, artifacts: dict) -> None:
        self.device = torch.device("cpu")
        self.shape = tuple(artifacts["shape"])
        self.performance = artifacts["performance"]
        self.det = WMDClassifier3D(num_classes=len(CLASS_NAMES))
        self.det.load_state_dict(artifacts["det_state"])
        self.det.eval()
        self.mm = MultimodalWMDClassifier(NUM_CLINICAL_FEATURES, num_classes=len(ETIOLOGY_CLASS_NAMES))
        self.mm.load_state_dict(artifacts["mm_state"])
        self.mm.eval()

    @classmethod
    def load(cls) -> "Predictor | None":
        if not ARTIFACT_PATH.exists():
            return None
        return cls(torch.load(str(ARTIFACT_PATH), map_location="cpu", weights_only=False))

    @torch.no_grad()
    def _detect(self, vol_t):
        return torch.softmax(self.det(vol_t), dim=1).cpu().numpy()[0]

    def predict(self, volume: np.ndarray, answers: dict[str, float]) -> dict:
        vol = normalize(volume.astype(np.float32))
        vol_t = torch.tensor(vol[None, None])
        clin_t = torch.tensor(encode_clinical(answers)[None])
        det_prob = self._detect(vol_t)
        with torch.no_grad():
            eti_prob = torch.softmax(self.mm(vol_t, clin_t), dim=1).cpu().numpy()[0]
        eti_idx = int(eti_prob.argmax())
        # Grad-CAM: most salient axial slice for the predicted cause.
        cam = grad_cam(self.mm, vol_t, clin_t, eti_idx)
        salient = int(np.argmax(cam.sum(axis=(1, 2)))) if cam.ndim == 3 else 0
        eti_name = ETIOLOGY_CLASS_NAMES[eti_idx]
        return {
            "wmd_probability": float(det_prob[1]),
            "detected": bool(det_prob[1] >= 0.5),
            "etiology": eti_name,
            "etiology_label": ETIOLOGY_LABELS[eti_name],
            "etiology_confidence": float(eti_prob[eti_idx]),
            "cause_probs": sorted(
                [(ETIOLOGY_LABELS[n], float(p)) for n, p in zip(ETIOLOGY_CLASS_NAMES, eti_prob) if n != "no_wmd"],
                key=lambda t: t[1], reverse=True),
            "next_steps": ETIOLOGY_NEXT_STEPS[eti_name],
            "salient_slice": salient,
        }


def load_volume_from_bytes(name: str, data: bytes, shape) -> np.ndarray:
    """Load an uploaded NIfTI file, or fall back to a random synthetic volume."""
    lname = name.lower()
    if lname.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        suffix = ".nii.gz" if lname.endswith(".nii.gz") else ".nii"
        tmp = Path(f"/tmp/upload_{abs(hash(name))}{suffix}")
        tmp.write_bytes(data)
        try:
            arr = np.asarray(nib.load(str(tmp)).get_fdata(), dtype=np.float32)
        finally:
            tmp.unlink(missing_ok=True)
        # resize to model shape with simple nearest-neighbour sampling
        idx = [np.linspace(0, s - 1, t).astype(int) for s, t in zip(arr.shape, shape)]
        return arr[np.ix_(idx[0], idx[1], idx[2])]
    raise ValueError("unsupported file")


# ---------------------------------------------------------------------------
# 8. WEB APP (FastAPI) — prediction page + Model-Performance page
# ---------------------------------------------------------------------------


def _cell_style(count: int, peak: int, diagonal: bool) -> str:
    intensity = round(count / peak, 3) if peak else 0.0
    if diagonal:
        return f"background: rgba(34,197,94,{0.10 + 0.70 * intensity:.3f})"
    return f"background: rgba(56,189,248,{0.06 + 0.74 * intensity:.3f})"


def _confusion_html(rep: dict) -> str:
    names = [ETIOLOGY_LABELS.get(n, n) for n in rep["class_names"]]
    matrix = rep["confusion_matrix"]
    peak = max((max(r) for r in matrix), default=1) or 1
    head = "".join(f"<th>{n}</th>" for n in names)
    rows = ""
    for i, (name, row) in enumerate(zip(names, matrix)):
        cells = "".join(
            f'<td style="{_cell_style(c, peak, i == j)}" '
            f'title="{name} predicted as {names[j]}: {c}">{c}</td>'
            for j, c in enumerate(row))
        rows += f"<tr><th class='rowlab'>{name}</th>{cells}</tr>"
    metric_map = {"accuracy": "Accuracy", "roc_auc": "ROC-AUC", "roc_auc_macro": "ROC-AUC (macro)",
                  "sensitivity": "Sensitivity", "specificity": "Specificity"}
    cards = "".join(
        f'<div class="card2"><span class="v">{v:.3f}</span>'
        f'<span class="l">{metric_map.get(k, k)}</span></div>'
        for k, v in rep["metrics"].items())
    cards += f'<div class="card2 muted"><span class="v">{rep["n_samples"]}</span><span class="l">Test scans</span></div>'
    return (f"<h3>{rep['task']}</h3><div class='cards'>{cards}</div>"
            f"<div class='cmwrap'><table class='cm'><thead>"
            f"<tr><th class='rowlab'>Actual \\ Predicted</th>{head}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


_STYLE = """
:root{--bg:#0f172a;--card:#1e293b;--text:#e2e8f0;--muted:#94a3b8;--accent:#38bdf8;--border:#334155}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,sans-serif;background:var(--bg);color:var(--text);line-height:1.5}
.site-header{text-align:center;padding:2rem 1rem 1rem}.site-header h1{margin:0;font-size:1.6rem}
.subtitle{color:var(--muted);margin:.25rem 0 0}.nav{margin-top:.85rem;display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap}
.nav a{color:var(--text);text-decoration:none;font-size:.9rem;padding:.35rem .85rem;border:1px solid var(--border);border-radius:999px;background:rgba(148,163,184,.08)}
.nav a:hover{border-color:var(--accent);color:var(--accent)}.container{max-width:820px;margin:0 auto;padding:0 1rem 3rem}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1.5rem}
.muted{color:var(--muted)}.banner{padding:.6rem .9rem;border-radius:8px;margin:.75rem 0}
.banner-info{background:rgba(56,189,248,.12);border:1px solid #0ea5e9}.banner-warn{background:rgba(234,179,8,.15);border:1px solid #ca8a04}
label{display:block;margin:.4rem 0}input[type=number],select{background:#0b1120;color:var(--text);border:1px solid var(--border);border-radius:6px;padding:.35rem .5rem}
button{background:var(--accent);color:#04222f;border:none;border-radius:8px;padding:.6rem 1.2rem;font-weight:700;cursor:pointer;margin-top:1rem}
.cards{display:flex;flex-wrap:wrap;gap:.75rem;margin:1rem 0}
.card2{background:rgba(56,189,248,.10);border:1px solid var(--border);border-radius:10px;padding:.7rem 1.1rem;min-width:7rem;text-align:center}
.card2.muted{background:rgba(148,163,184,.10)}.card2 .v{display:block;font-size:1.5rem;font-weight:700;color:var(--accent)}
.card2.muted .v{color:var(--text)}.card2 .l{display:block;font-size:.8rem;color:var(--muted)}
.cmwrap{overflow-x:auto}table.cm{border-collapse:collapse;font-variant-numeric:tabular-nums}
table.cm th,table.cm td{border:1px solid var(--border);padding:.5rem .7rem;text-align:center}
.rowlab{color:var(--muted);font-size:.8rem;text-align:right;white-space:nowrap}
.bar{height:14px;background:var(--accent);border-radius:4px;display:inline-block}
footer{color:var(--muted);text-align:center;padding:1.5rem;font-size:.85rem}
"""


def _page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>NeuroPredict — {title}</title>
<style>{_STYLE}</style></head><body>
<header class=site-header><h1>NeuroPredict</h1>
<p class=subtitle>Early White Matter Disease risk prediction from MRI — 3D CNN (research demo)</p>
<nav class=nav><a href="/">Predict</a><a href="/performance">Model performance</a></nav></header>
<main class=container>{body}</main>
<footer><strong>Disclaimer:</strong> {RESEARCH_DISCLAIMER}</footer></body></html>"""


def build_app():
    app = FastAPI(title="NeuroPredict (all-in-one)")
    predictor = Predictor.load()

    def _questionnaire_html() -> str:
        rows = ""
        for f in CLINICAL_FIELDS:
            if f.kind == "age":
                rows += f'<label>{f.label} <input type=number name={f.name} value=65 min=0 max=120></label>'
            else:
                rows += f'<label><input type=checkbox name={f.name} value=1> {f.label}</label>'
        return rows

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": predictor is not None}

    @app.get("/", response_class=HTMLResponse)
    def index():
        if predictor is None:
            return _page("Predict", "<div class='card'><div class='banner banner-warn'>"
                         "No model loaded. Run <code>python neuropredict_all_in_one.py train</code>.</div></div>")
        opts = "".join(f'<option value={i}>{ETIOLOGY_LABELS[n]}</option>'
                       for i, n in enumerate(ETIOLOGY_CLASS_NAMES))
        body = f"""<section class=card>
<h2>Predict white matter disease &amp; its cause</h2>
<p class=muted>Upload a brain MRI (.nii/.nii.gz) or let the demo generate a synthetic scan,
answer the questionnaire, and get a probability, the likely cause, and next steps.</p>
<form action="/predict" method=post enctype="multipart/form-data">
<h3>1. MRI scan</h3>
<label>Upload NIfTI (optional): <input type=file name=scan accept=".nii,.nii.gz"></label>
<label>...or generate a synthetic scan of type: <select name=synthetic>{opts}</select></label>
<h3>2. Questionnaire</h3>{_questionnaire_html()}
<button type=submit>Run prediction</button></form></section>"""
        return _page("Predict", body)

    @app.post("/predict", response_class=HTMLResponse)
    async def predict(request: Request):
        form = await request.form()
        answers = {}
        for f in CLINICAL_FIELDS:
            raw = form.get(f.name)
            answers[f.name] = float(raw) if (f.kind == "age" and raw) else (1.0 if raw else 0.0)
        # obtain a volume: uploaded file or synthetic
        volume = None
        scan = form.get("scan")
        if scan is not None and getattr(scan, "filename", ""):
            try:
                volume = load_volume_from_bytes(scan.filename, await scan.read(), predictor.shape)
            except Exception:
                volume = None
        if volume is None:
            synthetic = int(form.get("synthetic") or 0)
            volume = make_volume(synthetic, predictor.shape, np.random.default_rng())
        r = predictor.predict(volume, answers)
        causes = "".join(
            f"<tr><td>{lbl}</td><td><span class=bar style='width:{int(p*200)}px'></span> {p*100:.1f}%</td></tr>"
            for lbl, p in r["cause_probs"])
        steps = "".join(f"<li>{s}</li>" for s in r["next_steps"])
        verdict = "Detected" if r["detected"] else "Not detected"
        body = f"""<section class=card>
<h2>Result</h2>
<div class='banner banner-info'>White matter disease: <strong>{verdict}</strong>
(probability {r['wmd_probability']*100:.1f}%)</div>
<p><strong>Most likely cause:</strong> {r['etiology_label']}
(confidence {r['etiology_confidence']*100:.1f}%)</p>
<p class=muted>Grad-CAM most salient axial slice: #{r['salient_slice']}</p>
<h3>Cause probabilities</h3><table class=cm><tbody>{causes}</tbody></table>
<h3>Suggested next steps</h3><ul>{steps}</ul>
<p><a href="/">&larr; Predict another</a></p></section>"""
        return _page("Result", body)

    @app.get("/performance", response_class=HTMLResponse)
    def performance():
        if predictor is None:
            return _page("Performance", "<div class='card'><div class='banner banner-warn'>"
                         "No model loaded. Run the training command first.</div></div>")
        perf = predictor.performance
        body = (f"<section class=card><h2>Model performance</h2>"
                f"<p class=muted>Measured on a held-out test set the models never trained on. "
                f"Each confusion matrix compares predictions (columns) to the truth (rows): "
                f"the diagonal is correct, off-diagonal is a mistake.</p>"
                f"<div class='banner banner-info'>{perf['note']}</div>"
                f"{_confusion_html(perf['detection'])}{_confusion_html(perf['etiology'])}</section>")
        return _page("Performance", body)

    return app


# FastAPI app object so `uvicorn neuropredict_all_in_one:app` works (e.g. on HF Spaces).
try:  # pragma: no cover - only when fastapi is installed
    app = build_app()
except Exception:  # pragma: no cover
    app = None


# ---------------------------------------------------------------------------
# 9. COMMAND-LINE ENTRYPOINT
# ---------------------------------------------------------------------------


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "auto"
    if cmd == "train":
        train_everything()
        return
    if cmd in ("serve", "auto"):
        if not ARTIFACT_PATH.exists():
            print("No trained artifacts found — training first...")
            train_everything()
        import uvicorn
        print("Serving on http://0.0.0.0:8000  (Ctrl+C to stop)")
        uvicorn.run(build_app(), host="0.0.0.0", port=8000)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
