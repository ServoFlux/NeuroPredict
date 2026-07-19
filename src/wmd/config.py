"""Central configuration for the WMD detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project layout
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_MODEL_PATH = MODELS_DIR / "wmd_cnn.pt"
DEFAULT_MULTIMODAL_MODEL_PATH = MODELS_DIR / "wmd_multimodal.pt"

# Class labels for the image-only binary model (index maps to output logits).
CLASS_NAMES: tuple[str, ...] = ("no_wmd", "early_wmd")

# Class labels for the multimodal etiology model. Index 0 ("no_wmd") is the
# healthy class; the rest are the suspected *cause* of white matter disease.
ETIOLOGY_CLASS_NAMES: tuple[str, ...] = (
    "no_wmd",
    "vascular",
    "autoimmune",
    "genetic",
    "metabolic",
    "infectious",
)

# Human-readable etiology labels for the UI.
ETIOLOGY_LABELS: dict[str, str] = {
    "no_wmd": "No white matter disease",
    "vascular": "Vascular (small-vessel disease)",
    "autoimmune": "Autoimmune (e.g. multiple sclerosis)",
    "genetic": "Genetic (e.g. CADASIL / CARASIL)",
    "metabolic": "Metabolic (e.g. leukodystrophy, B12 deficiency)",
    "infectious": "Infectious (e.g. HIV, Lyme, PML)",
}

# Suggested next steps per prediction. Educational guidance only -- NOT medical
# advice. Always framed around consulting a qualified clinician.
ETIOLOGY_NEXT_STEPS: dict[str, list[str]] = {
    "no_wmd": [
        "No white matter disease was flagged. This is not a diagnosis -- if you have symptoms, still speak with a doctor.",
        "Protect brain health: stay active, eat well, don't smoke, and keep blood pressure, blood sugar, and cholesterol in a healthy range.",
        "Repeat imaging only if a clinician recommends it or new symptoms appear.",
    ],
    "vascular": [
        "Share this result with a primary-care doctor or neurologist.",
        "Ask about checking and controlling blood pressure, diabetes, and cholesterol -- the main drivers of small-vessel disease.",
        "Lifestyle steps help: regular exercise, a heart-healthy diet, and stopping smoking.",
        "A clinician may order follow-up MRI to track changes over time.",
    ],
    "autoimmune": [
        "Ask for a referral to a neurologist to evaluate for an autoimmune cause such as multiple sclerosis.",
        "Further tests may include a contrast MRI of the brain and spine and, sometimes, a lumbar puncture (spinal fluid test).",
        "Bring a record of any episodes of vision changes, numbness, weakness, or balance problems.",
    ],
    "genetic": [
        "Consider genetic counseling to discuss inherited small-vessel diseases (e.g. CADASIL/NOTCH3, CARASIL/HTRA1, COL4A1).",
        "A clinician may recommend genetic testing and screening of close family members.",
        "Manage stroke risk factors (blood pressure, no smoking) while the workup proceeds.",
    ],
    "metabolic": [
        "See a physician about a metabolic workup -- for example vitamin B12, thyroid, and other blood panels.",
        "Mention diet, medications, and any known metabolic conditions so reversible causes can be checked.",
        "Some metabolic causes are treatable, so early evaluation matters.",
    ],
    "infectious": [
        "See a doctor promptly about an infection workup (e.g. HIV, Lyme, or other CNS infections).",
        "Mention recent infections, travel, tick exposure, or fevers.",
        "Many infectious causes are treatable when identified early.",
    ],
}


# Severity indicator bands (research only). Given the model's estimated
# probability of white matter disease, map a *positive* result to a plain-language
# band so the page conveys how pronounced the signal is -- not just yes/no. These
# are confidence-derived bands, NOT a clinical severity grade. Ordered high->low;
# the first band whose threshold is met wins.
SEVERITY_BANDS: tuple[tuple[str, float, str], ...] = (
    ("Severe", 0.85, "The model's white-matter-disease signal is very strong."),
    ("Moderate", 0.70, "The model's white-matter-disease signal is clear."),
    ("Mild", 0.0, "The model's white-matter-disease signal is present but modest."),
)


@dataclass(frozen=True)
class Severity:
    """A research severity band derived from the model's WMD probability."""

    level: str  # "Mild" | "Moderate" | "Severe"
    description: str


def assess_severity(wmd_probability: float) -> Severity:
    """Map a 0-1 white-matter-disease probability to a research severity band.

    This is an interpretability aid, not a clinical grade: it simply buckets how
    confident/pronounced the model's positive signal is.
    """
    for level, threshold, description in SEVERITY_BANDS:
        if wmd_probability >= threshold:
            return Severity(level=level, description=description)
    level, _, description = SEVERITY_BANDS[-1]
    return Severity(level=level, description=description)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for turning a raw scan into a model-ready tensor."""

    # Volume is resampled to this (depth, height, width) before going to the CNN.
    target_shape: tuple[int, int, int] = (64, 64, 64)
    # Intensity normalization clip percentiles. The upper bound is deliberately
    # high (99.9) so that bright white-matter hyperintensities -- the signal of
    # interest -- are preserved rather than clipped away.
    clip_percentiles: tuple[float, float] = (0.5, 99.9)
    # Optional salt-and-pepper (impulse) noise removal. A median filter of this
    # window size is applied before resampling; 0 or 1 disables it. A size of 3
    # (a 3x3x3 neighbourhood) removes lone bright/dark specks -- important for
    # noisy scans and especially the film-digitizer path -- while preserving
    # real lesions.
    denoise_median_size: int = 0
    # Optional cross-scanner harmonization (off by default so the existing
    # trained model keeps working). When True, a lightweight N4-style bias-field
    # correction flattens the smooth brightness shading scanners impose.
    bias_correct: bool = False
    # Intensity normalization mode:
    #   "minmax"      -- robust percentile clip to [0, 1] (default; what the
    #                    shipped model was trained on).
    #   "zscore"      -- mean-0/std-1 within the brain mask.
    #   "whitestripe" -- WhiteStripe: anchor on normal-appearing white matter so
    #                    the same tissue maps to the same value across scanners.
    # "zscore"/"whitestripe" harmonize better across machines but require
    # retraining the model to match the new intensity scale.
    intensity_norm: str = "minmax"


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training the 3D CNN."""

    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    seed: int = 42
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


# Disclaimer surfaced everywhere a prediction is shown.
RESEARCH_DISCLAIMER = (
    "This tool is for research and educational purposes only. It is NOT a "
    "medical device and must NOT be used for diagnosis or clinical "
    "decision-making. Always consult a qualified clinician."
)
