"""Clinical questionnaire features for the multimodal WMD model.

The *cause* (etiology) of white matter disease is often distinguished more by
clinical history and genetics than by the MRI alone. This module defines the
questionnaire schema, encodes answers into a fixed-length feature vector, and
generates synthetic answers whose profile depends on the etiology, so the
multimodal model has a meaningful clinical/genomic signal to fuse with the scan.

Etiologies (see ``ETIOLOGY_CLASS_NAMES`` in config):
  - vascular: small-vessel disease (hypertension, diabetes, age, prior stroke)
  - autoimmune: e.g. multiple sclerosis (younger, autoimmune history)
  - genetic: e.g. CADASIL/CARASIL (NOTCH3/HTRA1/COL4A1, family history)
  - metabolic: e.g. leukodystrophy / B12 deficiency (metabolic disorder, MTHFR)
  - infectious: e.g. HIV / Lyme / PML (CNS infection history)

Research/educational only -- these correlations are illustrative, not clinical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClinicalField:
    """One questionnaire item."""

    name: str
    label: str  # human-readable question for the UI
    kind: str  # "binary" or "age"
    category: str = "History"  # UI grouping


# Order in which categories are rendered in the UI.
CATEGORY_ORDER: tuple[str, ...] = ("Demographics", "History", "Symptoms", "Genomic")

# Order matters: it defines the feature-vector layout and must stay stable.
CLINICAL_FIELDS: tuple[ClinicalField, ...] = (
    ClinicalField("age", "Age (years)", "age", "Demographics"),
    # --- Medical history / risk factors ---
    ClinicalField("hypertension", "High blood pressure (hypertension)", "binary", "History"),
    ClinicalField("diabetes", "Diabetes", "binary", "History"),
    ClinicalField("prior_stroke", "Prior stroke or TIA", "binary", "History"),
    ClinicalField("smoking", "Current or former smoker", "binary", "History"),
    ClinicalField("high_cholesterol", "High cholesterol", "binary", "History"),
    ClinicalField("autoimmune_history", "Diagnosed autoimmune disease (e.g. MS, lupus)", "binary", "History"),
    ClinicalField("recent_cns_infection", "Recent / chronic CNS infection (e.g. HIV, Lyme)", "binary", "History"),
    ClinicalField("metabolic_disorder", "Known metabolic disorder (e.g. B12 deficiency, leukodystrophy)", "binary", "History"),
    # --- Symptoms ---
    ClinicalField("memory_problems", "Memory problems", "binary", "Symptoms"),
    ClinicalField("slow_gait", "Slow walking / gait changes", "binary", "Symptoms"),
    ClinicalField("balance_problems", "Balance problems / falls", "binary", "Symptoms"),
    ClinicalField("poor_concentration", "Reduced concentration / performance", "binary", "Symptoms"),
    ClinicalField("low_mood", "Low mood / depression", "binary", "Symptoms"),
    ClinicalField("urinary_incontinence", "Urinary incontinence", "binary", "Symptoms"),
    # --- Genomic markers ---
    ClinicalField("apoe4_carrier", "APOE \u03b54 carrier", "binary", "Genomic"),
    ClinicalField("notch3_variant", "NOTCH3 pathogenic variant (CADASIL)", "binary", "Genomic"),
    ClinicalField("htra1_variant", "HTRA1 variant (CARASIL / small-vessel disease)", "binary", "Genomic"),
    ClinicalField("col4a1_variant", "COL4A1 / COL4A2 variant", "binary", "Genomic"),
    ClinicalField("mthfr_677tt", "MTHFR C677T homozygous (TT genotype)", "binary", "Genomic"),
    ClinicalField("family_history_stroke", "Family history of stroke / vascular dementia", "binary", "Genomic"),
    ClinicalField("high_wmh_prs", "Elevated white-matter-hyperintensity polygenic risk score", "binary", "Genomic"),
)

NUM_CLINICAL_FEATURES = len(CLINICAL_FIELDS)
CLINICAL_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in CLINICAL_FIELDS)

_AGE_SCALE = 100.0  # normalize age into roughly [0, 1]

# Per-etiology synthetic profile: an integer age range and the prevalence of
# each binary field. Fields not listed fall back to ``_BASELINE_P``.
_BASELINE_P = 0.07

_ETIOLOGY_PROFILES: dict[str, dict[str, object]] = {
    "no_wmd": {
        "age": (40, 66),
        "fields": {},  # everything stays at baseline
        "baseline": 0.04,
    },
    "vascular": {
        "age": (62, 86),
        "fields": {
            "hypertension": 0.80, "diabetes": 0.50, "high_cholesterol": 0.60,
            "smoking": 0.50, "prior_stroke": 0.40, "slow_gait": 0.55,
            "balance_problems": 0.45, "urinary_incontinence": 0.45,
            "memory_problems": 0.45, "poor_concentration": 0.40,
            "high_wmh_prs": 0.50, "family_history_stroke": 0.40,
            "apoe4_carrier": 0.35,
        },
    },
    "autoimmune": {
        "age": (25, 50),
        "fields": {
            "autoimmune_history": 0.85, "balance_problems": 0.55,
            "poor_concentration": 0.55, "low_mood": 0.50,
            "memory_problems": 0.40, "urinary_incontinence": 0.40,
            "slow_gait": 0.35,
        },
    },
    "genetic": {
        "age": (35, 60),
        "fields": {
            "notch3_variant": 0.60, "htra1_variant": 0.30,
            "col4a1_variant": 0.25, "family_history_stroke": 0.80,
            "apoe4_carrier": 0.50, "prior_stroke": 0.45,
            "memory_problems": 0.45, "slow_gait": 0.45,
            "high_wmh_prs": 0.55, "balance_problems": 0.40,
        },
    },
    "metabolic": {
        "age": (30, 65),
        "fields": {
            "metabolic_disorder": 0.85, "diabetes": 0.55, "mthfr_677tt": 0.55,
            "poor_concentration": 0.55, "memory_problems": 0.45,
            "low_mood": 0.35, "balance_problems": 0.35,
        },
    },
    "infectious": {
        "age": (28, 62),
        "fields": {
            "recent_cns_infection": 0.85, "poor_concentration": 0.55,
            "memory_problems": 0.45, "low_mood": 0.35,
            "balance_problems": 0.35, "slow_gait": 0.30,
        },
    },
}


def encode_clinical(answers: dict[str, float]) -> np.ndarray:
    """Encode a questionnaire answer dict into a fixed-length float32 vector.

    Missing fields default to 0 (age 0). Binary fields are coerced to 0/1.
    """
    vec = np.zeros(NUM_CLINICAL_FEATURES, dtype=np.float32)
    for i, field in enumerate(CLINICAL_FIELDS):
        raw = answers.get(field.name)
        if raw is None:
            continue
        if field.kind == "age":
            vec[i] = float(raw) / _AGE_SCALE
        else:
            vec[i] = 1.0 if float(raw) >= 0.5 else 0.0
    return vec


def make_clinical(
    etiology: int, rng: np.random.Generator | None = None
) -> dict[str, float]:
    """Generate synthetic questionnaire answers for an etiology.

    ``etiology`` is an index into ``ETIOLOGY_CLASS_NAMES`` (0 = no_wmd). For
    backward compatibility, 0 yields a healthy profile and 1 the vascular
    profile (the original binary "diseased" case).
    """
    from .config import ETIOLOGY_CLASS_NAMES

    rng = rng or np.random.default_rng()
    name = ETIOLOGY_CLASS_NAMES[etiology]
    profile = _ETIOLOGY_PROFILES[name]
    age_lo, age_hi = profile["age"]  # type: ignore[misc]
    field_p: dict[str, float] = profile["fields"]  # type: ignore[assignment]
    baseline = float(profile.get("baseline", _BASELINE_P))

    answers: dict[str, float] = {}
    for field in CLINICAL_FIELDS:
        if field.kind == "age":
            answers[field.name] = float(rng.integers(age_lo, age_hi))
        else:
            p = field_p.get(field.name, baseline)
            answers[field.name] = float(rng.random() < p)
    return answers
