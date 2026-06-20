"""Vascular-risk scoring and fusion with the MRI prediction.

White matter disease is strongly associated with cerebrovascular risk factors
(hypertension, low oxygenation, abnormal heart rate, age). The IoT companion
device streams these vitals; here we turn them into a transparent, rule-based
vascular-risk score and fuse it with the CNN's MRI probability.

This is intentionally a simple, explainable scoring rule -- NOT a trained model
and NOT a clinical risk calculator. It exists to demonstrate combining an
imaging model with live sensor data for research/educational purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Vitals:
    """A single set of readings from the IoT device."""

    heart_rate: float | None = None  # beats per minute
    spo2: float | None = None  # blood-oxygen saturation, %
    systolic: float | None = None  # blood pressure, mmHg
    diastolic: float | None = None  # blood pressure, mmHg
    age: float | None = None  # years


@dataclass
class VascularRisk:
    """Result of scoring a set of vitals."""

    score: float  # 0..1, higher = more vascular risk
    level: str  # "low" | "moderate" | "high"
    factors: list[str] = field(default_factory=list)  # human-readable contributors


def _level_from_score(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.3:
        return "moderate"
    return "low"


def vascular_risk_score(vitals: Vitals) -> VascularRisk:
    """Score vitals into a 0..1 vascular-risk value with explanatory factors.

    Each abnormal reading contributes points; the total is normalized by the
    maximum achievable so the result is always in [0, 1].
    """
    points = 0.0
    max_points = 0.0
    factors: list[str] = []

    # Hypertension (systolic) — strongest single contributor.
    if vitals.systolic is not None:
        max_points += 3.0
        if vitals.systolic >= 160:
            points += 3.0
            factors.append(f"severe hypertension (systolic {vitals.systolic:.0f})")
        elif vitals.systolic >= 140:
            points += 2.0
            factors.append(f"stage-2 hypertension (systolic {vitals.systolic:.0f})")
        elif vitals.systolic >= 130:
            points += 1.0
            factors.append(f"elevated blood pressure (systolic {vitals.systolic:.0f})")

    # Diastolic hypertension.
    if vitals.diastolic is not None:
        max_points += 2.0
        if vitals.diastolic >= 100:
            points += 2.0
            factors.append(f"high diastolic ({vitals.diastolic:.0f})")
        elif vitals.diastolic >= 90:
            points += 1.0
            factors.append(f"elevated diastolic ({vitals.diastolic:.0f})")

    # Low blood-oxygen saturation.
    if vitals.spo2 is not None:
        max_points += 2.0
        if vitals.spo2 < 90:
            points += 2.0
            factors.append(f"hypoxemia (SpO2 {vitals.spo2:.0f}%)")
        elif vitals.spo2 < 94:
            points += 1.0
            factors.append(f"low SpO2 ({vitals.spo2:.0f}%)")

    # Abnormal heart rate (either direction).
    if vitals.heart_rate is not None:
        max_points += 1.0
        if vitals.heart_rate > 100:
            points += 1.0
            factors.append(f"tachycardia (HR {vitals.heart_rate:.0f})")
        elif vitals.heart_rate < 50:
            points += 1.0
            factors.append(f"bradycardia (HR {vitals.heart_rate:.0f})")

    # Age — a non-modifiable but important risk factor.
    if vitals.age is not None:
        max_points += 2.0
        if vitals.age >= 75:
            points += 2.0
            factors.append(f"advanced age ({vitals.age:.0f})")
        elif vitals.age >= 60:
            points += 1.0
            factors.append(f"older age ({vitals.age:.0f})")

    score = (points / max_points) if max_points > 0 else 0.0
    if not factors:
        factors.append("no abnormal vascular risk factors detected")
    return VascularRisk(score=score, level=_level_from_score(score), factors=factors)


@dataclass
class CombinedRisk:
    """Fusion of the MRI probability and the vascular-risk score."""

    score: float  # 0..1
    level: str  # "low" | "moderate" | "high"
    mri_probability: float
    vascular_score: float


def combined_risk(
    mri_probability: float,
    vascular: VascularRisk,
    mri_weight: float = 0.7,
) -> CombinedRisk:
    """Weighted fusion of the MRI WMD probability and vascular-risk score.

    The MRI model carries most of the weight (it directly observes the brain);
    the vitals provide supporting context. Returns a combined 0..1 risk.
    """
    mri_weight = min(max(mri_weight, 0.0), 1.0)
    score = mri_weight * mri_probability + (1.0 - mri_weight) * vascular.score
    return CombinedRisk(
        score=score,
        level=_level_from_score(score),
        mri_probability=mri_probability,
        vascular_score=vascular.score,
    )
