"""Tests for the vascular-risk scoring and fusion."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.risk import Vitals, combined_risk, vascular_risk_score  # noqa: E402


def test_healthy_vitals_low_risk() -> None:
    risk = vascular_risk_score(
        Vitals(heart_rate=70, spo2=98, systolic=118, diastolic=76, age=40)
    )
    assert risk.score == 0.0
    assert risk.level == "low"


def test_abnormal_vitals_high_risk() -> None:
    risk = vascular_risk_score(
        Vitals(heart_rate=105, spo2=88, systolic=165, diastolic=102, age=78)
    )
    assert risk.score > 0.6
    assert risk.level == "high"
    assert any("hypertension" in f for f in risk.factors)


def test_score_always_in_unit_range() -> None:
    risk = vascular_risk_score(Vitals(systolic=300, diastolic=200, spo2=50, heart_rate=200, age=120))
    assert 0.0 <= risk.score <= 1.0


def test_partial_vitals_only_score_present_fields() -> None:
    risk = vascular_risk_score(Vitals(spo2=92))
    assert 0.0 < risk.score <= 1.0  # one abnormal field out of one present


def test_combined_risk_weights_mri() -> None:
    low = vascular_risk_score(Vitals(heart_rate=70, spo2=98, systolic=115))
    fused = combined_risk(0.9, low, mri_weight=0.7)
    # 0.7*0.9 + 0.3*0.0 = 0.63
    assert abs(fused.score - 0.63) < 1e-6
    assert fused.level == "high"
    assert fused.mri_probability == 0.9
