from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.config import assess_severity

def test_severity_buckets() -> None:
    assert assess_severity(0.90).level == "Severe"
    assert assess_severity(0.85).level == "Severe"
    assert assess_severity(0.75).level == "Moderate"
    assert assess_severity(0.70).level == "Moderate"
    assert assess_severity(0.55).level == "Mild"
    assert assess_severity(0.0).level == "Mild"

def test_severity_is_monotonic() -> None:
    order = {"Mild": 0, "Moderate": 1, "Severe": 2}
    levels = [order[assess_severity(p / 100).level] for p in range(0, 101)]
    assert levels == sorted(levels)

def test_severity_has_description() -> None:
    assert assess_severity(0.9).description
