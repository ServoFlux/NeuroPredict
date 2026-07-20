from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
DEFAULT_MODEL_PATH = MODELS_DIR / 'wmd_cnn.pt'
DEFAULT_MULTIMODAL_MODEL_PATH = MODELS_DIR / 'wmd_multimodal.pt'
CLASS_NAMES: tuple[str, ...] = ('no_wmd', 'early_wmd')
ETIOLOGY_CLASS_NAMES: tuple[str, ...] = ('no_wmd', 'vascular', 'autoimmune', 'genetic', 'metabolic', 'infectious')
ETIOLOGY_LABELS: dict[str, str] = {'no_wmd': 'No white matter disease', 'vascular': 'Vascular (small-vessel disease)', 'autoimmune': 'Autoimmune (e.g. multiple sclerosis)', 'genetic': 'Genetic (e.g. CADASIL / CARASIL)', 'metabolic': 'Metabolic (e.g. leukodystrophy, B12 deficiency)', 'infectious': 'Infectious (e.g. HIV, Lyme, PML)'}
ETIOLOGY_NEXT_STEPS: dict[str, list[str]] = {'no_wmd': ['No white matter disease was flagged. This is not a diagnosis -- if you have symptoms, still speak with a doctor.', "Protect brain health: stay active, eat well, don't smoke, and keep blood pressure, blood sugar, and cholesterol in a healthy range.", 'Repeat imaging only if a clinician recommends it or new symptoms appear.'], 'vascular': ['Share this result with a primary-care doctor or neurologist.', 'Ask about checking and controlling blood pressure, diabetes, and cholesterol -- the main drivers of small-vessel disease.', 'Lifestyle steps help: regular exercise, a heart-healthy diet, and stopping smoking.', 'A clinician may order follow-up MRI to track changes over time.'], 'autoimmune': ['Ask for a referral to a neurologist to evaluate for an autoimmune cause such as multiple sclerosis.', 'Further tests may include a contrast MRI of the brain and spine and, sometimes, a lumbar puncture (spinal fluid test).', 'Bring a record of any episodes of vision changes, numbness, weakness, or balance problems.'], 'genetic': ['Consider genetic counseling to discuss inherited small-vessel diseases (e.g. CADASIL/NOTCH3, CARASIL/HTRA1, COL4A1).', 'A clinician may recommend genetic testing and screening of close family members.', 'Manage stroke risk factors (blood pressure, no smoking) while the workup proceeds.'], 'metabolic': ['See a physician about a metabolic workup -- for example vitamin B12, thyroid, and other blood panels.', 'Mention diet, medications, and any known metabolic conditions so reversible causes can be checked.', 'Some metabolic causes are treatable, so early evaluation matters.'], 'infectious': ['See a doctor promptly about an infection workup (e.g. HIV, Lyme, or other CNS infections).', 'Mention recent infections, travel, tick exposure, or fevers.', 'Many infectious causes are treatable when identified early.']}
SEVERITY_BANDS: tuple[tuple[str, float, str], ...] = (('Severe', 0.85, "The model's white-matter-disease signal is very strong."), ('Moderate', 0.7, "The model's white-matter-disease signal is clear."), ('Mild', 0.0, "The model's white-matter-disease signal is present but modest."))
@dataclass(frozen=True)
class Severity:
    level: str
    description: str
def assess_severity(wmd_probability: float) -> Severity:
    for level, threshold, description in SEVERITY_BANDS:
        if wmd_probability >= threshold:
            return Severity(level=level, description=description)
    level, _, description = SEVERITY_BANDS[-1]
    return Severity(level=level, description=description)
@dataclass(frozen=True)
class PreprocessConfig:
    target_shape: tuple[int, int, int] = (64, 64, 64)
    clip_percentiles: tuple[float, float] = (0.5, 99.9)
    denoise_median_size: int = 0
    bias_correct: bool = False
    intensity_norm: str = 'minmax'
@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    val_fraction: float = 0.2
    seed: int = 42
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
RESEARCH_DISCLAIMER = 'This tool is for research and educational purposes only. It is NOT a medical device and must NOT be used for diagnosis or clinical decision-making. Always consult a qualified clinician.'
