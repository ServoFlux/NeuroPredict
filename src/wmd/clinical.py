from __future__ import annotations
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class ClinicalField:
    name: str
    label: str
    kind: str
    category: str = 'History'
    help: str = ''
CATEGORY_ORDER: tuple[str, ...] = ('Demographics', 'History', 'Symptoms', 'Genomic')
CLINICAL_FIELDS: tuple[ClinicalField, ...] = (ClinicalField('age', 'Age (years)', 'age', 'Demographics', help="The patient's age in years. White matter disease becomes more common with age, so this is one of the strongest clues."), ClinicalField('hypertension', 'High blood pressure (hypertension)', 'binary', 'History', help="Has a doctor said the patient has high blood pressure, or are they on blood-pressure medication? High blood pressure slowly damages the brain's small blood vessels."), ClinicalField('diabetes', 'Diabetes', 'binary', 'History', help='Has the patient been diagnosed with diabetes (high blood sugar)? Diabetes can damage small blood vessels, including those in the brain.'), ClinicalField('prior_stroke', 'Prior stroke or TIA', 'binary', 'History', help="Has the patient ever had a stroke or a 'mini-stroke' (TIA) — sudden weakness, slurred speech, or vision loss, even if it went away within a day?"), ClinicalField('smoking', 'Current or former smoker', 'binary', 'History', help='Does the patient smoke now, or did they smoke in the past? Smoking harms blood vessels and raises stroke risk.'), ClinicalField('high_cholesterol', 'High cholesterol', 'binary', 'History', help='Has a doctor said the patient has high cholesterol, or are they on a cholesterol medication (e.g. a statin)? High cholesterol can clog blood vessels.'), ClinicalField('autoimmune_history', 'Diagnosed autoimmune disease (e.g. MS, lupus)', 'binary', 'History', help="An autoimmune disease is when the immune system attacks the body's own tissue. Multiple sclerosis (MS) and lupus are examples that can affect the brain's white matter."), ClinicalField('recent_cns_infection', 'Recent / chronic CNS infection (e.g. HIV, Lyme)', 'binary', 'History', help='Has the patient had an infection that affects the brain or nervous system — for example HIV, Lyme disease, or a serious brain/spinal infection?'), ClinicalField('metabolic_disorder', 'Known metabolic disorder (e.g. B12 deficiency, leukodystrophy)', 'binary', 'History', help='A metabolic disorder is a problem with how the body processes nutrients or chemicals — e.g. severe vitamin B12 deficiency, or an inherited condition called leukodystrophy.'), ClinicalField('memory_problems', 'Memory problems', 'binary', 'Symptoms', help='Does the patient have trouble remembering recent events, names, or appointments — more than normal forgetfulness?'), ClinicalField('slow_gait', 'Slow walking / gait changes', 'binary', 'Symptoms', help="Has the patient's walking become slower, shorter-stepped, or shuffling? White matter disease can affect the brain's movement pathways."), ClinicalField('balance_problems', 'Balance problems / falls', 'binary', 'Symptoms', help='Does the patient feel unsteady, lose their balance, or fall more often than before?'), ClinicalField('poor_concentration', 'Reduced concentration / performance', 'binary', 'Symptoms', help='Does the patient have trouble focusing, thinking clearly, or keeping up with tasks at work or school?'), ClinicalField('low_mood', 'Low mood / depression', 'binary', 'Symptoms', help='Has the patient felt persistently sad, down, or depressed? Mood changes can accompany white matter disease.'), ClinicalField('urinary_incontinence', 'Urinary incontinence', 'binary', 'Symptoms', help='Does the patient have trouble controlling their bladder (leaking urine or sudden urges)? This can be a sign when brain pathways are affected.'), ClinicalField('apoe4_carrier', 'APOE ε4 carrier', 'binary', 'Genomic', help="A version of the APOE gene that raises the risk of Alzheimer's and vascular brain disease. You only know this from a DNA/genetic test — leave it unchecked if the patient has never had one."), ClinicalField('notch3_variant', 'NOTCH3 pathogenic variant (CADASIL)', 'binary', 'Genomic', help="A change in the NOTCH3 gene that causes CADASIL, an inherited disease of the brain's small blood vessels. Known only from genetic testing — leave unchecked if untested."), ClinicalField('htra1_variant', 'HTRA1 variant (CARASIL / small-vessel disease)', 'binary', 'Genomic', help='A change in the HTRA1 gene linked to an inherited small-vessel brain disease (CARASIL). Known only from genetic testing — leave unchecked if untested.'), ClinicalField('col4a1_variant', 'COL4A1 / COL4A2 variant', 'binary', 'Genomic', help='A gene change that weakens small blood vessel walls in the brain, raising the risk of bleeds and white matter damage. Known only from genetic testing — leave unchecked if untested.'), ClinicalField('mthfr_677tt', 'MTHFR C677T homozygous (TT genotype)', 'binary', 'Genomic', help="A common gene variant (the 'TT' version) that can raise homocysteine, a chemical weakly linked to vascular risk. Known only from genetic testing — leave unchecked if untested."), ClinicalField('family_history_stroke', 'Family history of stroke / vascular dementia', 'binary', 'Genomic', help="Did a close blood relative (parent, sibling) have a stroke or vascular dementia? This hints at an inherited risk and doesn't need a genetic test."), ClinicalField('high_wmh_prs', 'Elevated white-matter-hyperintensity polygenic risk score', 'binary', 'Genomic', help="A 'polygenic risk score' adds up many tiny genetic effects into one risk number; a high score means inherited risk for white matter disease. Comes from a genetic analysis — leave unchecked if untested."))
NUM_CLINICAL_FEATURES = len(CLINICAL_FIELDS)
CLINICAL_FIELD_NAMES: tuple[str, ...] = tuple((f.name for f in CLINICAL_FIELDS))
_AGE_SCALE = 100.0
_BASELINE_P = 0.07
_ETIOLOGY_PROFILES: dict[str, dict[str, object]] = {'no_wmd': {'age': (40, 66), 'fields': {}, 'baseline': 0.04}, 'vascular': {'age': (62, 86), 'fields': {'hypertension': 0.8, 'diabetes': 0.5, 'high_cholesterol': 0.6, 'smoking': 0.5, 'prior_stroke': 0.4, 'slow_gait': 0.55, 'balance_problems': 0.45, 'urinary_incontinence': 0.45, 'memory_problems': 0.45, 'poor_concentration': 0.4, 'high_wmh_prs': 0.5, 'family_history_stroke': 0.4, 'apoe4_carrier': 0.35}}, 'autoimmune': {'age': (25, 50), 'fields': {'autoimmune_history': 0.85, 'balance_problems': 0.55, 'poor_concentration': 0.55, 'low_mood': 0.5, 'memory_problems': 0.4, 'urinary_incontinence': 0.4, 'slow_gait': 0.35}}, 'genetic': {'age': (35, 60), 'fields': {'notch3_variant': 0.6, 'htra1_variant': 0.3, 'col4a1_variant': 0.25, 'family_history_stroke': 0.8, 'apoe4_carrier': 0.5, 'prior_stroke': 0.45, 'memory_problems': 0.45, 'slow_gait': 0.45, 'high_wmh_prs': 0.55, 'balance_problems': 0.4}}, 'metabolic': {'age': (30, 65), 'fields': {'metabolic_disorder': 0.85, 'diabetes': 0.55, 'mthfr_677tt': 0.55, 'poor_concentration': 0.55, 'memory_problems': 0.45, 'low_mood': 0.35, 'balance_problems': 0.35}}, 'infectious': {'age': (28, 62), 'fields': {'recent_cns_infection': 0.85, 'poor_concentration': 0.55, 'memory_problems': 0.45, 'low_mood': 0.35, 'balance_problems': 0.35, 'slow_gait': 0.3}}}
def encode_clinical(answers: dict[str, float]) -> np.ndarray:
    vec = np.zeros(NUM_CLINICAL_FEATURES, dtype=np.float32)
    for i, field in enumerate(CLINICAL_FIELDS):
        raw = answers.get(field.name)
        if raw is None:
            continue
        if field.kind == 'age':
            vec[i] = float(raw) / _AGE_SCALE
        else:
            vec[i] = 1.0 if float(raw) >= 0.5 else 0.0
    return vec
def make_clinical(etiology: int, rng: np.random.Generator | None=None) -> dict[str, float]:
    from .config import ETIOLOGY_CLASS_NAMES
    rng = rng or np.random.default_rng()
    name = ETIOLOGY_CLASS_NAMES[etiology]
    profile = _ETIOLOGY_PROFILES[name]
    age_lo, age_hi = profile['age']
    field_p: dict[str, float] = profile['fields']
    baseline = float(profile.get('baseline', _BASELINE_P))
    answers: dict[str, float] = {}
    for field in CLINICAL_FIELDS:
        if field.kind == 'age':
            answers[field.name] = float(rng.integers(age_lo, age_hi))
        else:
            p = field_p.get(field.name, baseline)
            answers[field.name] = float(rng.random() < p)
    return answers
