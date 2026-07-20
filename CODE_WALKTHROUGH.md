# NeuroPredict — Full Code Walkthrough (judge-ready, plain English)

This document walks through **every software file** in NeuroPredict in plain
language, with the *why* behind each design choice and the honest limitations.
If you can explain this document, you can answer almost any judge question.

> **Scope note:** This covers the software (the model, the prediction code, the
> questionnaire, the film digitizer *algorithm*, the website, training, data, and
> tests). The IoT hardware files (ESP32/XIAO firmware and the device simulator)
> are intentionally left out for now — they're set aside until the hardware is
> bought. The server-side digitizer code (`filmscan.py`) that the *website* uses
> is included, because that's software the site runs.

---

## The big picture in one breath

A person uploads a brain MRI and answers a short questionnaire. The MRI goes
through a 3D AI "eye" (a convolutional neural network) that looks for
white-matter damage. The questionnaire goes through a small second network. The
two are combined to predict whether disease is present and its likely *cause*.
The app also shows a heatmap of where the AI looked, how much the scan vs. the
answers mattered, and suggested next steps. There's also a "digitizer" that turns
a photo of an old MRI film sheet into a scan the AI can read.

### How the files fit together (the data's journey)

```
                       (training, done once)
  synthetic.py  ──>  generate_dataset()  ──>  data/synthetic/*.nii.gz + manifest.csv
       │                                              │
       │ (fake brains + fake patients)                ▼
  clinical.py ─────────────────────────────►   dataset.py (reads manifest)
                                                      │
                                                      ▼
                                                  train.py  ──>  models/*.pt  (saved "brain")

                       (live prediction, every request)
  webapp/main.py  ──(upload)──►  preprocessing.py ─┐
       │                          clinical.py ─────┤
       │                                           ▼
       │                                    inference.py ──uses──► model.py
       │                                           │        └────► explain.py (Grad-CAM)
       │   (film photo path)                       ▼
       └──► filmscan.py ──rebuild volume──►  (same inference pipeline)
                                                   │
                                                   ▼
                              templates/*.html + static/style.css  (the result page)
```

Read it as: **the bottom-left files build the model once; the right-hand files
use it on every request.** `config.py` sits in the middle holding the settings
that everyone shares.

---

## Key ideas you should be comfortable saying out loud

- **Neural network:** a math function with thousands/millions of adjustable
  numbers ("weights") that learns patterns from examples.
- **3D CNN (Convolutional Neural Network):** an AI that scans an image with small
  sliding filters to detect shapes. "3D" because an MRI is a *stack* of slices (a
  cube of data), not a flat photo — so the filters slide in 3D.
- **Logits / softmax / probabilities:** the model outputs raw scores (logits);
  `softmax` turns them into percentages that add up to 100%.
- **Multimodal / fusion:** combining two different *kinds* of input — the image
  and the questionnaire — into one prediction.
- **Etiology:** the medical word for "the cause" of a disease.
- **Grad-CAM:** a method that highlights which parts of the image most influenced
  the decision (explainability — not a black box).
- **Tensor:** just a multi-dimensional array of numbers (the format PyTorch uses).
- **Checkpoint (`.pt` file):** the saved model — its learned weights plus the
  settings used to train it.
- **Synthetic data:** computer-generated fake patients we use because we don't
  have real labeled data. **This is the single most important honest caveat.**

---

# PART A — The "shared settings" file

## `src/wmd/config.py` — the project's control panel

Every other file imports its settings from here, so there's one source of truth.

- **`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`** (lines 9–11) — figure out the
  folder layout automatically from this file's location, so the code works on any
  computer without hard-coded paths.
- **`DEFAULT_MODEL_PATH`, `DEFAULT_MULTIMODAL_MODEL_PATH`** (13–14) — where the
  two trained models are saved (`models/wmd_cnn.pt` and `wmd_multimodal.pt`).
- **`CLASS_NAMES`** (17) — the two labels for the simple image-only model:
  `("no_wmd", "early_wmd")`. The *position* in this tuple is the model's output
  index (0 = healthy, 1 = disease).
- **`ETIOLOGY_CLASS_NAMES`** (21–28) — the six labels for the real multimodal
  model: healthy + five causes (`vascular, autoimmune, genetic, metabolic,
  infectious`). Index 0 is always "healthy."
- **`ETIOLOGY_LABELS`** (31–38) — friendly text for the screen (e.g. `genetic` →
  "Genetic (e.g. CADASIL / CARASIL)").
- **`ETIOLOGY_NEXT_STEPS`** (42–74) — the educational "what to do next" bullet
  list for each cause. *Every* entry is framed as "talk to a clinician" — never a
  diagnosis. This is a deliberate safety/ethics choice you can point to.
- **`SEVERITY_BANDS` + `assess_severity`** — turn a *positive* result into a
  plain-language band (**Mild / Moderate / Severe**) so the page shows *how
  pronounced* the signal is, not just yes/no. It buckets the model's estimated
  white-matter-disease probability: ≥85% → Severe, ≥70% → Moderate, else Mild.
  Crucially this is a **research/confidence indicator, NOT a clinical severity
  grade** — the UI says so explicitly. Keeping the thresholds here in one place
  means they're easy to justify to a judge and easy to tune.
- **`PreprocessConfig`** (77–86) — how a raw scan is standardized:
  - `target_shape = (64, 64, 64)` — every scan is resized to a 64×64×64 cube so
    the network always sees the same size.
  - `clip_percentiles = (0.5, 99.9)` — when normalizing brightness, ignore the
    darkest 0.5% and brightest 0.1% of pixels. *Why the high upper bound (99.9)?*
    Because white-matter lesions are **bright** — we deliberately keep the bright
    end so the signal of interest isn't clipped away. (Good judge detail.)
  - `bias_correct = False` and `intensity_norm = "minmax"` — optional
    cross-scanner **harmonization** knobs (see the harmonization section below).
    Defaults keep the shipped model's behavior; set them to harmonize scans from
    different machines.
- **`TrainConfig`** (89–99) — training settings: 15 epochs, batch size 8,
  learning rate 0.001, a 20% validation split, and seed 42 (so runs are
  reproducible). `@dataclass(frozen=True)` means these are read-only once created.
- **`RESEARCH_DISCLAIMER`** (103–107) — the "not a medical device" text shown on
  every page. Honesty, front and center.

**Judge point:** centralizing all labels, paths, and settings here means the
website, training, and inference can never disagree about what "class 3" means.

---

# PART B — The AI itself

## `src/wmd/model.py` — the AI model (the brain's architecture)

This file defines the *shape* of the AI. It doesn't train it; it builds the empty
network that training fills in. Built entirely from PyTorch's `nn` building blocks.

### `ConvBlock` (lines 9–27) — one reusable "vision" unit
```python
nn.Conv3d(in, out, kernel_size=3, padding=1, bias=False)
nn.GroupNorm(num_groups, out)
nn.ReLU(inplace=True)
nn.MaxPool3d(kernel_size=2)
```
In order, it:
- **`Conv3d`** — slides 3×3×3 filters across the 3D scan to detect local patterns
  (edges, blobs, bright spots). `out_channels` = how many different patterns this
  layer learns. `padding=1` keeps the size the same so *we* control shrinking with
  pooling.
- **`GroupNorm`** — re-centers/re-scales the numbers so training is stable.
  *Why GroupNorm and not the more common BatchNorm?* BatchNorm needs large batches
  to behave well; 3D MRIs are big, so we train with **small batches** on a CPU.
  GroupNorm doesn't depend on batch size, so it behaves identically during
  training and prediction. (Strong judge answer — shows you understand the
  tradeoff.)
- **`ReLU`** — keeps positive signals, zeroes out negatives. This "non-linearity"
  is what lets the network learn complex shapes, not just straight lines.
- **`MaxPool3d(2)`** — halves the cube in each direction, keeping the strongest
  response in each little neighborhood. Faster, and focuses on prominent features.

### `WMDClassifier3D` (lines 30–67) — the image-only model
- **`self.features`** — four ConvBlocks stacked (8 → 16 → 32 → 64 channels). Early
  blocks learn simple things (edges); later blocks learn complex things
  (lesion-like shapes). This is the "deep" in deep learning.
- **`AdaptiveMaxPool3d(1)`** (line 49) — squashes the whole cube to one number per
  channel (64 numbers). *Why MAX and not average?* White-matter lesions are
  **small bright focal spots**; averaging would dilute a tiny bright spot into the
  surrounding darkness, while **max** keeps it. A deliberate, defensible choice for
  this disease. (Strong judge point.)
- **`classifier`** (50–56) — turns those 64 numbers into the final scores.
  `Dropout(0.3)` randomly ignores 30% of connections *during training only* to
  prevent memorizing (overfitting). `Linear` layers make the decision.
- **`embed`** (58–62) — returns just the 64-number image "summary," used by the
  multimodal model.

### `MultimodalWMDClassifier` (lines 73–122) — the real star (image + questionnaire)
This is the model the website uses. It does **late fusion**: encode each input
separately, then combine.
- The **image branch** (`features` + `pool`) turns the MRI into 64 numbers.
- The **clinical branch** (`clinical_encoder`, 101–106) turns the 22 questionnaire
  answers into 16 numbers via a tiny 2-layer network.
- **`forward`** (118–122):
  ```python
  img  = self.image_embedding(volume)     # 64 numbers from the scan
  clin = self.clinical_encoder(clinical)  # 16 numbers from the answers
  fused = torch.cat([img, clin], dim=1)   # glue them: 80 numbers
  return self.head(fused)                  # one combined decision
  ```
  Gluing the two summaries and sending them through one shared `head` is what lets
  the scan **and** the history jointly decide the cause.
- *Why late fusion instead of early fusion?* Image and questionnaire are
  completely different data types and sizes. Encoding each in its own branch and
  then combining compact summaries trains better on little data and — key for us —
  lets us **turn one input off** to measure how much each mattered (see
  `inference.py` attribution).

### Builder functions (lines 125–134)
`build_model` and `build_multimodal_model` are small helpers that create the
networks with the right number of classes.

**Honest limitation to volunteer:** the network is intentionally *small* so it
trains on a laptop CPU for a demo. A clinical model would be larger and trained on
real labeled MRIs.

---

## `src/wmd/explain.py` — Grad-CAM (the "where did the AI look" heatmap)

This is the explainability engine. Instead of only returning a label, it shows
*which regions of the brain* drove the prediction.

### `grad_cam` (lines 16–76)
The idea: run the scan through the model, then ask "if I wanted *more* of the
predicted class, which feature-map regions would I turn up?" Those regions are the
heatmap.
- **Hooks** (41–50) — temporarily attach listeners to the last conv layer to grab
  its outputs (`activations`) and their gradients (`gradients`) during a backward
  pass.
- **The math** (52–59): run forward, call `.backward()` on the predicted class
  score, then `weights = grads.mean(...)` measures how important each of the 64
  feature channels is. `relu(weighted sum)` keeps only regions that *support* the
  class. Result: a small 3D importance map.
- **Upsample** (64–69) — stretch that small map back to the input size so it lines
  up with the brain.
- **Normalize** (71–76) — scale to 0–1 for display.
- The `try/finally` (51–62) guarantees the hooks are always removed afterward, so
  there's no memory leak.

### `heatmap_rgb` (79–85) and `overlay_cam_on_slice` (88–97)
Turn the 0–1 importance values into a blue→green→red "jet" color map and blend it
over the grayscale brain slice (red = most influential).

### `most_salient_axial_index` (100–102)
Picks the single slice with the strongest response, so the UI shows the most
informative view.

**Judge phrasing:** "It's not a black box — Grad-CAM back-projects the decision
onto the actual brain regions that moved it, and we show that heatmap next to the
input."

---

# PART C — The data pipeline (how scans get prepared)

## `src/wmd/preprocessing.py` — clean & standardize a scan

Real scans come in different sizes, formats, and brightness ranges. This file
makes them uniform before the AI sees them.

- **`load_volume`** (26–49) — reads a scan from disk. Handles NIfTI (`.nii`,
  `.nii.gz`) and DICOM (single file or a folder of slices). If it can't tell the
  type, it tries both as a fallback.
- **`_load_nifti`** (52–63) — loads a NIfTI; if it's a 4D series (e.g. a time
  series) it takes the first volume; errors clearly if it isn't 3D.
- **`_load_dicom_series`** (66–95) — reads a folder of DICOM slices and, crucially,
  **sorts them into the correct anatomical order** using each slice's position
  (`ImagePositionPatient`), falling back to `InstanceNumber`. Out-of-order slices
  would scramble the brain.
- **`normalize_intensity`** (98–113) — clips to the configured percentiles and
  rescales to roughly 0–1. Guards against a flat/blank scan (avoids divide-by-zero).
- **`resample_to_shape`** (116–124) — trilinearly resizes any volume to the fixed
  64×64×64 cube (`F.interpolate`). Trilinear = smooth 3D interpolation.
- **`median_filter_3d`** — removes **salt-and-pepper noise** (see below), an
  optional denoising step controlled by `PreprocessConfig.denoise_median_size`.
- **`preprocess_volume`** — the full chain: *(optional harmonize)* → normalize →
  *(optional denoise)* → resample → add a channel dimension, returning a
  `(1, D, H, W)` tensor.
- **`load_and_preprocess`** — convenience: do both load and preprocess.

### Cross-scanner harmonization (`src/wmd/harmonization.py`)

**The problem (Dr. Tohka's point):** archival scans come from different machines
(1.5T vs 3T; Philips vs Siemens vs GE) with different shading and intensity
scales. If we don't correct for that, the CNN can learn *"which scanner"* instead
of *"is there disease"* — a hidden bias that inflates scores and fails on new
scanners. Harmonization makes the same tissue look the same everywhere first.

Three per-scan, dependency-free steps (NumPy/PyTorch only — honest
approximations of the standard ANTs N4 / WhiteStripe tools):

- **`otsu_brain_mask`** — separates brain from background with Otsu's threshold,
  so the other steps use brain voxels only (not air).
- **`bias_field_correct`** — an **N4-style bias-field correction**. Scanners
  impose a smooth multiplicative brightness gradient (surface coils are brighter
  nearby). We estimate that slow field as a heavily blurred version of the image
  (in the log domain), then divide it out so one tissue has a consistent
  intensity across the whole brain. Enable with `--bias-correct`.
- **`zscore_normalize` / `white_stripe_normalize`** — intensity normalization so
  the *same tissue maps to the same number* across scanners. Z-score uses the
  whole brain; **WhiteStripe** anchors on normal-appearing white matter (a tissue
  that should look identical across machines), making it scale-invariant to a
  scanner's raw units. Select with `--intensity-norm zscore|whitestripe`.

These are **opt-in** (defaults are unchanged: `minmax` normalization, no bias
correction), so the shipped model keeps working. A harmonizing mode changes the
intensity scale, so a model must be *trained and served with the same mode*.

**Honest limits:** these are lightweight approximations, not validated clinical
harmonization. True **ComBat** — the gold-standard for removing site effects —
is a *cohort-level* statistical method: it needs a batch of scans with known
site labels to estimate and remove each scanner's effect, so it can't run on a
single live upload. It's the natural next step once we have multi-site scans
(e.g. from Dr. Tohka). Isotropic mm-spacing resampling is another future add.

**One-line summary for judges:** *"Before the AI sees a scan, we flatten the
scanner's brightness shading and put every scan on a common intensity scale
(WhiteStripe), so the model learns disease — not which machine took the scan."*

### Scanner/site leakage audit (`scripts/leakage_audit.py`)

**Why this exists:** harmonization *reduces* scanner bias, but how do we *prove*
the reported ROC-AUC reflects anatomy rather than a "which scanner" shortcut? Dr.
Tohka's exact advice: compare the image model against a model that sees **only**
the scanner/site metadata. If the metadata-only baseline scores nearly as high,
the labels are correlated with (leaked through) the scanner and the headline
number is partly fake.

**What it does:** it trains simple logistic-regression baselines on the training
manifest and evaluates them on the held-out test manifest — the *same* protocol
as the image model — then prints a comparison:

- **`site_only`** — one-hot of the three sites (Amsterdam / Singapore / Utrecht).
- **`clinical_only`** — the questionnaire columns (all zero in the MICCAI data,
  since those fields weren't collected there — so this is expected to sit at 0.5,
  a useful sanity check that the number reporting is honest).

It also reports each baseline's within-training 5-fold cross-validated AUC for
stability and writes a JSON report to `models/leakage_audit.json`.

**Result on the MICCAI WMH data:**

| Model | Test ROC-AUC |
|---|---|
| Image CNN (shipped) | **0.766** |
| Site-only baseline | 0.532 |
| Clinical-only baseline | 0.500 |

The metadata-only baselines are essentially at chance (0.5), so **there is no
meaningful leakage** — the CNN's 0.77 comes from the brain images, not from
guessing the scanner. That is the honest, defensible answer to the leakage
question, and it's also the prerequisite that makes any future ComBat/metadata
work trustworthy.

Run it with:

```bash
python scripts/leakage_audit.py \
  --train-manifest data/wmh_real/manifest_train/manifest.csv \
  --test-manifest  data/wmh_real/manifest_test/manifest.csv
```

**One-line summary for judges:** *"To make sure the score isn't cheating, we
trained a model on scanner/site labels alone — it scored 0.53 (basically a coin
flip), while the image model scored 0.77, so the AI is reading the brain, not the
barcode of the machine."*

### Salt-and-pepper noise, and the two ways we handle it

**What it is:** "salt-and-pepper" (a.k.a. *impulse*) noise is scattered pure-white
and pure-black specks in an image — like grains of salt and pepper sprinkled on
top. In MRI it comes from dead/hot scanner pixels, patient motion, and — most
relevant to this project — the **archive digitizer** (photographing film picks up
dust, scratches, glare, and camera-sensor noise). This matters because those
bright specks look exactly like the small bright white-matter hyperintensities the
model hunts for, so noise can invent **false lesions** → false positives → lower
accuracy. (Strictly, a scanner's *native* noise is usually Rician; salt-and-pepper
is the dominant nuisance on the film-photo path — worth saying precisely to
judges.)

We tackle it with **two complementary techniques**:

1. **Median-filter denoising (clean the input) — `median_filter_3d`.** For every
   voxel we look at its small 3×3×3 neighbourhood and replace it with the
   **median** (middle value) of those 27 voxels. A lone white or black speck is an
   extreme outlier, so the median simply ignores it — the speck vanishes — while
   real lesions and edges (which are backed up by their neighbours) survive. This
   is *why a median beats a blur*: an average would smear the speck around instead
   of removing it, and would soften true lesions. It's implemented with pure
   PyTorch tensor shifts (no SciPy dependency) and turned on with
   `--denoise 3`.
2. **Salt-and-pepper *augmentation* (toughen the model) — `add_salt_and_pepper`
   in `train_real.py`.** During training we *deliberately* sprinkle a small,
   random amount of salt-and-pepper noise onto each scan before the model sees it.
   By repeatedly showing the model noisy copies whose correct answer hasn't
   changed, it learns that isolated specks are *not* lesions and stops reacting to
   them. This is the standard way to make a model **robust** to noise it will meet
   in the real world (again, especially the film-digitizer path). Turned on with
   `--salt-pepper 0.02`.

**One-line summary for judges:** *"We both clean the image (a median filter that
deletes stray specks) and train the model on deliberately noisy scans, so
salt-and-pepper noise from the film scanner can't be mistaken for a lesion."*

**Judge point:** sorting DICOM slices by physical position (not filename) is the
kind of correctness detail that separates a real pipeline from a toy.

---

## `src/wmd/clinical.py` — the questionnaire

Defines the questions, turns answers into numbers, and (for training) generates
realistic synthetic answers per cause.

### `ClinicalField` and category order (26–37)
`ClinicalField` describes one question: `name` (code name), `label` (what the user
sees), `kind` ("age" or "binary" yes/no), and `category` (Demographics, History,
Symptoms, Genomic). `CATEGORY_ORDER` fixes the on-screen grouping order.

### `CLINICAL_FIELDS` — the actual questions (40–66)
A fixed, ordered list of 22 items: age, then risk factors (hypertension, diabetes,
prior stroke, smoking, cholesterol, autoimmune history, CNS infection, metabolic
disorder), symptoms (memory, gait, balance, concentration, mood, incontinence),
and genetic markers (APOE ε4, NOTCH3, HTRA1, COL4A1, MTHFR, family history, WMH
polygenic risk score).
- **Why the order is "frozen" (line 39 comment):** the position in this list *is*
  the position in the number vector the model learns. Reordering it after training
  would put every answer in the wrong slot.
- These markers map to real causes (e.g. NOTCH3 → CADASIL, a genetic small-vessel
  disease), which makes the questionnaire medically sensible.
- **`NUM_CLINICAL_FEATURES`** (68) = 22; **`CLINICAL_FIELD_NAMES`** (69) is just
  the list of code names.

### `_ETIOLOGY_PROFILES` — synthetic patient profiles (77–129)
For each cause, an **age range** and the **probability** that each yes/no field is
"yes." Example — vascular (83–93): older (62–86), 80% hypertension, 50% diabetes.
Genetic (103–112): younger, high NOTCH3 / family history. `_BASELINE_P = 0.07`
(line 75) means any field not listed for a cause is "yes" ~7% of the time
(background noise), so profiles aren't unrealistically clean.
- **This is the honest heart of the caveat:** "Our training patients are
  synthetic — we *designed* these correlations from medical literature. So the
  model proves the *pipeline* works; real accuracy needs real patient data."
  Saying this unprompted earns trust with judges.

### `encode_clinical` (132–146) — answers → numbers
Turns an answers dict into a fixed-length float vector: start all-zeros; age is
divided by 100 (so it's roughly 0–1 — neural nets like small comparable numbers);
yes/no becomes 1.0/0.0; **missing answers stay 0**, which is why a blank
questionnaire still works.

### `make_clinical` (149–174) — generate a synthetic patient
Given a cause, randomly draws an age in that cause's range and flips each yes/no
"coin" with that cause's probability. `rng` is a seedable random generator
(reproducible). Used only to build the training set.

---

## `src/wmd/synthetic.py` — fake brains for training & demos

Because we don't have a real labeled MRI dataset, this file *generates* brain-like
volumes so the whole pipeline is runnable. The module docstring says plainly it's
a stand-in for real data (e.g. OASIS-3, MICCAI WMH) and "has no clinical meaning."

- **`_ellipsoid_mask`** (24–35) — carves out an egg-shaped "brain" region, with a
  little random jitter so no two brains are identical.
- **`_add_blob`** (38–48) — adds a soft bright Gaussian "lesion" at a location.
  This is how we paint white-matter hyperintensities.
- **`make_volume`** (51–69) — builds one brain: base tissue brightness + gentle
  texture + noise; if `label == 1` (diseased) it adds vascular-style lesions.
- **`_add_etiology_lesions`** (72–114) — **the clever part.** Each cause gets a
  *loosely* characteristic lesion pattern: vascular = scattered deep spots;
  autoimmune = ovoid periventricular lesions near the midline (MS-like); genetic =
  symmetric anterior lesions in both hemispheres (CADASIL-like); metabolic =
  diffuse faint scatter; infectious = a few larger patchy lesions. The docstring
  explains *why these overlap on purpose* (77–80): in real life, imaging alone is
  rarely cause-specific, so the **questionnaire** is what mainly pins down the
  cause. This is what makes "fusion" meaningful rather than the image doing
  everything.
- **`make_etiology_volume`** (117–134) — like `make_volume` but for a specific
  cause index.
- **`generate_dataset`** (137–206) — builds a balanced set of volumes, saves each
  as a `.nii.gz`, and writes a `manifest.csv` listing every file with its label,
  cause, and questionnaire columns. Two important options:
  - `with_clinical` adds the questionnaire columns (for the multimodal model).
  - `clinical_noise` (default 0.3, lines 156–158, 187–190) means **30% of patients
    get a questionnaire drawn from a *different* cause.** *Why deliberately add
    noise?* So the clinical data is only *partially* predictive — this forces the
    MRI to stay the main disease-vs-healthy signal and stops the model from
    "cheating" by reading the questionnaire alone. (Excellent methodology point to
    raise with judges.)
  - It shuffles rows (194) so training batches are mixed.

**Judge point:** the synthetic generator is honest *and* thoughtfully designed —
the deliberate overlap and the 30% clinical noise are what make the multimodal
fusion a real test rather than a giveaway.

---

## `src/wmd/dataset.py` — feeding data to PyTorch

A PyTorch `Dataset` is just an object that can return "example number i." Training
loops pull from it.

- **`ManifestDataset`** (22–59) — for the image-only model. Reads the manifest CSV,
  resolves each scan's path (absolute or relative to the CSV), validates the
  header, and on `__getitem__` loads + preprocesses the scan and returns
  `(volume, label)`. Errors clearly if the manifest is empty or malformed.
- **`MultimodalManifestDataset`** (62–117) — for the real model. Same idea but also
  reads the questionnaire columns and a **target column**. It auto-picks the target
  (88): `etiology` (the cause) if that column exists, otherwise `label` (binary).
  It checks all 22 clinical columns are present (92–94). `__getitem__` returns
  `(volume, clinical_vector, label)`.
- **`labels()`** on both — used by the trainer to split classes evenly.

**Judge point:** the dataset layer cleanly separates "how data is stored" from
"how the model trains," so the same model code works whether data is synthetic or
real — only the manifest changes.

---

# PART D — Training (how the model learns)

## `src/wmd/train.py` — the training & evaluation loop

This is where the empty network from `model.py` actually *learns* from the data in
`dataset.py`.

- **`_set_seed`** (28–30) — fixes randomness so results are reproducible.
- **`_split_dataset`** (33–42) — splits examples into train (80%) and validation
  (20%), **stratified** so both halves keep the same class balance.
- **`evaluate`** (45–65) — runs the model on validation data without learning
  (`@torch.no_grad()`), and reports **accuracy** and **ROC-AUC** (a standard
  measure of how well it ranks positives above negatives). `model.eval()` turns
  off dropout for fair measurement.
- **`train`** (68–136) — the image-only trainer:
  - Picks GPU if available, else CPU (79).
  - Builds loaders, model, the **Adam** optimizer, and **CrossEntropyLoss** (the
    standard classification loss).
  - The epoch loop (96–117): for each batch, `zero_grad → forward → loss →
    backward → step` (the four lines that *are* deep learning). It tracks the best
    validation score and **keeps the best weights** (115–117), not just the last
    ones — so a late-epoch dip doesn't ruin the result.
  - Saves a checkpoint (124–134) bundling the weights **and** the settings
    (class names, target shape, clip percentiles, metrics) so inference uses the
    exact same setup as training.
- **`evaluate_multimodal`** (139–169) and **`train_multimodal`** (172–248) — the
  same structure for the real model, but the model takes `(volume, clinical)` and
  the ROC-AUC is computed multi-class ("one-vs-rest, macro average") when there are
  more than two causes (160–168). The multimodal checkpoint also stores the
  clinical field list and count (239–240) so the predictor knows the questionnaire
  layout.
- **`main` / `_parse_args`** (251–283) — lets you train from the command line, e.g.
  `python -m wmd.train --manifest ... --multimodal`.

**Judge point:** "keep the best validation checkpoint," stratified splits, and
saving config alongside weights are all real-ML-engineering practices, not
shortcuts.

---

# PART E — Inference (making a live prediction)

## `src/wmd/inference.py` — using the trained model

"Inference" = using the trained model to predict on a new scan. This file loads
the saved checkpoint and produces the label, confidence, the MRI-vs-clinical
split, and the Grad-CAM explanation.

### Result containers / dataclasses (20–55)
Simple labeled bundles of values:
- **`Prediction`** — the answer: `label` (e.g. "genetic"), `label_index`,
  `confidence` (0–1), and `probabilities` (all classes).
- **`ModalityAttribution`** — the MRI-vs-clinical breakdown. The docstring (30–38)
  is worth reading: everything is "probability of white matter disease." From a
  **neutral baseline** (blank scan, average age, no conditions), we add one input
  at a time and measure how far it moves the probability. The `image_share` /
  `clinical_share` split the total movement between the two.
- **`Explanation`** — info about the Grad-CAM heatmap (which slice, how focused).

### `WMDPredictor` (58–138) — the image-only predictor
- **`__init__`** loads the checkpoint; if the file is missing it raises a clear
  "train one first" error (63–67). `torch.load(..., map_location="cpu")` runs
  without a GPU and reads the stored settings so prediction matches training.
  `model.eval()` (78) switches to prediction mode (dropout off).
- **`predict_volume`** (80–93): `@torch.no_grad()` (faster — not training).
  Preprocess → run model → `softmax` to percentages → pick the highest → return a
  tidy `Prediction`.
- **`explain_path`** (98–138): produces two images of the **same** slice — the
  plain input and the Grad-CAM overlay — so a person can compare "what went in" vs
  "where the AI looked." `requires_grad_(True)` (118) is needed because Grad-CAM
  uses gradients.

### `_ImageBranchWrapper` (141–155) — a small but clever trick
Grad-CAM expects a model with **one** input (an image), but our multimodal model
takes **two** (image + clinical). This wrapper **freezes the clinical answers** and
exposes only the image input, so the same Grad-CAM code works unchanged. It also
re-exposes `.features` so Grad-CAM can hook the conv layers.
*Judge phrasing:* "We explain the **image's** contribution by holding the
questionnaire fixed and asking which brain regions moved the decision."

### `MultimodalWMDPredictor` (158–289) — the one the site uses
- **`__init__`** (161–188) loads the multimodal checkpoint, including the clinical
  field list and which class index means "healthy."
- **`_wmd_signal`** (190–194) — "probability of *any* white matter disease" =
  `1 − P(healthy)`. Tracking this single number makes the MRI-vs-clinical
  attribution meaningful regardless of which specific cause wins.
- **`_reference_clinical`** (199–201) — the neutral person: no conditions, age 55.
  The baseline we compare against.
- **`predict`** (203–246) — the heart of it:
  1. Preprocess scan + encode answers (207–208).
  2. `_prob(...)` runs the model and returns probabilities (210–211).
  3. Get the actual prediction (213–220).
  4. **Attribution** (222–245): compute the disease probability four ways —
     `baseline` (blank scan + neutral answers), `image_alone` (real scan + neutral
     answers), `clinical_alone` (blank scan + real answers), and `combined` (both
     real). Then `image_delta` / `clinical_delta` are how far each input moved
     things from baseline, and the **shares** are each one's portion of the total
     movement. This is exactly the "MRI 73% / Clinical 27%" bar on the result page.
  *Why this way?* It's an honest, simple **ablation**: turn an input off and see
  how much the answer changes. No fancy attribution math, easy to defend.
- **`explain_path`** (253–289) — Grad-CAM for the multimodal model, using the
  wrapper above to hold the clinical vector fixed.

### `save_preview` (292–308)
Makes a clean PNG of the middle slice for display (normalizes brightness using the
1st–99th percentile so it isn't washed out). Cosmetic, for the UI.

**Honest limitations to volunteer:**
- Attribution via on/off ablation is a reasonable approximation, not a formal
  Shapley-value attribution.
- "Confidence" is the model's softmax probability; a confident-but-wrong model is
  possible — which is exactly why we also show the heatmap and the disclaimer.

---

## `src/wmd/filmscan.py` — the Archive digitizer (server-side algorithm)

The problem: many old MRIs exist only as **physical film** — one sheet printed
with a grid of slice images (a "contact sheet"). This file converts a **photo of
that sheet** back into a 3D volume the existing AI can read. It's a *bridge*, not a
new prediction. (The camera hardware is set aside for now; this is the software
the website runs.)

### Normalization helpers (25–41)
- **`_volume_norm_bounds`** — finds one global brightness low/high for the whole
  volume (1st and 99.9th percentile). *Why global, not per-slice?* If you
  normalized each slice on its own, a nearly-empty slice would get amplified into
  noise and bright lesions would lose their relative punch. One global scale keeps
  slice-to-slice brightness honest. (Subtle but nice judge point.)
- **`_to_uint8`** — rescales to the 0–255 range images use.

### Making a contact sheet (44–96) — the reverse direction, for demos
- **`grid_shape_for_depth`** — given N slices and a chosen number of columns,
  computes how many rows are needed.
- **`contact_sheet_from_volume`** — lays slices out left-to-right, top-to-bottom
  into one big grayscale image **with no gaps** (line 56–57 comment) so it can be
  split back apart by simple even division. Each slice is resized to a `cell`
  square.
- **`save_contact_sheet`** — saves that montage to a PNG/JPEG.

### Cleaning a real photo (99–121)
- **`_auto_crop`** — a real photo has a dark border around the film; this trims to
  the bright region containing the slices, so the grid lines up when divided. It
  builds a mask of "bright enough" pixels and crops to their bounding box.
- **`_load_grayscale`** — opens the photo and converts to grayscale (MRIs are
  grayscale; color carries no extra info).

### `volume_from_contact_sheet` (124–168) — the key function
Reconstructs the 3D volume from the photo:
1. Optionally auto-crop the dark border (148–149).
2. Compute each cell's size by dividing the image into `rows × cols` (152); guard
   against an image too small to split (153–156).
3. Loop over the grid, cut out each cell, collect as slices (158–164).
4. Keep only the first `depth` cells (166–167) so blank trailing tiles are dropped.
5. `np.stack(...)` (168) stacks the 2D slices into one 3D cube.

**Honest limitation (you already tested this — admitting it is a strength):** a
phone-style photo of film is far lower fidelity than a native scan. Disease yes/no
survives digitization, but the **cause** often doesn't, because photographing film
blurs the fine lesion *shape* that distinguishes causes. We document this rather
than hide it — exactly the integrity ISEF judges reward.

---

# PART F — The website

## `webapp/main.py` — the FastAPI web server

This serves the pages, accepts uploads, runs the prediction, and renders the
result. It's the glue connecting all the other files.

### Setup (22–75)
- Lines 22–24 add `src` to Python's path so the site can import `wmd`.
- Imports the questionnaire, config labels/next-steps, the digitizer functions,
  and the multimodal predictor.
- `UPLOAD_DIR` / `PREVIEW_DIR` (37–40) — where uploads and preview images go
  (created if missing).
- `ALLOWED_SUFFIXES` / `FILM_SUFFIXES` (42–43) — accepted scan vs. photo types.
- `_latest_digitized` (47) — remembers the most recent digitizer result so the
  dashboard can show it even after a device capture. Resets on restart.
- `PRETTY_LABELS` (50–54) + `_pretty` (59–60) — map code names to friendly text.
- `app` (63), `/static` mount (64), Jinja2 templates (65) — standard FastAPI setup.
- **`_load_predictor`** (68–72) tries to load the model; if the file isn't there
  it returns `None` instead of crashing, so the site still runs and shows a helpful
  message. Line 75 loads it **once at startup** — loading per request would be slow.

### Helpers (78–122)
- **`_clinical_groups_for_template`** (78–89) — organizes questions by category for
  tidy form sections.
- **`_parse_clinical`** (92–104) — reads the submitted form into an answers dict:
  age parsed as a number (blank → 0), checkboxes → 1.0/0.0. Defensive — bad input
  falls back to 0 instead of crashing.
- **`_has_allowed_suffix` / `_has_film_suffix`** (107–114) — validate filename type.
- **`_parse_int`** (117–121) — safely read grid columns/depth.

### Basic pages (124–159)
- **`/health`** (124–130) — JSON status check (is the model loaded?). Useful for
  deployment/monitoring.
- **`/`** (133–145) — renders the main upload page with the disclaimer, validation
  metrics, grouped questions, and latest digitized result.
- **`_empty_context`** (148–159) — a blank result-page context, reused everywhere.

### `_run_prediction` (162–231) — the shared engine
Both the normal upload and the digitizer funnel through this so they produce the
**identical** result page:
1. If no model, return a friendly error.
2. Sweep away any stale Grad-CAM previews (`_cleanup_old_previews`), then run
   prediction + attribution.
3. Make the preview + Grad-CAM images.
4. Build the `explanation` and `attribution` blocks, converting 0–1 numbers into
   percentages (185–202).
5. Build `cause_probs` (204–211): all causes except "no_wmd," sorted highest-first
   — the ranked list with colored bars.
6. Build the `result` block: label, confidence, overall disease %, all
   probabilities, the cause list, a **severity indicator** (Mild/Moderate/Severe,
   via `assess_severity` — only for a positive result), and the **tailored next
   steps** for the predicted cause (pulled from config).
7. Update the "latest digitized" memory.

### `_film_to_volume_path` (234–242)
Turns a film photo into a real `.nii.gz` volume on disk (using `filmscan.py`), so
the rest of the pipeline treats it exactly like a normal scan. `affine=np.eye(4)`
is a placeholder coordinate system (fine for a demo volume).

### `POST /predict` (245–274) — the normal upload
1. Read the uploaded file; reject wrong types with a clear message (248–257).
2. Parse the questionnaire; make a unique token for filenames (259–260).
3. Save the upload (262–264).
4. Run `_run_prediction`; catch **any** error and show it instead of crashing
   (266–270).
5. **Always delete the uploaded file** afterward (271–272) — privacy + tidy disk.
6. Render the result page.

### Digitizer pages (277–326)
- **`GET /digitizer`** (277–289) — renders the digitizer page.
- **`POST /digitizer`** (292–326) — like `/predict` but the upload is a **photo**:
  it first reconstructs the volume (`_film_to_volume_path`) then runs the same
  prediction, and deletes both the photo and the rebuilt volume afterward.

### `POST /ingest/film` — the device endpoint
What a camera device posts to. Like `/digitizer` but returns **JSON** (for a
machine) instead of HTML: 503 if the model isn't loaded, **401 if the API key is
required but wrong/missing** (see below), 400 if the photo is missing/wrong type;
otherwise reconstruct → predict → update the dashboard's "latest" memory with a
timestamp → return label, confidence, disease %, and all probabilities.

### `_summarize_answers` (386–396)
Builds a readable "here's what you entered" list for the result page (age as a
number, yes/no for the rest, "—" if blank).

### How the app protects a user's data (the security model)

A common judge question is *"what happens to my MRI and my answers?"* Here is the
honest, complete answer, and the safeguards in the code:

- **Encrypted in transit.** The live site is hosted on Hugging Face Spaces, which
  serves everything over **HTTPS/TLS**, so the scan and questionnaire are
  encrypted between the browser and the server.
- **The scan is never kept.** The uploaded MRI is saved to a temp file with a
  random name, used for the prediction, then **deleted in a `finally` block** —
  so it's removed even if the prediction errors out (`/predict`, `/digitizer`,
  and `/ingest/film` all do this).
- **The questionnaire is never stored.** Answers live only in memory for the
  duration of the request; nothing is written to a database or a log.
- **Brain-slice preview images don't linger.** The Grad-CAM heatmap PNGs have to
  be written so the browser can show them, but **`_cleanup_old_previews`** sweeps
  the preview folder on every prediction and deletes any image older than
  `PREVIEW_TTL_SECONDS` (10 min) — long enough to display, then gone.
- **The device endpoint can be locked.** If you set the `NEUROPREDICT_API_KEY`
  environment variable, `/ingest/film` requires the camera to send a matching
  `X-API-Key` header (checked by **`_ingest_key_ok`**) or it returns **401**. Left
  unset, the endpoint stays open for easy local demos. The ESP32-CAM firmware has
  a matching `API_KEY` setting.
- **Bad input can't crash it.** Uploads are validated by file type and everything
  runs under `try/except`, so a corrupt file returns a friendly error instead of
  taking the server down. The model loads once at startup for speed.

**Honest gaps to acknowledge (it's a research demo):** there are no user logins,
no rate limiting, and it is **not** HIPAA/clinical-grade — which is exactly why
every page carries the "research and educational use only" disclaimer.

---

## The HTML templates (`webapp/templates/`)

These use **Jinja2** — HTML with `{{ values }}` and `{% logic %}` filled in by the
server. `{% extends %}` and `{% block %}` let pages share one layout.

- **`base.html`** — the shared shell: page `<head>`, the header/title, the
  `{% block content %}` placeholder each page fills, and the footer **disclaimer**.
  Every page inherits this, so the disclaimer is impossible to forget.
- **`index.html`** — the home page. Explains the multimodal model, links to the
  digitizer, shows a warning banner if no model is loaded or the validation metrics
  if it is, then the upload form: a file input (33–41) and the questionnaire
  rendered by looping over the grouped fields (48–68), with age as a number box and
  everything else as checkboxes.
- **`digitizer.html`** — the same idea for film: explains the archive problem,
  shows the "last digitized capture" banner, then a photo upload plus grid settings
  (columns/depth) and the same questionnaire.
- **`result.html`** — the result page. Shows an error banner if something failed;
  otherwise the headline label + confidence + overall disease %, the input slice
  and Grad-CAM images side by side, the ranked **cause** list with bars, **all
  class probabilities**, the **MRI-vs-clinical contribution bar** with the ablation
  numbers, the **next steps**, the clinical answers used, and a plain-English
  **"how this prediction was made"** pipeline. This page *is* the explainability
  story in visual form.

## `webapp/static/style.css` — the look

Plain CSS, no framework. A dark theme via CSS variables (`:root`, lines 1–10), card
layout, colored result boxes (orange = positive/disease, green = negative/healthy,
lines 100–101), the probability bars (139–158), the two-color MRI-vs-clinical
contribution bar (191–210), and a responsive questionnaire grid that reflows on
mobile (171–176). Nothing here affects predictions — it's purely presentation.

---

# PART G — Scripts & tests

## `scripts/generate_demo_data.py`
A thin command-line wrapper around `synthetic.generate_dataset` so you can create a
demo dataset with one command (`python scripts/generate_demo_data.py
--with-clinical --multiclass`). Lines 9 add `src` to the path; the rest just parses
arguments and calls the generator.

## `scripts/train_demo.py`
The one-command "make it work end-to-end" script the run scripts call. It
regenerates synthetic data **with** clinical columns and per-cause classes (33–35),
then trains **both** models (the image-only CNN and the multimodal model) and
prints their validation metrics. After this runs, the website has trained models to
serve.

## `scripts/prepare_wmh_data.py` — turn a real dataset into a manifest

This is the bridge from **real** MRI scans to our training code. It reads the
**MICCAI WMH Segmentation Challenge** dataset (60 training + 110 test brain MRIs
from three hospitals, each with a radiologist-drawn white-matter-lesion mask)
and writes a `manifest.csv` that the existing `ManifestDataset` already knows how
to read.

How it works, in plain terms:
- It **walks the dataset folders** and, for each patient, finds the FLAIR scan
  (`pre/FLAIR.nii.gz`) and the lesion mask (`wmh.nii.gz`). It's robust to the two
  folder layouts the challenge uses (some sites add an extra scanner subfolder).
- For each patient it **measures the lesion load** — it counts the voxels marked
  as lesion and multiplies by the real-world voxel size (from the NIfTI header)
  to get the total lesion volume in millilitres.
- It turns that number into a **label**: at or below a threshold (default 5 mL) =
  "low burden" (0), above = "significant burden" (1). This matters because the
  WMH dataset has *no healthy controls* — everyone has some lesions — so a
  simple yes/no isn't meaningful; the clinically sensible question is "how much."
- Because the real dataset has no questionnaire, the clinical columns are filled
  with zeros so the file still matches our manifest format.

**Judge point:** this is exactly the "swap synthetic for real" step. Nothing
about the model changes — only where the data comes from.

## `scripts/train_real.py` — train on the real scans

The real-data twin of `train_demo.py`. It trains the same 3D CNN, but on real
MRI, and evaluates on the **completely separate** challenge test set (train on
60, test on 110 — no overlap, the proper scientific protocol). Key pieces:
- **`AugmentedDataset`** — because 60 scans is tiny for a 3D CNN, it randomly
  flips each volume and jitters its brightness on the fly. This fights
  overfitting (the model can't just memorize 60 exact brains). With
  `--strong-aug` it adds three more transforms (Gaussian noise, a random
  intensity *gamma*, and a small spatial *translation*), and with
  `--salt-pepper` it also sprinkles random salt-and-pepper noise so the model
  learns to ignore specks (see the preprocessing section for the full story).
- **`pretrain_on_synthetic`** — *transfer learning*, and the single biggest lever
  on real-data accuracy. Before touching the 60 real scans, it trains the CNN on
  a large batch of cheap **synthetic** bright-lesion brains so the convolutional
  filters first learn to detect focal hyperintensities; fine-tuning then starts
  from those weights instead of random noise. The synthetic data is generated
  independently of the real test set, so nothing leaks. Turned on with
  `--pretrain-synthetic 200`.
- **`add_salt_and_pepper`** and the `--denoise` flag — the two noise-robustness
  techniques (train on noisy copies; optionally median-filter the input clean).
- **`--class-weights`** — weights the loss by inverse class frequency so the model
  stops taking the lazy shortcut of predicting "diseased" for everyone.
- A **cosine learning-rate schedule** and best-model-by-AUC checkpointing.
- It writes `models/performance_real.json` (including the `training_recipe` used)
  — the same format the performance page reads — so the confusion matrix and
  metrics come straight from real data.

**The honest headline result (real data, held-out test set).** A plain
from-scratch model is *unstable* on 60 brains — depending on the random seed it
lands anywhere from ~0.60 to ~0.72 ROC-AUC and often collapses to "everyone is
sick" (specificity 0). Adding **synthetic pretraining + class weighting + strong
augmentation** fixes both problems: ROC-AUC ≈ **0.77**, accuracy ≈ **0.67**,
sensitivity ≈ **0.74**, specificity ≈ **0.56** — and it is *reproducible*
(three different seeds all landed 0.78 ± 0.003 in a 5-fold cross-validation,
versus the old 0.60→0.72 swings). This still sits well below the ~1.0 the model
scores on synthetic data, and that gap is the whole point: synthetic data proves
the pipeline works; real data shows the genuine difficulty (only 60 training
brains, real-world scanner variety across three hospitals). Being upfront about
this — and about *why* pretraining helps — is exactly what earns trust from
judges and scientists.

*Recipe:* `python scripts/train_real.py --train-manifest … --test-manifest …
--pretrain-synthetic 200 --strong-aug --class-weights`.

## `tests/` — automated checks (run with `pytest`)
These are fast "smoke tests" that prove each piece works without needing a GPU or
real data. They're your evidence that the pipeline is correct.

- **`test_pipeline.py`** — image-only path: preprocessing output shape is correct;
  the model produces the right output shape; the dataset loads; a tiny end-to-end
  **train→predict** run produces valid probabilities (they sum to 1); and Grad-CAM
  returns a correctly-shaped 0–1 map.
- **`test_multimodal.py`** — the real model: `encode_clinical` scales age correctly
  and keeps yes/no binary; **synthetic profiles actually correlate with the cause**
  (e.g. genetic patients carry NOTCH3 far more than vascular ones, vascular carry
  hypertension more than genetic — lines 105–117); the multimodal model trains,
  predicts, and the **attribution shares add up to 1** (102); and multi-class
  (cause) training works.
- **`test_filmscan.py`** — the digitizer: grid math is right; a volume → contact
  sheet → volume **round-trip preserves depth and brightness structure**
  (correlation > 0.8, line 49); and auto-crop correctly handles a dark photo
  border.

**Judge point:** the tests don't just check "it runs" — they check that the
synthetic data carries the *intended* signal and that the digitizer round-trip is
faithful. That's testing the *science*, not just the code.

---

# Anticipated judge questions (and honest answers)

- **"Is this a real medical tool?"** No — it's a research/education prototype
  trained on synthetic data. Every result page shows that disclaimer. The
  contribution is the working, explainable, multimodal *pipeline*.
- **"Why should I trust the AI?"** You shouldn't blindly — that's why we show
  Grad-CAM (where it looked), the MRI-vs-clinical split (why), the full probability
  list, and a disclaimer. It's explainable by design.
- **"What's novel?"** Honestly, each component exists in the literature; the
  project's value is integrating detection + **cause** + explainability +
  recommended next steps + an archive-film digitizer into one accessible tool. The
  most original angle is the digitizer (rescuing scans trapped on film).
- **"How do you know the model isn't just reading the questionnaire?"** Because we
  deliberately add 30% "clinical noise" in the synthetic data, so the questionnaire
  is only partly predictive — the MRI has to carry the disease-vs-healthy signal.
  The attribution bar then *measures* how much each input actually contributed.
- **"What would make it real?"** Train on real labeled datasets (e.g. MICCAI WMH,
  OASIS-3), validate against radiologist labels, and test the digitizer on actual
  film. **We've started this:** `scripts/prepare_wmh_data.py` and
  `scripts/train_real.py` train and evaluate on the real MICCAI WMH Challenge
  scans (see those sections). The honest real-data result (ROC-AUC ≈ 0.77 vs.
  ~1.0 on synthetic) shows both that the pipeline generalizes to real data *and*
  how much harder real data is. Synthetic pretraining + class weighting + strong
  augmentation lifted it from an unstable ~0.6–0.72 to a stable ~0.77; the next
  step is more real data (e.g. OASIS-3, or collaborators' scans).
- **"Why a 3D CNN and not a normal image classifier?"** An MRI is a 3D volume;
  lesions have 3D shape and location. Flattening to 2D throws away depth.
- **"Why max pooling / GroupNorm?"** Max pooling preserves tiny bright lesions;
  GroupNorm is stable with the small batches we can afford on a CPU.
- **"Why is the cause unreliable on digitized film?"** A photo of film blurs the
  fine lesion *shape* that distinguishes causes, so the image signal degrades. We
  document this and recommend the native-scan upload for cause; the digitizer is
  best for the disease yes/no call.

## `neuropredict_all_in_one.py` — the whole project condensed into one file

The main project is split across many small files (one job each) because that is
how real software is kept maintainable. But for reading or explaining the project
in one sitting, this single file re-creates the **entire software pipeline** in
one place, top to bottom, in about 800 lines. Nothing in the main project is
removed — this is an *added*, self-contained copy you can read start to finish.

It is organized as nine numbered sections that mirror the rest of this
walkthrough:

1. **Configuration** — the class names, the five causes, the human-readable
   labels, the next-step guidance, and the training settings.
2. **Questionnaire** — the same clinical/genomic questions, the per-cause
   synthetic profiles, and `encode_clinical` (answers → numbers).
3. **Synthetic MRI generator** — builds an ellipsoid "brain" and adds
   cause-specific bright blobs (the stand-in for white matter lesions).
4. **Models** — the `ConvBlock`, the image-only `WMDClassifier3D`, and the
   `MultimodalWMDClassifier` that fuses the MRI embedding with the questionnaire.
5. **Grad-CAM** — hooks the last conv layer to produce the "where did it look?"
   heatmap.
6. **Training + evaluation** — trains both models on a train/val split and then
   scores them on a **fresh held-out test set** (a different random seed, so the
   models have never seen it). `_report` builds the **confusion matrix** and
   derives accuracy, sensitivity, specificity, and ROC-AUC from it.
7. **Inference** — the `Predictor` loads the trained weights and predicts
   detection + cause + Grad-CAM for one patient; it can also read an uploaded
   NIfTI file.
8. **Web app** — a FastAPI app that renders the prediction form and the
   **Model-Performance page** (the confusion matrices are drawn as color-shaded
   HTML tables — green diagonal for correct, blue for mistakes). It exposes an
   `app` object so it deploys to Hugging Face Spaces exactly like the main app.
9. **Command line** — `python neuropredict_all_in_one.py train` trains and prints
   the confusion matrix in the terminal; `serve` launches the website.

Because everything lives in one file, you can trace a single scan from raw
synthetic volume → 3D CNN → fused cause prediction → confusion matrix → web page
without jumping between modules. The trade-off (and why the real project is *not*
written this way) is that one giant file is harder to test and reuse in pieces —
so keep this as the "read it all at once" companion to the modular `src/` code.

---

*This walkthrough covers every software file in the repository. The IoT hardware
files (ESP32-CAM / XIAO firmware and the device simulator) are intentionally
omitted for now and can be added once the hardware is in hand.*
