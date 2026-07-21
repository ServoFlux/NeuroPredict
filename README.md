# NeuroPredict — Early White Matter Disease Risk Prediction from Archive MRI Scans

A research/educational project that predicts **white matter disease** (white
matter hyperintensities, WMH) from brain MRI and, when present, its likely
**cause (etiology)** — vascular, autoimmune, genetic, metabolic, or infectious.
It uses a **multimodal model** that fuses a **3D CNN** over the MRI with a short
**clinical & genomic questionnaire**, exposed through a **FastAPI web
interface**.

> ⚠️ **Disclaimer:** This is NOT a medical device and must NOT be used for
> diagnosis or any clinical decision. It is for research and learning only.

---

## What it does

1. **Preprocess** an MRI volume (NIfTI or DICOM) → robust intensity
   normalization → resample to a fixed shape.
2. **Encode** a short clinical & genomic questionnaire (history, symptoms, and
   genetic markers) into a feature vector.
3. **Fuse & classify** with a multimodal model: a 3D CNN over the MRI plus a
   small MLP over the questionnaire, combined in a shared head. It predicts one
   of `no_wmd`, `vascular`, `autoimmune`, `genetic`, `metabolic`, `infectious`.
   (An image-only `no_wmd`/`early_wmd` CNN is also trained as a baseline.)
4. **Serve** predictions through a web app: upload a scan, answer the
   questionnaire, and see the predicted cause, the overall WMD probability, a
   per-modality contribution breakdown (MRI vs. clinical), a Grad-CAM heatmap,
   and suggested next steps.

The repo ships with a **synthetic data generator** and a **demo training
script** so the entire pipeline runs end-to-end without any gated data. You can
later swap in real labeled data and train a real model.

---

## Run it yourself (no account, no token, no Devin needed)

You can run the whole website on your own computer with **one command**. It works
offline and does not depend on any hosted session.

**macOS / Linux**
```bash
./run.sh
```
(the very first time, make it runnable once: `chmod +x run.sh`)

**Windows** — just **double-click `run.bat`** (or run `run.bat` in a terminal).

That single step does everything for you: it creates a private Python
environment, installs the libraries, trains the small demo model the first time
(a few minutes on a normal laptop, CPU only), then starts the site and opens
your browser at **http://localhost:8000**. Leave the window open while you use
it; press `Ctrl+C` to stop. Running it again later is instant (it reuses the
setup and the trained model).

> Requirement: **Python 3** must be installed. Get it from
> [python.org/downloads](https://www.python.org/downloads/) (on Windows, tick
> "Add Python to PATH" during install).

### Or run it with Docker (zero Python setup)

If you have Docker, you don't need Python at all:
```bash
docker compose up --build      # then open http://localhost:8000
```
(or `docker build -t neuropredict . && docker run -p 8000:8000 neuropredict`)

---

## Quickstart (manual steps, if you prefer)

```bash
# 1. Create an environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate synthetic demo data + train a small demo model
python scripts/train_demo.py        # writes models/wmd_cnn.pt + wmd_multimodal.pt

# 3. Launch the web app
uvicorn webapp.main:app --port 8000
# open http://localhost:8000
```

Upload one of the generated synthetic scans from
`data/synthetic/volumes/*.nii.gz` to see a prediction.

Run the tests:

```bash
pytest -q
```

---

## Project layout

```
src/wmd/
  config.py         # central config (shapes, classes, disclaimer)
  preprocessing.py  # load NIfTI/DICOM, normalize, resample
  dataset.py        # PyTorch Dataset from a manifest CSV
  model.py          # 3D CNN (WMDClassifier3D)
  train.py          # training + evaluation, saves checkpoint
  inference.py      # load checkpoint, predict, slice preview
  synthetic.py      # synthetic brain/lesion volume generator
scripts/
  generate_demo_data.py
  train_demo.py
webapp/
  main.py           # FastAPI app (upload -> predict)
  templates/        # Jinja2 HTML
  static/           # CSS + generated slice previews
tests/
  test_pipeline.py
```

---

## Training on real data

The demo uses synthetic data. For a real model you need real MRI scans **and
labels**. Two complementary public datasets:

### 1. OASIS-3 (scale + validation)
- [oasis-brains.org](https://www.oasis-brains.org/) — large longitudinal aging /
  Alzheimer's dataset (T1, T2, **FLAIR**, PET) in NIfTI/BIDS.
- **Access is gated:** register and sign the Data Use Agreement; data is served
  via XNAT Central / NITRC.
- OASIS-3 does **not** ship a ready "early WMD" label. White matter disease is
  best seen on **FLAIR** and graded by the **Fazekas scale (0–3)** or WMH
  volume. You derive labels from available derivatives / radiologic readings.

### 2. MICCAI WMH Segmentation Challenge 2017 (ground-truth labels) — recommended for supervised training
- [wmh.isi.uu.nl](https://wmh.isi.uu.nl/) — FLAIR + T1 scans with **expert WMH
  segmentation masks**. From masks you can derive a binary label (WMH volume
  above a threshold ⇒ `early_wmd`) or a load category.

### Wiring real data in
Create a `manifest.csv` with columns `path,label` (label `0 = no_wmd`,
`1 = early_wmd`), where `path` points to a NIfTI/DICOM scan, then:

```bash
python -m wmd.train --manifest /path/to/manifest.csv --epochs 50 \
    --model-path models/wmd_cnn.pt
```

The web app automatically picks up `models/wmd_cnn.pt` on restart.

### Recommended next steps for a real model
- Use **FLAIR** as the primary modality (optionally stack T1 as a second channel).
- Add **skull stripping** and **bias-field correction** (e.g. via FSL/ANTs or
  MONAI transforms) in `preprocessing.py`.
- Train on a **GPU**; increase `target_shape` (e.g. 96³ or 128³) and add data
  augmentation.
- Consider reframing as **WMH segmentation** (e.g. a 3D U-Net) and deriving the
  classification/Fazekas grade from the predicted lesion load.

---

## Notes
- CPU-friendly by design (small model + 64³ volumes) so it runs anywhere.
- Uploaded scans are processed in-memory and deleted after inference; only a
  downsized PNG slice preview is kept for display.
