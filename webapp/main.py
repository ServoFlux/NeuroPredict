from __future__ import annotations

import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import nibabel as nib
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.clinical import CATEGORY_ORDER, CLINICAL_FIELDS
from wmd.config import (
    DEFAULT_MULTIMODAL_MODEL_PATH,
    ETIOLOGY_LABELS,
    ETIOLOGY_NEXT_STEPS,
    RESEARCH_DISCLAIMER,
    assess_severity,
)
from wmd.filmscan import grid_shape_for_depth, volume_from_contact_sheet
from wmd.inference import MultimodalWMDPredictor

WEBAPP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = WEBAPP_DIR / "uploads"
PREVIEW_DIR = WEBAPP_DIR / "static" / "previews"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = (".nii", ".nii.gz", ".dcm", ".ima")
FILM_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

PREVIEW_TTL_SECONDS = 600

INGEST_API_KEY = os.getenv("NEUROPREDICT_API_KEY")

_latest_digitized: dict[str, object] = {"label_pretty": None, "wmd_pct": None, "ts": None}

PRETTY_LABELS = {
    "no_wmd": "No White Matter Disease",
    "early_wmd": "Early White Matter Disease",
    **ETIOLOGY_LABELS,
}

_TRUTHY = {"1", "on", "true", "yes"}

def _pretty(label: str) -> str:
    return PRETTY_LABELS.get(label, label.replace("_", " ").title())

app = FastAPI(title="NeuroPredict — Early White Matter Disease Risk Prediction")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

def _load_predictor() -> MultimodalWMDPredictor | None:
    try:
        return MultimodalWMDPredictor(DEFAULT_MULTIMODAL_MODEL_PATH)
    except FileNotFoundError:
        return None

predictor = _load_predictor()

def _clinical_groups_for_template() -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    for category in CATEGORY_ORDER:
        fields = [
            {"name": f.name, "label": f.label, "kind": f.kind, "help": f.help}
            for f in CLINICAL_FIELDS
            if f.category == category
        ]
        if fields:
            groups.append({"category": category, "fields": fields})
    return groups

def _parse_clinical(form: object) -> dict[str, float]:
    answers: dict[str, float] = {}
    for field in CLINICAL_FIELDS:
        raw = form.get(field.name)
        if field.kind == "age":
            try:
                answers[field.name] = float(raw) if raw not in (None, "") else 0.0
            except (TypeError, ValueError):
                answers[field.name] = 0.0
        else:
            answers[field.name] = 1.0 if str(raw).lower() in _TRUTHY else 0.0
    return answers

def _has_suffix(filename: str, suffixes: tuple[str, ...]) -> bool:
    name = filename.lower()
    return any(name.endswith(suffix) for suffix in suffixes)

def _parse_int(raw: object, default: int) -> int:
    try:
        return int(float(str(raw)))
    except (TypeError, ValueError):
        return default

def _cleanup_old_previews(max_age_seconds: int = PREVIEW_TTL_SECONDS) -> None:
    now = time.time()
    for preview in PREVIEW_DIR.glob("*.png"):
        try:
            if now - preview.stat().st_mtime > max_age_seconds:
                preview.unlink(missing_ok=True)
        except OSError:
            pass

def _ingest_key_ok(request: Request, form: object) -> bool:
    if not INGEST_API_KEY:
        return True
    provided = request.headers.get("x-api-key") or form.get("api_key")
    return provided == INGEST_API_KEY

@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_path": str(DEFAULT_MULTIMODAL_MODEL_PATH),
    }

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "disclaimer": RESEARCH_DISCLAIMER,
            "model_loaded": predictor is not None,
            "val_metrics": predictor.val_metrics if predictor else {},
            "clinical_groups": _clinical_groups_for_template(),
            "latest_digitized": _latest_digitized,
        },
    )

def _empty_context(filename: str | None = None) -> dict[str, object]:
    return {
        "disclaimer": RESEARCH_DISCLAIMER,
        "error": None,
        "result": None,
        "preview_url": None,
        "overlay_url": None,
        "explanation": None,
        "attribution": None,
        "answers_summary": [],
        "filename": filename,
    }

def _run_prediction(
    saved_path: Path, answers: dict[str, float], filename: str, token: str
) -> dict[str, object]:
    ctx = _empty_context(filename)
    if predictor is None:
        ctx["error"] = (
            "No trained model is available. Run `python scripts/train_demo.py` "
            "to train the demo models, then restart the server."
        )
        return ctx

    _cleanup_old_previews()
    prediction, attr = predictor.predict_path(saved_path, answers)

    preview_png = PREVIEW_DIR / f"{token}.png"
    overlay_png = PREVIEW_DIR / f"{token}_cam.png"
    exp = predictor.explain_path(
        saved_path, answers, prediction, overlay_png, preview_png
    )
    ctx["preview_url"] = f"/static/previews/{preview_png.name}"
    ctx["overlay_url"] = f"/static/previews/{overlay_png.name}"
    ctx["explanation"] = {
        "original_shape": "×".join(str(d) for d in exp.original_shape),
        "processed_shape": "×".join(str(d) for d in exp.processed_shape),
        "slice_index": exp.slice_index,
        "attention_pct": round(exp.attention_fraction * 100, 1),
        "clip_percentiles": predictor.preprocess.clip_percentiles,
    }
    ctx["attribution"] = {
        "combined": round(attr.combined * 100, 1),
        "baseline": round(attr.baseline * 100, 1),
        "image_delta": round(attr.image_delta * 100, 1),
        "clinical_delta": round(attr.clinical_delta * 100, 1),
        "image_share": round(attr.image_share * 100),
        "clinical_share": round(attr.clinical_share * 100),
    }
    ctx["answers_summary"] = _summarize_answers(answers)
    is_positive = prediction.label != "no_wmd"
    severity = assess_severity(attr.combined) if is_positive else None
    cause_probs = [
        {"label": _pretty(name), "pct": round(p * 100, 1)}
        for name, p in sorted(
            prediction.probabilities.items(),
            key=lambda kv: kv[1], reverse=True,
        )
        if name != "no_wmd"
    ]
    ctx["result"] = {
        "label": prediction.label,
        "label_pretty": _pretty(prediction.label),
        "confidence": round(prediction.confidence * 100, 1),
        "wmd_probability": round(attr.combined * 100, 1),
        "probabilities": {
            _pretty(name): round(p * 100, 1)
            for name, p in prediction.probabilities.items()
        },
        "cause_probs": cause_probs,
        "is_positive": is_positive,
        "severity": (
            {"level": severity.level, "description": severity.description}
            if severity is not None
            else None
        ),
        "next_steps": ETIOLOGY_NEXT_STEPS.get(
            prediction.label, ETIOLOGY_NEXT_STEPS["no_wmd"]
        ),
    }
    _latest_digitized.update(
        label_pretty=ctx["result"]["label_pretty"],
        wmd_pct=ctx["result"]["wmd_probability"],
    )
    return ctx

def _film_to_volume_path(
    image_path: Path, token: str, cols: int, depth: int
) -> Path:
    rows, cols = grid_shape_for_depth(depth, cols)
    volume = volume_from_contact_sheet(image_path, rows=rows, cols=cols, depth=depth)
    nii_path = UPLOAD_DIR / f"{token}.nii.gz"
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4)), str(nii_path))
    return nii_path

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request) -> HTMLResponse:
    form = await request.form()
    scan = form.get("scan")
    filename = getattr(scan, "filename", None)

    if scan is None or not filename or not _has_suffix(filename, ALLOWED_SUFFIXES):
        ctx = _empty_context(filename)
        ctx["error"] = (
            "Unsupported file type. Please upload a NIfTI (.nii/.nii.gz) or "
            "DICOM (.dcm) scan."
        )
        return templates.TemplateResponse(request, "result.html", ctx)

    answers = _parse_clinical(form)
    token = uuid.uuid4().hex
    suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else Path(filename).suffix
    saved_path = UPLOAD_DIR / f"{token}{suffix}"
    with saved_path.open("wb") as out:
        shutil.copyfileobj(scan.file, out)

    try:
        ctx = _run_prediction(saved_path, answers, filename, token)
    except Exception as exc:
        ctx = _empty_context(filename)
        ctx["error"] = f"Could not process this scan: {exc}"
    finally:
        saved_path.unlink(missing_ok=True)

    return templates.TemplateResponse(request, "result.html", ctx)

@app.get("/digitizer", response_class=HTMLResponse)
def digitizer(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "digitizer.html",
        {
            "disclaimer": RESEARCH_DISCLAIMER,
            "model_loaded": predictor is not None,
            "clinical_groups": _clinical_groups_for_template(),
            "latest": _latest_digitized,
        },
    )

@app.post("/digitizer", response_class=HTMLResponse)
async def digitizer_submit(request: Request) -> HTMLResponse:
    form = await request.form()
    sheet = form.get("sheet")
    filename = getattr(sheet, "filename", None)

    if sheet is None or not filename or not _has_suffix(filename, FILM_SUFFIXES):
        ctx = _empty_context(filename)
        ctx["error"] = (
            "Please upload a photo of the film sheet (PNG/JPEG)."
        )
        return templates.TemplateResponse(request, "result.html", ctx)

    answers = _parse_clinical(form)
    cols = _parse_int(form.get("cols"), default=8)
    depth = _parse_int(form.get("depth"), default=64)
    token = uuid.uuid4().hex
    photo_path = UPLOAD_DIR / f"{token}{Path(filename).suffix.lower()}"
    with photo_path.open("wb") as out:
        shutil.copyfileobj(sheet.file, out)

    nii_path: Path | None = None
    try:
        nii_path = _film_to_volume_path(photo_path, token, cols, depth)
        ctx = _run_prediction(nii_path, answers, f"{filename} (digitized film)", token)
    except Exception as exc:
        ctx = _empty_context(filename)
        ctx["error"] = f"Could not reconstruct this film sheet: {exc}"
    finally:
        photo_path.unlink(missing_ok=True)
        if nii_path is not None:
            nii_path.unlink(missing_ok=True)

    return templates.TemplateResponse(request, "result.html", ctx)

@app.post("/ingest/film")
async def ingest_film(request: Request) -> JSONResponse:
    if predictor is None:
        return JSONResponse({"ok": False, "error": "model not loaded"}, status_code=503)

    form = await request.form()
    if not _ingest_key_ok(request, form):
        return JSONResponse(
            {"ok": False, "error": "invalid or missing API key"}, status_code=401
        )

    sheet = form.get("sheet")
    filename = getattr(sheet, "filename", None)
    if sheet is None or not filename or not _has_suffix(filename, FILM_SUFFIXES):
        return JSONResponse(
            {"ok": False, "error": "missing film image (field 'sheet')"},
            status_code=400,
        )

    answers = _parse_clinical(form)
    cols = _parse_int(form.get("cols"), default=8)
    depth = _parse_int(form.get("depth"), default=64)
    token = uuid.uuid4().hex
    photo_path = UPLOAD_DIR / f"{token}{Path(filename).suffix.lower()}"
    with photo_path.open("wb") as out:
        shutil.copyfileobj(sheet.file, out)

    nii_path: Path | None = None
    try:
        nii_path = _film_to_volume_path(photo_path, token, cols, depth)
        prediction, attr = predictor.predict_path(nii_path, answers)
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    finally:
        photo_path.unlink(missing_ok=True)
        if nii_path is not None:
            nii_path.unlink(missing_ok=True)

    wmd_pct = round(attr.combined * 100, 1)
    _latest_digitized.update(
        label_pretty=_pretty(prediction.label), wmd_pct=wmd_pct, ts=time.time()
    )
    return JSONResponse(
        {
            "ok": True,
            "label": prediction.label,
            "label_pretty": _pretty(prediction.label),
            "confidence": round(prediction.confidence * 100, 1),
            "wmd_probability": wmd_pct,
            "probabilities": {
                name: round(p * 100, 1)
                for name, p in prediction.probabilities.items()
            },
        }
    )

def _summarize_answers(answers: dict[str, float]) -> list[dict[str, str]]:
    summary: list[dict[str, str]] = []
    for field in CLINICAL_FIELDS:
        val = answers.get(field.name, 0.0)
        if field.kind == "age":
            display = f"{int(val)}" if val else "—"
        else:
            display = "Yes" if val >= 0.5 else "No"
        summary.append({"label": field.label, "value": display})
    return summary
