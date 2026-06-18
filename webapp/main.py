"""FastAPI web interface for WMD prediction.

Run from the project root:
    uvicorn webapp.main:app --reload --port 8000
"""

from __future__ import annotations

import shutil
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Make the `wmd` package importable when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.config import DEFAULT_MODEL_PATH, RESEARCH_DISCLAIMER  # noqa: E402
from wmd.inference import WMDPredictor  # noqa: E402
from wmd.risk import (  # noqa: E402
    Vitals,
    combined_risk,
    vascular_risk_score,
)

WEBAPP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = WEBAPP_DIR / "uploads"
PREVIEW_DIR = WEBAPP_DIR / "static" / "previews"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = (".nii", ".nii.gz", ".dcm", ".ima")

# Human-friendly display names for the model's class labels.
PRETTY_LABELS = {
    "no_wmd": "No White Matter Disease",
    "early_wmd": "Early White Matter Disease",
}


def _pretty(label: str) -> str:
    return PRETTY_LABELS.get(label, label.replace("_", " ").title())


app = FastAPI(title="NeuroPredict — Early White Matter Disease Risk Prediction")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))


def _load_predictor() -> WMDPredictor | None:
    try:
        return WMDPredictor(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        return None


predictor = _load_predictor()

# In-memory IoT state (resets on restart): the latest vitals reading per device
# and the most recent MRI prediction, used to compute a combined risk.
_latest_vitals: dict[str, dict] = {}
_latest_mri: dict[str, object] = {"probability": None, "label_pretty": None, "ts": None}


class VitalsIn(BaseModel):
    """JSON body posted by the ESP32 companion device."""

    device_id: str = "esp32"
    heart_rate: float | None = None
    spo2: float | None = None
    systolic: float | None = None
    diastolic: float | None = None
    age: float | None = None


def _has_allowed_suffix(filename: str) -> bool:
    name = filename.lower()
    return any(name.endswith(suffix) for suffix in ALLOWED_SUFFIXES)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_path": str(DEFAULT_MODEL_PATH),
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
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, scan: UploadFile = File(...)) -> HTMLResponse:
    error: str | None = None
    result = None
    preview_url = None
    explanation = None
    overlay_url = None

    if predictor is None:
        error = (
            "No trained model is available. Run `python scripts/train_demo.py` "
            "to train the demo model, then restart the server."
        )
    elif not scan.filename or not _has_allowed_suffix(scan.filename):
        error = (
            "Unsupported file type. Please upload a NIfTI (.nii/.nii.gz) or "
            "DICOM (.dcm) scan."
        )
    else:
        token = uuid.uuid4().hex
        suffix = ".nii.gz" if scan.filename.lower().endswith(".nii.gz") else Path(scan.filename).suffix
        saved_path = UPLOAD_DIR / f"{token}{suffix}"
        with saved_path.open("wb") as out:
            shutil.copyfileobj(scan.file, out)

        try:
            prediction = predictor.predict_path(saved_path)

            preview_png = PREVIEW_DIR / f"{token}.png"
            overlay_png = PREVIEW_DIR / f"{token}_cam.png"
            exp = predictor.explain_path(
                saved_path, prediction, overlay_png, preview_png
            )
            preview_url = f"/static/previews/{preview_png.name}"
            overlay_url = f"/static/previews/{overlay_png.name}"
            explanation = {
                "original_shape": "×".join(str(d) for d in exp.original_shape),
                "processed_shape": "×".join(str(d) for d in exp.processed_shape),
                "slice_index": exp.slice_index,
                "attention_pct": round(exp.attention_fraction * 100, 1),
                "clip_percentiles": predictor.preprocess.clip_percentiles,
            }
            result = {
                "label": prediction.label,
                "label_pretty": _pretty(prediction.label),
                "confidence": round(prediction.confidence * 100, 1),
                "probabilities": {
                    _pretty(name): round(p * 100, 1)
                    for name, p in prediction.probabilities.items()
                },
                "is_positive": prediction.label_index == 1,
            }
            # Record for IoT combined-risk fusion on the /iot dashboard.
            _latest_mri["probability"] = prediction.probabilities.get("early_wmd")
            _latest_mri["label_pretty"] = _pretty(prediction.label)
            _latest_mri["ts"] = time.time()
        except Exception as exc:  # noqa: BLE001 - surface any decode/inference error
            error = f"Could not process this scan: {exc}"
        finally:
            saved_path.unlink(missing_ok=True)

    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "disclaimer": RESEARCH_DISCLAIMER,
            "error": error,
            "result": result,
            "preview_url": preview_url,
            "overlay_url": overlay_url,
            "explanation": explanation,
            "filename": scan.filename,
        },
    )


@app.post("/iot/vitals")
def ingest_vitals(payload: VitalsIn) -> JSONResponse:
    """Ingest a vitals reading from the ESP32 companion device.

    Computes the vascular-risk score, stores the latest reading for the device,
    and echoes the score back (handy for an on-device display).
    """
    vitals = Vitals(
        heart_rate=payload.heart_rate,
        spo2=payload.spo2,
        systolic=payload.systolic,
        diastolic=payload.diastolic,
        age=payload.age,
    )
    risk = vascular_risk_score(vitals)
    _latest_vitals[payload.device_id] = {
        "vitals": payload.model_dump(),
        "vascular_score": round(risk.score, 3),
        "vascular_level": risk.level,
        "factors": risk.factors,
        "ts": time.time(),
    }
    return JSONResponse(
        {
            "ok": True,
            "device_id": payload.device_id,
            "vascular_score": round(risk.score, 3),
            "vascular_level": risk.level,
            "factors": risk.factors,
        }
    )


@app.get("/iot", response_class=HTMLResponse)
def iot_dashboard(request: Request) -> HTMLResponse:
    """Live dashboard: latest device vitals + combined (MRI + vascular) risk."""
    devices = []
    mri_prob = _latest_mri.get("probability")
    for device_id, entry in sorted(_latest_vitals.items()):
        combined = None
        if mri_prob is not None:
            risk = vascular_risk_score(
                Vitals(**{
                    k: entry["vitals"].get(k)
                    for k in ("heart_rate", "spo2", "systolic", "diastolic", "age")
                })
            )
            fused = combined_risk(float(mri_prob), risk)
            combined = {
                "score": round(fused.score * 100, 1),
                "level": fused.level,
            }
        devices.append({
            "device_id": device_id,
            "vitals": entry["vitals"],
            "vascular_score": round(entry["vascular_score"] * 100, 1),
            "vascular_level": entry["vascular_level"],
            "factors": entry["factors"],
            "age_seconds": int(time.time() - entry["ts"]),
            "combined": combined,
        })

    return templates.TemplateResponse(
        request,
        "iot.html",
        {
            "disclaimer": RESEARCH_DISCLAIMER,
            "devices": devices,
            "mri": _latest_mri,
            "mri_pct": round(float(mri_prob) * 100, 1) if mri_prob is not None else None,
        },
    )
