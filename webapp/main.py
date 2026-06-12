"""FastAPI web interface for WMD prediction.

Run from the project root:
    uvicorn webapp.main:app --reload --port 8000
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Make the `wmd` package importable when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wmd.config import DEFAULT_MODEL_PATH, RESEARCH_DISCLAIMER  # noqa: E402
from wmd.inference import WMDPredictor, save_preview  # noqa: E402

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
            save_preview(saved_path, preview_png)
            preview_url = f"/static/previews/{preview_png.name}"
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
            "filename": scan.filename,
        },
    )
