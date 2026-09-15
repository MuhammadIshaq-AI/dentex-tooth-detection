"""Vercel entrypoint: FastAPI app serving the upload page and a JSON prediction API.

    POST /api/predict  (multipart form field "file")  -> findings with calibrated confidence + conformal intervals
    GET  /api/health
    GET  /                                             -> upload page

Run locally:  uvicorn app:app --reload   (from deploy/vercel/)
"""
import io
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from inference import Detector

BASE = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 4_400_000  # Vercel request bodies are capped at 4.5 MB; the page downsizes before upload
DISCLAIMER = ("Research prototype - not a medical device. Findings are model predictions trained on the DENTEX "
              "dataset and must be reviewed by a qualified dentist.")

app = FastAPI(title="DENTEX dental X-ray analysis", docs_url="/api/docs", openapi_url="/api/openapi.json")
_detector = None


def detector() -> Detector:
    global _detector  # load once per warm function instance
    if _detector is None:
        _detector = Detector(BASE / "model")
    return _detector


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html", media_type="text/html")


@app.get("/api/health")
def health():
    d = detector()
    return {"status": "ok", "classes": list(d.names.values()), "imgsz": d.imgsz}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image too large; please upload an image under 4.4 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read the uploaded file as an image.")

    t0 = time.perf_counter()
    findings = detector().detect(image)
    return {
        "image": {"width": image.width, "height": image.height},
        "findings": findings,
        "alpha": detector().meta["alpha"],
        "inference_ms": round(1000 * (time.perf_counter() - t0)),
        "disclaimer": DISCLAIMER,
    }
