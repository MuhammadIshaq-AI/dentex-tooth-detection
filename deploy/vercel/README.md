# Vercel deployment

**Live:** https://dentex-tooth-detection.vercel.app/

A lightweight, torch-free version of the detector for Vercel:

- `app.py`: FastAPI entrypoint (Vercel zero-config Python runtime). Serves the upload page at `/` and the JSON API at
  `POST /api/predict`, plus `/api/health` and `/api/docs`.
- `inference.py`: ONNX Runtime + NumPy/OpenCV inference. Its preprocessing matches Ultralytics predict (rectangular
  letterbox, linear resize), then applies class-aware NMS, per-class Platt calibration and conformal box intervals.
- `static/index.html`: upload page. It downsizes images in the browser to ≤ 2048 px JPEG before upload (Vercel request
  bodies are capped at 4.5 MB) and draws boxes, intervals and a findings table.
- `model/`: `dentex_yolo26s.onnx` (~39 MB) and `meta.json` (class names, calibration, conformal margins), produced by
  `python scripts/export_onnx.py` from the repo root.

MC-dropout is **not** available here, because 20 stochastic CPU passes per request is too slow for a serverless function. Use the
local Gradio app (`python app/app.py`) for that.

## Deploy

1. Refresh the model if you retrained: `python scripts/export_onnx.py`, then commit `deploy/vercel/model/`.
2. In Vercel, **Add New → Project**, import the GitHub repo, and set **Root Directory** to `deploy/vercel`.
   No build settings are needed; FastAPI is detected from `requirements.txt` and `app.py`.
3. Or use the CLI from this folder:
   ```bash
   npm i -g vercel
   cd deploy/vercel
   vercel          # preview deployment
   vercel --prod   # production
   ```

`vercel.json` sets `maxDuration: 60` for the function. Cold starts load the ONNX model once per instance (a few seconds);
warm requests take about 0.5 s on CPU.

## Run locally

```bash
pip install -r deploy/vercel/requirements.txt uvicorn
cd deploy/vercel
uvicorn app:app --reload        # http://127.0.0.1:8000
```

Research prototype, **not a medical device**. DENTEX data and derived weights are CC BY-NC-SA 4.0 (non-commercial).
