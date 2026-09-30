"""HTTP API: upload a page, get the translated page back. Also serves the frontend."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .pipeline import translate_image

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="Mantran", version="0.2.0")
# Lets the frontend call the API when it is opened from another origin
# (a different dev server, or straight from disk).
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health_check():
    return {"status": "ok"}


# A plain ``def`` so FastAPI runs the CPU-heavy pipeline in its thread pool.
@app.post("/api/translate")
def translate_upload(
    file: UploadFile = File(...),
    source_lang: str = Form("ja"),
    target_lang: str = Form("en"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    image = cv2.imdecode(np.frombuffer(file.file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    result, metadata = translate_image(image, source_lang=source_lang, target_lang=target_lang)
    ok, encoded = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode result image")

    return {
        "output_image": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "mime_type": "image/png",
        "extracted_text": metadata["extracted_text"],
        "translated_text": metadata["translated_text"],
        "regions": metadata["regions"],
    }


# Mounted last so the API routes above take precedence over static files.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
