"""HTTP API: translate single pages or whole MangaDex chapters. Also serves the frontend."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import mangadex
from .jobs import build_zip, manager, zip_name
from .languages import LANGUAGES, check_pair, language_name, source_languages, target_languages
from .pipeline import translate_image

MAX_UPLOAD_FILES = 500
UPLOAD_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

app = FastAPI(title="Mantran", version="0.3.0")
# Lets the frontend call the API when it is opened from another origin
# (a different dev server, or straight from disk).
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"]
)


def _bad_request(exc: Exception):
    raise HTTPException(status_code=400, detail=str(exc)) from exc


def _mangadex_error(exc: mangadex.MangaDexError):
    raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/languages")
def list_languages():
    return {
        "sources": [{"code": lang.code, "name": lang.name} for lang in source_languages()],
        "targets": [{"code": lang.code, "name": lang.name} for lang in target_languages()],
    }


# Plain ``def`` handlers so FastAPI runs the CPU-heavy work in its thread pool.
@app.post("/api/translate")
def translate_upload(
    file: UploadFile = File(...),
    source_lang: str = Form("ja"),
    target_lang: str = Form("en"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        check_pair(source_lang, target_lang)
    except ValueError as exc:
        _bad_request(exc)

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


# --- MangaDex ----------------------------------------------------------------------


@app.get("/api/mangadex/lookup")
def mangadex_lookup(url: str):
    """Resolve a title or chapter link to the title, its languages and (for chapter links) the chapter."""
    try:
        kind, item_id = mangadex.parse_link(url)
        chapter = None
        if kind in ("chapter", "unknown"):
            try:
                chapter = mangadex.get_chapter(item_id)
            except mangadex.MangaDexError:
                if kind == "chapter":
                    raise
        manga = mangadex.get_manga(chapter["manga_id"] if chapter else item_id)
    except mangadex.MangaDexError as exc:
        _mangadex_error(exc)

    manga["languages"] = [
        {"code": code, "name": language_name(code), "supported": code in LANGUAGES}
        for code in manga["languages"]
    ]
    if chapter:
        chapter["label"] = mangadex.chapter_label(chapter)
    return {"manga": manga, "chapter": chapter}


@app.get("/api/mangadex/manga/{manga_id}/chapters")
def mangadex_chapters(manga_id: str, lang: str):
    try:
        chapters = mangadex.list_chapters(manga_id, lang)
    except mangadex.MangaDexError as exc:
        _mangadex_error(exc)
    for chapter in chapters:
        chapter["label"] = mangadex.chapter_label(chapter)
    return {"chapters": chapters}


# --- Jobs: MangaDex chapters and uploaded batches -----------------------------------


class ChapterRequest(BaseModel):
    chapter_id: str
    source_lang: str
    target_lang: str


def _job_or_404(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job (the server may have restarted)")
    return job


@app.post("/api/jobs")
def start_chapter_job(request: ChapterRequest):
    try:
        return manager.submit(request.chapter_id, request.source_lang, request.target_lang).to_dict()
    except ValueError as exc:
        _bad_request(exc)


@app.post("/api/uploads")
def start_upload_job(
    files: list[UploadFile] = File(...),
    source_lang: str = Form("ja"),
    target_lang: str = Form("en"),
    title: str = Form(""),
):
    """Translate uploaded pages as one batch, in the order they were sent."""
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_UPLOAD_FILES} pages at a time")
    pages = []
    for upload in files:
        name = upload.filename or "page.png"
        if Path(name).suffix.lower() not in UPLOAD_EXTENSIONS:
            continue  # e.g. a thumbs.db or text file that came along with a folder
        pages.append((name, upload.file.read()))
    if not pages:
        raise HTTPException(status_code=400, detail="None of the files are images (PNG, JPG, WEBP or BMP)")
    try:
        return manager.submit_upload(pages, source_lang, target_lang, title.strip()).to_dict()
    except ValueError as exc:
        _bad_request(exc)


@app.get("/api/jobs/{job_id}")
def chapter_job_status(job_id: str):
    return _job_or_404(job_id).to_dict()


@app.delete("/api/jobs/{job_id}")
def cancel_chapter_job(job_id: str):
    _job_or_404(job_id)
    return manager.cancel(job_id).to_dict()


@app.get("/api/jobs/{job_id}/pages/{index}")
def chapter_job_page(job_id: str, index: int):
    path = _job_or_404(job_id).page_path(index)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="That page isn't translated yet")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/download")
def chapter_job_download(job_id: str):
    job = _job_or_404(job_id)
    if job.status != "done" or not job.pages:
        raise HTTPException(status_code=409, detail="The pages are still being translated")
    return FileResponse(build_zip(job), media_type="application/zip", filename=zip_name(job))


# Mounted last so the API routes above take precedence over static files.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
