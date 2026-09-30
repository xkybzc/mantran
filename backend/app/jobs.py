"""Translate many pages in the background: MangaDex chapters or uploaded batches.

Jobs run one at a time on a worker thread. Every page of a job shares one
translation context, so context-aware translators see the earlier pages.

MangaDex pages are cached in ``backend/data/chapters/<chapter>/<src>-<dst>/``, so
asking for the same chapter again is instant and an interrupted run resumes.
Uploaded batches live in ``backend/data/uploads/<job>/`` for a few days.
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import mangadex
from .languages import check_pair
from .ocr import DATA_DIR
from .pipeline import translate_image
from .translation import TranslationContext

CHAPTERS_DIR = DATA_DIR / "chapters"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOAD_RETENTION_SECONDS = 3 * 24 * 3600
LOAD_WORKERS = 2  # pages download/load ahead of translation, politely


@dataclass
class TranslationJob:
    source_lang: str
    target_lang: str
    kind: str = "chapter"  # "chapter" (MangaDex) or "upload"
    chapter_id: str | None = None
    title: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"  # queued -> running -> done | error | cancelled
    chapter: dict = field(default_factory=dict)  # what the reader shows: title, label, group
    total: int = 0
    pages: dict[int, str] = field(default_factory=dict)  # page index -> file name
    failed_pages: list[int] = field(default_factory=list)
    untranslated: int = 0  # text regions no backend could translate
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    upload_names: list[str] = field(default_factory=list)  # output file names, in page order
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def key(self) -> tuple:
        return self.kind, self.chapter_id or self.id, self.source_lang, self.target_lang

    @property
    def folder(self) -> Path:
        if self.kind == "chapter":
            return CHAPTERS_DIR / self.chapter_id / f"{self.source_lang}-{self.target_lang}"
        return UPLOADS_DIR / self.id / "translated"

    @property
    def source_folder(self) -> Path:
        return UPLOADS_DIR / self.id / "source"

    def page_path(self, index: int) -> Path | None:
        name = self.pages.get(index)
        return self.folder / name if name else None

    def to_dict(self) -> dict:
        end = self.finished or time.time()
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "chapter_id": self.chapter_id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "chapter": self.chapter,
            "total": self.total,
            "completed": sorted(dict(self.pages)),
            "page_names": {str(index): name for index, name in dict(self.pages).items()},
            "failed_pages": list(self.failed_pages),
            "untranslated": self.untranslated,
            "error": self.error,
            "elapsed": round(end - self.started, 1) if self.started else 0,
        }


def _output_name(name: str) -> str:
    """Translated pages keep their name; PNGs stay PNG, everything else becomes JPEG."""
    path = Path(name)
    return f"{path.stem}.png" if path.suffix.lower() == ".png" else f"{path.stem}.jpg"


def _save_page(folder: Path, name: str, image) -> str:
    name = _output_name(name)
    params = [] if name.endswith(".png") else [cv2.IMWRITE_JPEG_QUALITY, 95]
    # imencode + write_bytes rather than imwrite, which can't handle non-ASCII paths on Windows.
    ok, encoded = cv2.imencode(Path(name).suffix, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode {name}")
    (folder / name).write_bytes(encoded.tobytes())
    return name


def _translate_pages(job: TranslationJob, load, names: list[str], right_to_left: bool, on_page=None):
    """Shared page loop. ``load(i)`` returns page i's image bytes."""
    folder = job.folder
    folder.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(job.total) if i not in job.pages]
    context = TranslationContext()

    with ThreadPoolExecutor(LOAD_WORKERS) as pool:
        loads = {i: pool.submit(load, i) for i in todo}
        for i in todo:
            if job.cancel_event.is_set():
                for future in loads.values():
                    future.cancel()
                job.status, job.finished = "cancelled", time.time()
                return
            try:
                data = loads[i].result()
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("not a readable image")
                result, metadata = translate_image(
                    image, job.source_lang, job.target_lang, context=context, right_to_left=right_to_left
                )
                job.untranslated += sum(1 for r in metadata["regions"] if r.get("failed"))
                job.pages[i] = _save_page(folder, names[i], result)
            except Exception as exc:
                print(f"Page {i + 1} of job {job.id} failed: {exc}")
                job.failed_pages.append(i)
            if on_page:
                on_page(job)

    job.status, job.finished = "done", time.time()


def _download(urls: list[str], index: int, chapter_id: str) -> bytes:
    try:
        return mangadex.download_page(urls[index])
    except mangadex.MangaDexError:
        # The image server may have gone away; ask MangaDex for a fresh one.
        return mangadex.download_page(mangadex.get_page_urls(chapter_id)[index])


def translate_chapter(job: TranslationJob, on_page=None) -> TranslationJob:
    """Run a MangaDex chapter job in the calling thread. ``on_page(job)`` fires per page."""
    check_pair(job.source_lang, job.target_lang)
    job.status, job.started = "running", time.time()

    chapter = mangadex.get_chapter(job.chapter_id)
    if not chapter["pages"]:
        raise mangadex.MangaDexError("This chapter is hosted outside MangaDex, so it can't be downloaded.")
    original_language = mangadex.get_manga(chapter["manga_id"]).get("original_language") if chapter["manga_id"] else None
    job.chapter = {
        "manga_id": chapter["manga_id"],
        "manga_title": chapter["manga_title"],
        "label": mangadex.chapter_label(chapter),
        "group": chapter["group"],
        "language": chapter["language"],
    }
    urls = mangadex.get_page_urls(job.chapter_id)
    job.total = len(urls)

    # Pick up pages translated by an earlier run.
    for path in job.folder.glob("[0-9][0-9][0-9].*") if job.folder.exists() else []:
        index = int(path.stem) - 1
        if 0 <= index < job.total:
            job.pages[index] = path.name

    names = [f"{i + 1:03d}{Path(url).suffix}" for i, url in enumerate(urls)]
    # Scanlations keep the original page layout: Japanese manga read right to left.
    right_to_left = (original_language or job.source_lang) == "ja"
    _translate_pages(job, lambda i: _download(urls, i, job.chapter_id), names, right_to_left, on_page)
    if job.status == "done":
        (job.folder / "chapter.json").write_text(json.dumps(job.chapter, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


def translate_upload(job: TranslationJob, on_page=None) -> TranslationJob:
    """Run an uploaded batch in the calling thread."""
    check_pair(job.source_lang, job.target_lang)
    job.status, job.started = "running", time.time()
    sources = sorted(job.source_folder.iterdir())
    job.total = len(sources)
    _translate_pages(job, lambda i: sources[i].read_bytes(), job.upload_names, job.source_lang == "ja", on_page)
    shutil.rmtree(job.source_folder, ignore_errors=True)
    return job


def _safe_stem(name: str) -> str:
    stem = Path(name.replace("\\", "/")).stem
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", stem).strip(" .")


def zip_name(job: TranslationJob) -> str:
    if job.kind == "chapter":
        name = f"{job.chapter.get('manga_title', 'Chapter')} - {job.chapter.get('label', job.chapter_id)}"
    else:
        name = job.title or "Translated pages"
    return re.sub(r'[<>:"/\\|?*]+', "", f"{name} [{job.target_lang}]").strip() + ".zip"


def build_zip(job: TranslationJob) -> Path:
    """Zip the translated pages (images are already compressed, so just store them)."""
    path = job.folder / "pages.zip"
    pages = [job.folder / job.pages[i] for i in sorted(job.pages)]
    newest = max((p.stat().st_mtime for p in pages), default=0)
    if not path.exists() or path.stat().st_mtime < newest:
        partial = path.with_suffix(".part")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as archive:
            for page in pages:
                archive.write(page, arcname=page.name)
        partial.replace(path)
    return path


def remove_old_uploads(max_age=UPLOAD_RETENTION_SECONDS):
    if not UPLOADS_DIR.exists():
        return
    cutoff = time.time() - max_age
    for folder in UPLOADS_DIR.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)


def new_upload_job(files: list[tuple[str, bytes]], source_lang: str, target_lang: str,
                   title: str = "") -> TranslationJob:
    """Store uploaded pages, given as ``(file name, image bytes)`` in reading order, as a job."""
    check_pair(source_lang, target_lang)
    if not files:
        raise ValueError("No pages were uploaded")
    remove_old_uploads()
    job = TranslationJob(source_lang, target_lang, kind="upload", title=title or "Translated pages")
    job.source_folder.mkdir(parents=True)
    used: set[str] = set()
    for index, (name, data) in enumerate(files):
        stem = _safe_stem(name) or f"page_{index + 1:03d}"
        unique, n = stem, 2
        while unique.lower() in used:
            unique, n = f"{stem}_{n}", n + 1
        used.add(unique.lower())
        suffix = Path(name).suffix.lower() or ".png"
        (job.source_folder / f"{index:04d}{suffix}").write_bytes(data)
        job.upload_names.append(unique + suffix)
    job.total = len(files)
    pages = "page" if len(files) == 1 else "pages"
    job.chapter = {"manga_title": job.title, "label": f"{len(files)} {pages}"}
    return job


def natural_key(name: str):
    """Sort "page2" before "page10"."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


class JobManager:
    """Runs jobs one at a time on a background thread."""

    def __init__(self):
        self._jobs: dict[str, TranslationJob] = {}
        self._queue: queue.Queue[TranslationJob] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def _enqueue(self, job: TranslationJob) -> TranslationJob:
        self._jobs[job.id] = job
        self._queue.put(job)
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, name="translation-worker", daemon=True)
            self._worker.start()
        return job

    def submit(self, chapter_id: str, source_lang: str, target_lang: str) -> TranslationJob:
        """Queue a MangaDex chapter (or return the job already doing it)."""
        check_pair(source_lang, target_lang)
        key = ("chapter", chapter_id, source_lang, target_lang)
        with self._lock:
            for job in self._jobs.values():
                if job.key == key and job.status in ("queued", "running", "done"):
                    return job
            return self._enqueue(TranslationJob(source_lang, target_lang, chapter_id=chapter_id))

    def submit_upload(self, files: list[tuple[str, bytes]], source_lang: str, target_lang: str,
                      title: str = "") -> TranslationJob:
        """Queue uploaded pages, given as ``(file name, image bytes)`` in reading order."""
        job = new_upload_job(files, source_lang, target_lang, title)
        with self._lock:
            return self._enqueue(job)

    def get(self, job_id: str) -> TranslationJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> TranslationJob | None:
        job = self._jobs.get(job_id)
        if job is not None:
            job.cancel_event.set()
            if job.status == "queued":
                job.status = "cancelled"
        return job

    def _work(self):
        while True:
            job = self._queue.get()
            if job.cancel_event.is_set():
                job.status = "cancelled"
                continue
            try:
                (translate_chapter if job.kind == "chapter" else translate_upload)(job)
            except Exception as exc:
                job.status, job.error, job.finished = "error", str(exc), time.time()


manager = JobManager()
