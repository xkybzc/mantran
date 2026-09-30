"""Translate whole MangaDex chapters in the background, page by page.

Finished pages are cached in ``backend/data/chapters/<chapter>/<src>-<dst>/``,
so asking for the same chapter again is instant and an interrupted run resumes.
"""

from __future__ import annotations

import json
import queue
import re
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

CHAPTERS_DIR = DATA_DIR / "chapters"
DOWNLOAD_WORKERS = 2  # pages download ahead of translation, politely


@dataclass
class ChapterJob:
    chapter_id: str
    source_lang: str
    target_lang: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = "queued"  # queued -> running -> done | error | cancelled
    chapter: dict = field(default_factory=dict)
    total: int = 0
    pages: dict[int, str] = field(default_factory=dict)  # page index -> file name
    failed_pages: list[int] = field(default_factory=list)
    untranslated: int = 0  # text regions no backend could translate
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.chapter_id, self.source_lang, self.target_lang

    @property
    def folder(self) -> Path:
        return CHAPTERS_DIR / self.chapter_id / f"{self.source_lang}-{self.target_lang}"

    def page_path(self, index: int) -> Path | None:
        name = self.pages.get(index)
        return self.folder / name if name else None

    def to_dict(self) -> dict:
        end = self.finished or time.time()
        return {
            "id": self.id,
            "status": self.status,
            "chapter_id": self.chapter_id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "chapter": self.chapter,
            "total": self.total,
            "completed": sorted(dict(self.pages)),
            "failed_pages": list(self.failed_pages),
            "untranslated": self.untranslated,
            "error": self.error,
            "elapsed": round(end - self.started, 1) if self.started else 0,
        }


def _save_page(folder: Path, index: int, url: str, image) -> str:
    if url.lower().endswith(".png"):
        name = f"{index + 1:03d}.png"
        ok = cv2.imwrite(str(folder / name), image)
    else:
        name = f"{index + 1:03d}.jpg"
        ok = cv2.imwrite(str(folder / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Could not write {name}")
    return name


def _download(urls: list[str], index: int, chapter_id: str) -> bytes:
    try:
        return mangadex.download_page(urls[index])
    except mangadex.MangaDexError:
        # The image server may have gone away; ask MangaDex for a fresh one.
        return mangadex.download_page(mangadex.get_page_urls(chapter_id)[index])


def translate_chapter(job: ChapterJob, on_page=None) -> ChapterJob:
    """Run ``job`` to completion in the calling thread. ``on_page(job)`` fires per page."""
    check_pair(job.source_lang, job.target_lang)
    job.status, job.started = "running", time.time()

    chapter = mangadex.get_chapter(job.chapter_id)
    if not chapter["pages"]:
        raise mangadex.MangaDexError("This chapter is hosted outside MangaDex, so it can't be downloaded.")
    job.chapter = {
        "manga_id": chapter["manga_id"],
        "manga_title": chapter["manga_title"],
        "label": mangadex.chapter_label(chapter),
        "group": chapter["group"],
        "language": chapter["language"],
    }
    urls = mangadex.get_page_urls(job.chapter_id)
    job.total = len(urls)

    folder = job.folder
    folder.mkdir(parents=True, exist_ok=True)
    for path in folder.glob("[0-9][0-9][0-9].*"):
        index = int(path.stem) - 1
        if 0 <= index < job.total:
            job.pages[index] = path.name
    todo = [i for i in range(job.total) if i not in job.pages]

    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        downloads = {i: pool.submit(_download, urls, i, job.chapter_id) for i in todo}
        for i in todo:
            if job.cancel_event.is_set():
                for future in downloads.values():
                    future.cancel()
                job.status, job.finished = "cancelled", time.time()
                return job
            try:
                data = downloads[i].result()
                image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("not a readable image")
                result, metadata = translate_image(image, job.source_lang, job.target_lang)
                job.untranslated += sum(1 for r in metadata["regions"] if r["text"] and not r["translation"])
                job.pages[i] = _save_page(folder, i, urls[i], result)
            except Exception as exc:
                print(f"Page {i + 1} of chapter {job.chapter_id} failed: {exc}")
                job.failed_pages.append(i)
            if on_page:
                on_page(job)

    (folder / "chapter.json").write_text(json.dumps(job.chapter, ensure_ascii=False, indent=2), encoding="utf-8")
    job.status, job.finished = "done", time.time()
    return job


def zip_name(job: ChapterJob) -> str:
    name = f"{job.chapter.get('manga_title', 'Chapter')} - {job.chapter.get('label', job.chapter_id)} [{job.target_lang}]"
    return re.sub(r'[<>:"/\\|?*]+', "", name).strip() + ".zip"


def build_zip(job: ChapterJob) -> Path:
    """Zip the translated pages (images are already compressed, so just store them)."""
    path = job.folder / "chapter.zip"
    pages = [job.folder / job.pages[i] for i in sorted(job.pages)]
    newest = max((p.stat().st_mtime for p in pages), default=0)
    if not path.exists() or path.stat().st_mtime < newest:
        partial = path.with_suffix(".part")
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_STORED) as archive:
            for page in pages:
                archive.write(page, arcname=page.name)
        partial.replace(path)
    return path


class JobManager:
    """Runs chapter jobs one at a time on a background thread."""

    def __init__(self):
        self._jobs: dict[str, ChapterJob] = {}
        self._queue: queue.Queue[ChapterJob] = queue.Queue()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    def submit(self, chapter_id: str, source_lang: str, target_lang: str) -> ChapterJob:
        check_pair(source_lang, target_lang)
        key = (chapter_id, source_lang, target_lang)
        with self._lock:
            for job in self._jobs.values():
                if job.key == key and job.status in ("queued", "running", "done"):
                    return job
            job = ChapterJob(chapter_id, source_lang, target_lang)
            self._jobs[job.id] = job
            self._queue.put(job)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name="chapter-worker", daemon=True)
                self._worker.start()
        return job

    def get(self, job_id: str) -> ChapterJob | None:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> ChapterJob | None:
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
                translate_chapter(job)
            except Exception as exc:
                job.status, job.error, job.finished = "error", str(exc), time.time()


jobs = JobManager()
