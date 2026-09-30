import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import chapters, mangadex, translation
from app.languages import check_pair, get_language
from app.ocr import _join_lines, clean_ocr_text

CHAPTER_ID = "466845e4-2a29-43da-a073-71845caf1d71"


def test_parse_link_accepts_titles_chapters_and_bare_ids():
    assert mangadex.parse_link(f"https://mangadex.org/title/{CHAPTER_ID}/some-slug") == ("manga", CHAPTER_ID)
    assert mangadex.parse_link(f"https://mangadex.org/chapter/{CHAPTER_ID}/3") == ("chapter", CHAPTER_ID)
    assert mangadex.parse_link(f"  {CHAPTER_ID.upper()} ") == ("unknown", CHAPTER_ID)
    with pytest.raises(mangadex.MangaDexError):
        mangadex.parse_link("https://example.com/title/123")


def test_check_pair_rejects_same_language_and_other_targets():
    check_pair("es-la", "en")
    check_pair("en", "vi")
    with pytest.raises(ValueError):
        check_pair("en", "en")
    with pytest.raises(ValueError):
        check_pair("ja", "fr")
    with pytest.raises(ValueError):
        check_pair("xx", "en")


def test_prepare_source_fixes_spanish_marks_and_all_caps():
    spanish = get_language("es-la")
    assert translation.prepare_source("iHola! ¿iQué pasa?", spanish) == "¡Hola! ¿¡Qué pasa?"
    assert translation.prepare_source("NO ME RINDO. ¡VAMOS!", spanish) == "No me rindo. ¡Vamos!"
    assert translation.prepare_source("WHERE AM i GOING?", get_language("en")) == "Where am I going?"


def test_ocr_lines_are_joined_and_dehyphenated():
    assert _join_lines(["I CAN'T BE-", "LIEVE IT", "- REALLY"], " ") == "I CAN'T BELIEVE IT - REALLY"
    assert _join_lines(["你好", "世界"], "") == "你好世界"
    assert clean_ocr_text("| HELLO __ THERE ]") == "HELLO THERE"


def test_translate_texts_batches_caches_and_falls_back(monkeypatch):
    calls = []

    def broken(texts, src, dst):
        raise RuntimeError("rate limited")

    def fake(texts, src, dst):
        calls.append(list(texts))
        return [f"<{text}>" if text != "skip" else None for text in texts]

    monkeypatch.setattr(translation, "BACKENDS", {"first": broken, "second": fake})
    monkeypatch.setenv("MANTRAN_TRANSLATORS", "first,second")
    monkeypatch.setattr(translation, "_cache", {})
    monkeypatch.setattr(translation, "_cooldown_until", {})

    assert translation.translate_texts(["uno", "", "skip", "dos"], "es", "en") == ["<uno>", "", None, "<dos>"]
    assert calls == [["uno", "skip", "dos"]]  # one batch for the whole page
    assert translation._cooldown_until["first"] > 0  # the failing backend is skipped for a while

    # Cached results don't hit the backend again.
    assert translation.translate_texts(["uno"], "es", "en") == ["<uno>"]
    assert len(calls) == 1


def test_chapter_job_translates_every_page_and_zips(monkeypatch, tmp_path):
    page = cv2.imencode(".jpg", np.full((40, 30, 3), 200, dtype=np.uint8))[1].tobytes()
    urls = [f"https://node.example/data/hash/{n}.jpg" for n in ("a", "b", "c")]

    monkeypatch.setattr(chapters, "CHAPTERS_DIR", tmp_path)
    monkeypatch.setattr(mangadex, "get_chapter", lambda chapter_id: {
        "id": chapter_id, "manga_id": "m1", "manga_title": "Test Manga", "volume": "1", "chapter": "2",
        "title": "Start", "language": "es-la", "pages": 3, "group": "Group", "external_url": None,
    })
    monkeypatch.setattr(mangadex, "get_page_urls", lambda chapter_id: urls)

    def download(url):
        if url.endswith("b.jpg"):
            raise mangadex.MangaDexError("node down")
        return page

    monkeypatch.setattr(mangadex, "download_page", download)
    monkeypatch.setattr(chapters, "translate_image", lambda image, src, dst: (
        image, {"regions": [{"text": "hola", "translation": "hello"}, {"text": "x", "translation": ""}]}
    ))

    progress = []
    job = chapters.translate_chapter(chapters.ChapterJob(CHAPTER_ID, "es-la", "en"), on_page=progress.append)

    assert job.status == "done"
    assert job.total == 3
    assert sorted(job.pages) == [0, 2]
    assert job.failed_pages == [1]
    assert job.untranslated == 2
    assert len(progress) == 3  # progress is reported for failed pages too
    assert job.chapter["label"] == "Vol. 1 Ch. 2 - Start"

    with zipfile.ZipFile(chapters.build_zip(job)) as archive:
        assert archive.namelist() == ["001.jpg", "003.jpg"]
    assert chapters.zip_name(job) == "Test Manga - Vol. 1 Ch. 2 - Start [en].zip"

    # Running again reuses the cached pages and only retries the failed one.
    retried = chapters.translate_chapter(chapters.ChapterJob(CHAPTER_ID, "es-la", "en"))
    assert sorted(retried.pages) == [0, 2] and retried.failed_pages == [1]
