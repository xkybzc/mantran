import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import jobs, mangadex, translation
from app.detection import TextRegion, reading_order
from app.languages import check_pair, get_language
from app.ocr import _join_lines, clean_ocr_text, correct_ocr_words, fix_spanish_marks

CHAPTER_ID = "466845e4-2a29-43da-a073-71845caf1d71"


@pytest.fixture
def fresh_translation(monkeypatch):
    monkeypatch.setattr(translation, "_cache", {})
    monkeypatch.setattr(translation, "_cooldown_until", {})


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


def test_prepare_source_fixes_all_caps():
    assert translation.prepare_source("NO ME RINDO. ¡VAMOS!", get_language("es")) == "No me rindo. ¡Vamos!"
    assert translation.prepare_source("WHERE AM i GOING?", get_language("en")) == "Where am I going?"


def test_spanish_opening_marks_are_restored():
    assert fix_spanish_marks("iHola! ¿iQué pasa? Sí, iba") == "¡Hola! ¿¡Qué pasa? Sí, iba"


def test_ocr_misreads_are_corrected_to_common_words():
    spanish = get_language("es-la")
    assert correct_ocr_words("Lo hicimos duntas.", spanish) == "Lo hicimos juntas."  # one misread beats two
    assert correct_ocr_words("Una ovesa roja", spanish) == "Una oveja roja"
    assert correct_ocr_words("Para que se £ormen", spanish) == "Para que se formen"
    assert correct_ocr_words("EL SUEGO", spanish) == "EL JUEGO"
    # Names and real words are left alone.
    assert correct_ocr_words("Ena y Kyary cantan", spanish) == "Ena y Kyary cantan"
    # Languages without Latin-script confusions are untouched.
    assert correct_ocr_words("duntas", get_language("ja")) == "duntas"


def test_ocr_lines_are_joined_and_dehyphenated():
    assert _join_lines(["I CAN'T BE-", "LIEVE IT", "- REALLY"], " ") == "I CAN'T BELIEVE IT - REALLY"
    assert _join_lines(["你好", "世界"], "") == "你好世界"
    assert clean_ocr_text("| HELLO __ THERE ]") == "HELLO THERE"


def test_reading_order_follows_rows_and_direction():
    left_top = TextRegion([(10, 10, 60, 60)])
    right_top = TextRegion([(200, 20, 260, 70)])
    bottom = TextRegion([(100, 200, 160, 260)])
    regions = [bottom, left_top, right_top]
    assert reading_order(regions, right_to_left=True) == [right_top, left_top, bottom]
    assert reading_order(regions, right_to_left=False) == [left_top, right_top, bottom]


def test_translate_texts_batches_caches_and_falls_back(monkeypatch, fresh_translation):
    calls = []

    def broken(texts, src, dst, context=None):
        raise RuntimeError("rate limited")

    def fake(texts, src, dst, context=None):
        calls.append(list(texts))
        return [None if text == "skip" else "" if text == "noise" else f"<{text}>" for text in texts]

    monkeypatch.setattr(translation, "BACKENDS", {"first": broken, "second": fake})
    monkeypatch.setenv("MANTRAN_TRANSLATORS", "first,second")

    result = translation.translate_texts(["uno", "", "skip", "noise", "dos"], "es", "en")
    assert result == ["<uno>", "", None, "", "<dos>"]  # None failed; "" is a deliberate skip
    assert calls == [["uno", "skip", "noise", "dos"]]  # one batch for the whole page
    assert translation._cooldown_until["first"] > 0  # the failing backend is skipped for a while

    # Cached results don't hit the backend again.
    assert translation.translate_texts(["uno"], "es", "en") == ["<uno>"]
    assert len(calls) == 1


def test_claude_backend_is_skipped_without_credentials(monkeypatch, fresh_translation):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("MANTRAN_TRANSLATORS", raising=False)
    monkeypatch.setattr(translation, "BACKENDS", {**translation.BACKENDS, "google": lambda t, s, d, c=None: ["hi"]})
    assert translation.translate_texts(["hola"], "es", "en") == ["hi"]
    assert translation._cooldown_until["claude"] == float("inf")


def test_claude_backend_sends_page_with_context_and_reads_translations(monkeypatch, fresh_translation):
    import anthropic
    import httpx2

    sent = {}

    def handler(request):
        sent["headers"] = dict(request.headers)
        sent["body"] = json.loads(request.content)
        answer = {
            "translations": [{"id": 1, "text": "We did it together."}, {"id": 2, "text": ""}],
            "names": [{"source": "Ena", "target": "Ena"}],
        }
        return httpx2.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": translation.CLAUDE_MODEL,
            "content": [{"type": "text", "text": json.dumps(answer)}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 20},
        })

    client = anthropic.Anthropic(api_key="test-key", http_client=anthropic.DefaultHttpxClient(
        transport=httpx2.MockTransport(handler)))
    monkeypatch.setattr(translation, "_claude_client", lambda: client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    context = translation.TranslationContext()
    context.add(["¿Quién lo hizo?"], ["Who did it?"])
    result = translation._claude(["Lo hicimos juntas.", "~~ %%"], get_language("es-la"), get_language("vi"), context)

    assert result == ["We did it together.", ""]
    assert context.names == {"Ena": "Ena"}
    body = sent["body"]
    assert body["model"] == translation.CLAUDE_MODEL
    assert body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in sent["headers"]["anthropic-beta"]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "Vietnamese pronouns" in body["system"]
    prompt = body["messages"][0]["content"]
    assert "Who did it?" in prompt and '"id": 1' in prompt and "Lo hicimos juntas." in prompt


def _fake_translate_image(image, src, dst, context=None, right_to_left=None):
    return image, {"regions": [
        {"text": "hola", "translation": "hello", "failed": False},
        {"text": "x", "translation": "", "failed": True},
        {"text": "noise", "translation": "", "failed": False},
    ]}


def test_chapter_job_translates_every_page_and_zips(monkeypatch, tmp_path):
    page = cv2.imencode(".jpg", np.full((40, 30, 3), 200, dtype=np.uint8))[1].tobytes()
    urls = [f"https://node.example/data/hash/{n}.jpg" for n in ("a", "b", "c")]

    monkeypatch.setattr(jobs, "CHAPTERS_DIR", tmp_path)
    monkeypatch.setattr(mangadex, "get_chapter", lambda chapter_id: {
        "id": chapter_id, "manga_id": "m1", "manga_title": "Test Manga", "volume": "1", "chapter": "2",
        "title": "Start", "language": "es-la", "pages": 3, "group": "Group", "external_url": None,
    })
    monkeypatch.setattr(mangadex, "get_manga", lambda manga_id: {"original_language": "ja"})
    monkeypatch.setattr(mangadex, "get_page_urls", lambda chapter_id: urls)

    def download(url):
        if url.endswith("b.jpg"):
            raise mangadex.MangaDexError("node down")
        return page

    monkeypatch.setattr(mangadex, "download_page", download)
    seen_directions = []

    def fake(image, src, dst, context=None, right_to_left=None):
        seen_directions.append(right_to_left)
        return _fake_translate_image(image, src, dst)

    monkeypatch.setattr(jobs, "translate_image", fake)

    progress = []
    job = jobs.translate_chapter(jobs.TranslationJob("es-la", "en", chapter_id=CHAPTER_ID), on_page=progress.append)

    assert job.status == "done"
    assert job.total == 3
    assert sorted(job.pages) == [0, 2]
    assert job.failed_pages == [1]
    assert job.untranslated == 2  # one failed region on each translated page; the skipped noise doesn't count
    assert len(progress) == 3  # progress is reported for failed pages too
    assert seen_directions == [True, True]  # a Japanese manga reads right to left
    assert job.chapter["label"] == "Vol. 1 Ch. 2 - Start"

    with zipfile.ZipFile(jobs.build_zip(job)) as archive:
        assert archive.namelist() == ["001.jpg", "003.jpg"]
    assert jobs.zip_name(job) == "Test Manga - Vol. 1 Ch. 2 - Start [en].zip"

    # Running again reuses the cached pages and only retries the failed one.
    retried = jobs.translate_chapter(jobs.TranslationJob("es-la", "en", chapter_id=CHAPTER_ID))
    assert sorted(retried.pages) == [0, 2] and retried.failed_pages == [1]


def test_upload_job_keeps_order_and_names(monkeypatch, tmp_path):
    monkeypatch.setattr(jobs, "UPLOADS_DIR", tmp_path)
    monkeypatch.setattr(jobs, "translate_image", _fake_translate_image)
    png = cv2.imencode(".png", np.full((40, 30, 3), 255, dtype=np.uint8))[1].tobytes()
    jpg = cv2.imencode(".jpg", np.full((40, 30, 3), 255, dtype=np.uint8))[1].tobytes()
    files = [("chapter/page 1.png", png), ("chapter/page 2.jpg", jpg), ("other/page 1.png", png),
             ("broken.webp", b"not an image")]

    job = jobs.new_upload_job(files, "id", "vi", "Chapter 5")
    jobs.translate_upload(job)

    assert job.status == "done"
    assert [job.pages[i] for i in sorted(job.pages)] == ["page 1.png", "page 2.jpg", "page 1_2.png"]
    assert job.failed_pages == [3]
    assert not job.source_folder.exists()  # originals are removed once translated
    with zipfile.ZipFile(jobs.build_zip(job)) as archive:
        assert archive.namelist() == ["page 1.png", "page 2.jpg", "page 1_2.png"]
    assert jobs.zip_name(job) == "Chapter 5 [vi].zip"


def test_natural_sort_puts_page_2_before_page_10():
    names = ["page10.jpg", "page2.jpg", "Page1.jpg"]
    assert sorted(names, key=jobs.natural_key) == ["Page1.jpg", "page2.jpg", "page10.jpg"]
