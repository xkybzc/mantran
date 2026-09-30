"""Read the original text out of detected text regions."""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
from PIL import Image

from .detection import Box, TextRegion, pad_box

TESSERACT_LANGS = {"ja": "jpn_vert+jpn", "en": "eng", "ko": "kor", "zh-CN": "chi_sim", "zh-TW": "chi_tra"}


@lru_cache(maxsize=1)
def _manga_ocr():
    from manga_ocr import MangaOcr

    return MangaOcr()


@lru_cache(maxsize=1)
def _tesseract():
    import pytesseract

    if os.name == "nt":
        for candidate in (
            os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        ):
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
    return pytesseract


def read_box(image, box: Box, lang="ja") -> str:
    x0, y0, x1, y1 = pad_box(box, 2, image.shape)
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return ""

    if lang == "ja":
        try:
            return _manga_ocr()(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).strip()
        except Exception as exc:
            print(f"manga-ocr failed ({exc}); falling back to Tesseract.")

    try:
        vertical = lang == "ja" and (y1 - y0) > (x1 - x0)
        text = _tesseract().image_to_string(
            crop, lang=TESSERACT_LANGS.get(lang, "eng"), config="--psm 5" if vertical else "--psm 6"
        )
    except Exception as exc:
        print(f"Tesseract OCR failed: {exc}")
        return ""
    return " ".join(text.split())


def read_region(image, region: TextRegion, lang="ja") -> str:
    """OCR every text box of a region in reading order and join the results."""
    separator = "" if lang in ("ja", "zh-CN", "zh-TW") else " "
    parts = (read_box(image, box, lang) for box in region.text_boxes)
    return separator.join(part for part in parts if part)
