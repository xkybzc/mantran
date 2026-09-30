"""Read the original text out of detected text regions.

Japanese uses manga-ocr. Every other language uses Tesseract, whose language
data is downloaded on first use into ``backend/data/tessdata``.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from .detection import Box, TextRegion, pad_box
from .languages import Language, get_language

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TESSDATA_DIR = DATA_DIR / "tessdata"
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/main/{name}.traineddata"
# Mean word confidence below this is treated as noise (sound effects, art).
# Text outside bubbles needs more: real captions score 80+, misread art under 60.
MIN_CONFIDENCE = 45
FREE_TEXT_MIN_CONFIDENCE = 70

_download_lock = threading.Lock()


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
    # An env var rather than --tessdata-dir: pytesseract mangles quoted paths on Windows.
    TESSDATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
    return pytesseract


def ensure_traineddata(name: str) -> Path:
    """Download Tesseract's model for ``name`` (e.g. "spa") unless it's already there."""
    path = TESSDATA_DIR / f"{name}.traineddata"
    if path.exists():
        return path
    with _download_lock:
        if not path.exists():
            TESSDATA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Downloading Tesseract language data: {name}")
            response = requests.get(TESSDATA_URL.format(name=name), timeout=300)
            response.raise_for_status()
            partial = path.with_suffix(".part")
            partial.write_bytes(response.content)
            partial.replace(path)
    return path


def _crop(image, box: Box, interior=None):
    """Crop ``box`` (slightly padded); pixels outside the bubble's inside become white."""
    x0, y0, x1, y1 = pad_box(box, 2, image.shape)
    crop = image[y0:y1, x0:x1].copy()
    if interior is not None:
        bx0, by0, _, _ = interior.box
        inside = np.zeros(crop.shape[:2], dtype=bool)
        sub = interior.mask[max(0, y0 - by0):y1 - by0, max(0, x0 - bx0):x1 - bx0]
        oy, ox = max(0, by0 - y0), max(0, bx0 - x0)
        inside[oy:oy + sub.shape[0], ox:ox + sub.shape[1]] = sub
        crop[~inside] = 255
    return crop


def _prepare_for_tesseract(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Tesseract reads best with glyphs ~30px tall; scanlation lettering is often smaller.
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() < 127:  # light text on a dark background
        binary = 255 - binary
    return cv2.copyMakeBorder(binary, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)


def _join_lines(lines: list[str], separator: str) -> str:
    text = ""
    for line in lines:
        if not text:
            text = line
        elif separator and len(text) > 1 and text.endswith("-") and text[-2].isalpha() and line[:1].isalpha():
            text = text[:-1] + line  # re-join a word hyphenated across lines
        else:
            text = f"{text}{separator}{line}"
    return text


def clean_ocr_text(text: str) -> str:
    text = re.sub(r"[|_\\\[\]{}<>~=]+", " ", text)
    return " ".join(text.split())


def _read_with_tesseract(crop, language: Language, vertical: bool, min_confidence: float) -> str:
    tesseract = _tesseract()
    model = language.tesseract_vertical if vertical and language.tesseract_vertical else language.tesseract
    ensure_traineddata(model)

    data = tesseract.image_to_data(
        _prepare_for_tesseract(crop),
        lang=model,
        config="--psm 5" if model.endswith("_vert") else "--psm 6",
        output_type=tesseract.Output.DICT,
    )
    lines: dict[tuple, list[str]] = {}
    weighted, letters = 0.0, 0
    for i, word in enumerate(data["text"]):
        word, confidence = word.strip(), float(data["conf"][i])
        if not word or confidence < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        weighted += confidence * len(word)
        letters += len(word)

    if not letters or weighted / letters < min_confidence:
        return ""
    joiner = language.word_separator
    text = clean_ocr_text(_join_lines([joiner.join(words) for words in lines.values()], joiner or ""))
    return text if sum(ch.isalpha() for ch in text) >= 2 else ""


def read_box(image, box: Box, lang="ja", interior=None, min_confidence=MIN_CONFIDENCE) -> str:
    language = get_language(lang)
    crop = _crop(image, box, interior)
    if crop.size == 0:
        return ""

    if language.tesseract is None:
        try:
            return _manga_ocr()(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).strip()
        except Exception as exc:
            print(f"manga-ocr failed ({exc}); falling back to Tesseract.")
            language = replace(language, tesseract="jpn")

    height, width = crop.shape[:2]
    try:
        return _read_with_tesseract(crop, language, height > 1.5 * width, min_confidence)
    except Exception as exc:
        print(f"Tesseract OCR failed: {exc}")
        return ""


def read_region(image, region: TextRegion, lang="ja", interior=None) -> str:
    """OCR every text box of a region in reading order and join the results."""
    separator = get_language(lang).word_separator
    min_confidence = MIN_CONFIDENCE if region.bubble_box is not None else FREE_TEXT_MIN_CONFIDENCE
    parts = (read_box(image, box, lang, interior, min_confidence) for box in region.text_boxes)
    return separator.join(part for part in parts if part)
