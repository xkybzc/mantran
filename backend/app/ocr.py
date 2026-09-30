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


# Letters Tesseract mixes up in comic lettering: misread -> what it usually was.
_CONFUSIONS = {
    "d": "j", "s": "j", "i": "lj", "l": "i", "1": "li", "0": "o", "5": "s", "c": "e", "e": "c",
    "n": "h", "h": "n", "u": "v", "v": "u", "o": "a", "a": "o", "t": "f", "f": "t", "£": "f", "€": "e",
}
_CONFUSIONS.update({k.upper(): v.upper() for k, v in list(_CONFUSIONS.items()) if k.isalpha()})
_CONFUSIONS["S"] += "j"  # a misread capital S is often a lowercase j ("Suego" -> "juego")
_MULTI_CONFUSIONS = (("rn", "m"), ("cl", "d"), ("vv", "w"), ("RN", "M"), ("CL", "D"), ("VV", "W"))
# Latin-script languages with a word-frequency list to check corrections against.
_WORD_FIX_LANGS = {"en", "es", "pt", "id", "fr", "it", "de", "tr", "pl"}
KNOWN_WORD_ZIPF = 2.0      # at least this common: leave the word alone
CORRECTION_MIN_ZIPF = 3.0  # a replacement must be at least this common


def _variants_by_edits(word: str, max_edits=2) -> list[set[str]]:
    """Candidate spellings, grouped by how many misreads they undo (1, then 2)."""
    seen, frontier, levels = {word}, {word}, []
    for edit in range(max_edits):
        found = set()
        for candidate in frontier:
            for i, ch in enumerate(candidate):
                for alt in _CONFUSIONS.get(ch, ""):
                    found.add(candidate[:i] + alt + candidate[i + 1:])
            for bad, good in _MULTI_CONFUSIONS:
                start = candidate.find(bad)
                while start != -1:
                    found.add(candidate[:start] + good + candidate[start + len(bad):])
                    start = candidate.find(bad, start + 1)
            if edit == 0 and len(candidate) >= 5:
                found.add(candidate[1:])  # a stray mark read as a leading letter
        frontier = found - seen
        seen |= frontier
        levels.append(frontier)
    return levels


def fix_spanish_marks(text: str) -> str:
    """Tesseract reads the opening "¡" as "i" (e.g. "iHola" or "¿iQué")."""
    return re.sub(r"(^|[\s¿\"'(])i(?=[A-ZÁÉÍÓÚÑ])", r"\1¡", text)


def correct_ocr_words(text: str, language: Language) -> str:
    """Fix words that are unknown in the language but one or two typical misreads
    away from a common word (e.g. Spanish "duntas" -> "juntas")."""
    lang = language.translator
    if lang not in _WORD_FIX_LANGS:
        return text
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return text

    def fix(match: re.Match) -> str:
        word = match.group(0)
        if len(word) < 3:
            return word
        original = zipf_frequency(word.lower(), lang)
        if original >= KNOWN_WORD_ZIPF:
            return word
        # Prefer undoing one misread over two ("duntas" -> "juntas", not "juntos").
        for level in _variants_by_edits(word):
            scored = [(zipf_frequency(v.lower(), lang), v) for v in level if v.isalpha()]
            best_score, best = max(scored, default=(0.0, word))
            if best_score >= CORRECTION_MIN_ZIPF and best_score - original >= 1.5:
                break
        else:
            return word
        if word.isupper():
            return best.upper()
        return best[0].upper() + best[1:] if word[0].isupper() and best[0].isalpha() else best

    return re.sub(r"[\w£€]+", fix, text)


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
    if sum(ch.isalpha() for ch in text) < 2:
        return ""
    if language.code.startswith("es"):
        text = fix_spanish_marks(text)
    return correct_ocr_words(text, language)


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
