"""Translate recognised text, trying several backends in turn.

Backends, in the default order:
  google    - Google Translate's free web endpoint (fast, good; rate limited per IP)
  mymemory  - MyMemory (5,000 chars/day, 50,000 with MYMEMORY_EMAIL set)
  local     - NLLB-200 running on this machine (offline and unlimited; ~2.5 GB
              download on first use, and slower)

Set MANTRAN_TRANSLATORS (e.g. "local" or "google,local") to change the order.
"""

from __future__ import annotations

import html
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from .languages import Language, get_language

DEFAULT_BACKENDS = "google,mymemory,local"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
# A backend that errors out is skipped for this long, so one rate limit
# doesn't cost a network round trip for every remaining bubble.
BACKEND_COOLDOWN_SECONDS = 300
QUOTA_COOLDOWN_SECONDS = 3600
MYMEMORY_PARALLEL_REQUESTS = 4

_cache: dict[tuple[str, str, str], str] = {}
_cooldown_until: dict[str, float] = {}
_local_lock = threading.Lock()


class QuotaExceeded(Exception):
    pass


def _google(texts, src: Language, dst: Language):
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=src.translator, target=dst.translator)
    if len(texts) > 1:
        # One request per page instead of one per bubble; fall back if the
        # line structure didn't survive.
        parts = translator.translate("\n".join(texts)).split("\n")
        if len(parts) == len(texts):
            return parts
    return [translator.translate(text) for text in texts]


def _mymemory(texts, src: Language, dst: Language):
    from translate import Translator

    def one(text):
        translator = Translator(
            from_lang=src.translator, to_lang=dst.translator, email=os.environ.get("MYMEMORY_EMAIL", "")
        )
        return translator.translate(text)

    # MyMemory has no batch call, so send a page's bubbles a few at a time.
    with ThreadPoolExecutor(max_workers=MYMEMORY_PARALLEL_REQUESTS) as pool:
        results = list(pool.map(one, texts))
    over_quota = ["MYMEMORY WARNING" in result.upper() for result in results]
    if all(over_quota):
        raise QuotaExceeded(results[0])
    if any(over_quota):
        _cooldown_until["mymemory"] = time.monotonic() + QUOTA_COOLDOWN_SECONDS
    return [None if bad else result for result, bad in zip(results, over_quota)]


@lru_cache(maxsize=1)
def _nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading local translation model {NLLB_MODEL} (downloads ~2.5 GB the first time)")
    return AutoTokenizer.from_pretrained(NLLB_MODEL), AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL).eval()


def _local(texts, src: Language, dst: Language):
    import torch

    tokenizer, model = _nllb()
    with _local_lock:  # the tokenizer's source language is shared state
        tokenizer.src_lang = src.nllb
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            output = model.generate(
                **batch, forced_bos_token_id=tokenizer.convert_tokens_to_ids(dst.nllb), num_beams=3
            )
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    # NLLB tends to invent whole sentences for interjections ("Paaapiii!"):
    # drop outputs far longer than their source.
    return [
        result if len(result.split()) <= 2 * len(text.split()) + 2 else None
        for text, result in zip(texts, results)
    ]


BACKENDS = {"google": _google, "mymemory": _mymemory, "local": _local}


def backend_order() -> list[str]:
    names = os.environ.get("MANTRAN_TRANSLATORS", DEFAULT_BACKENDS).split(",")
    return [name.strip() for name in names if name.strip() in BACKENDS]


def _sentence_case(text: str) -> str:
    """Scanlations letter in ALL CAPS, which translators handle poorly."""
    letters = [ch for ch in text if ch.isalpha()]
    if len(letters) < 4 or sum(ch.isupper() for ch in letters) < 0.8 * len(letters):
        return text
    text = text.lower()
    return re.sub(
        r"(^|[.!?…]\s+)([¿¡\"'«(\-]*)(\w)",
        lambda m: m.group(1) + m.group(2) + m.group(3).upper(),
        text,
    )


def prepare_source(text: str, language: Language) -> str:
    """Tidy OCR output before translation."""
    text = (text or "").strip()
    if language.code == "ja":
        text = text.replace("．．．", "…").replace("...", "…")
        text = re.sub(r"[ー〜~]{2,}", "ー", text)
        return text
    if language.code.startswith("es"):
        # Tesseract reads the opening "¡" as "i" (e.g. "iHola" or "¿iQué").
        text = re.sub(r"(^|[\s¿\"'(])i(?=[A-ZÁÉÍÓÚÑ])", r"\1¡", text)
    text = _sentence_case(text)
    if language.code == "en":
        text = re.sub(r"\bi\b", "I", text)
    return text


def _clean_result(result) -> str:
    return " ".join(html.unescape(result or "").split())


def translate_texts(texts, src="ja", dst="en") -> list[str | None]:
    """Translate a batch (e.g. every bubble of a page).

    Returns one entry per input: the translation, "" for empty input, or
    ``None`` where no backend could translate it.
    """
    source, target = get_language(src), get_language(dst)
    prepared = [prepare_source(text, source) for text in texts]
    results: list[str | None] = [None] * len(prepared)
    pending = []
    for i, text in enumerate(prepared):
        key = (text, source.translator, target.translator)
        if not text:
            results[i] = ""
        elif key in _cache:
            results[i] = _cache[key]
        else:
            pending.append(i)

    for name in backend_order():
        if not pending:
            break
        if _cooldown_until.get(name, 0) > time.monotonic():
            continue
        try:
            outputs = BACKENDS[name]([prepared[i] for i in pending], source, target)
        except ImportError:
            _cooldown_until[name] = float("inf")
            continue
        except QuotaExceeded as exc:
            print(f"{name} translation quota used up: {exc}")
            _cooldown_until[name] = time.monotonic() + QUOTA_COOLDOWN_SECONDS
            continue
        except Exception as exc:
            print(f"{name} translation failed: {exc}")
            _cooldown_until[name] = time.monotonic() + BACKEND_COOLDOWN_SECONDS
            continue

        still_pending = []
        for i, output in zip(pending, outputs):
            output = _clean_result(output)
            if output:
                results[i] = output
                _cache[(prepared[i], source.translator, target.translator)] = output
            else:
                still_pending.append(i)
        pending = still_pending

    if pending:
        print(f"No translation backend could translate {len(pending)} text(s).")
    return results


def translate_text(text, src="ja", dst="en") -> str | None:
    """Translate one string; returns ``None`` if every backend fails."""
    return translate_texts([text], src, dst)[0]
