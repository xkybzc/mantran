"""Translate recognised text, trying several free backends in turn."""

from __future__ import annotations

import html
import re
import time

# A backend that errors out is skipped for this long, so one rate limit
# doesn't cost a network round trip for every remaining bubble.
BACKEND_COOLDOWN_SECONDS = 300

_cache: dict[tuple[str, str, str], str] = {}
_cooldown_until: dict[str, float] = {}


def _google(text, src, dst):
    from deep_translator import GoogleTranslator

    return GoogleTranslator(source=src, target=dst).translate(text)


def _mymemory(text, src, dst):
    from translate import Translator

    result = Translator(from_lang=src, to_lang=dst).translate(text)
    if "MYMEMORY WARNING" in result.upper():
        raise RuntimeError(result)
    return result


BACKENDS = (("google", _google), ("mymemory", _mymemory))


def normalize_source(text: str) -> str:
    """Tidy manga-ocr output (full-width punctuation, dotted ellipses)."""
    text = text.replace("．．．", "…").replace("...", "…")
    text = re.sub(r"[ー〜~]{2,}", "ー", text)
    return text.strip()


def translate_text(text, src="ja", dst="en") -> str | None:
    """Translate ``text``; returns ``None`` if every backend fails."""
    text = normalize_source(text or "")
    if not text:
        return ""

    key = (text, src, dst)
    if key in _cache:
        return _cache[key]

    for name, backend in BACKENDS:
        if _cooldown_until.get(name, 0) > time.monotonic():
            continue
        try:
            result = backend(text, src, dst)
        except ImportError:
            _cooldown_until[name] = float("inf")
            continue
        except Exception as exc:
            print(f"{name} translation failed: {exc}")
            _cooldown_until[name] = time.monotonic() + BACKEND_COOLDOWN_SECONDS
            continue
        result = " ".join(html.unescape(result or "").split())
        if result:
            _cache[key] = result
            return result

    print("No translation backend is available.")
    return None
