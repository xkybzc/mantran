"""Translate recognised text, trying several backends in turn.

Backends, in the default order:
  claude    - Claude via the Anthropic API. Best quality: sees the whole page and
              the previous pages, fixes OCR mistakes, keeps names consistent.
              Only used when ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) is set;
              billed to that account.
  google    - Google Translate's free web endpoint (fast, good; rate limited per IP)
  mymemory  - MyMemory (5,000 chars/day, 50,000 with MYMEMORY_EMAIL set)
  local     - NLLB-200 running on this machine (offline and unlimited; ~2.5 GB
              download on first use, and slower)

Set MANTRAN_TRANSLATORS (e.g. "local" or "claude,local") to change the order.
"""

from __future__ import annotations

import html
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

from .languages import Language, get_language

DEFAULT_BACKENDS = "claude,google,mymemory,local"
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
CLAUDE_MODEL = os.environ.get("MANTRAN_CLAUDE_MODEL", "claude-opus-5")
# Bubble translation is short, well-specified work; medium keeps a chapter fast.
CLAUDE_EFFORT = os.environ.get("MANTRAN_CLAUDE_EFFORT", "medium")
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


class BackendUnavailable(Exception):
    """The backend can't be used at all in this setup (not installed, no credentials)."""


@dataclass
class TranslationContext:
    """What earlier pages of the same chapter said, for backends that use context."""

    history: list[tuple[str, str]] = field(default_factory=list)  # (source, translation)
    names: dict[str, str] = field(default_factory=dict)
    max_history: int = 40

    def add(self, sources, translations):
        for source, translation in zip(sources, translations):
            if source and translation:
                self.history.append((source, translation))
        del self.history[:-self.max_history]


# --- Backends: (texts, source, target, context) -> list of str | None -----------------
# None means "couldn't translate this one, try the next backend"; "" means "this
# isn't real text (OCR noise), leave the bubble alone".


def _google(texts, src: Language, dst: Language, context=None):
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source=src.translator, target=dst.translator)
    if len(texts) > 1:
        # One request per page instead of one per bubble (which also gives the
        # translator some context); fall back if the line structure didn't survive.
        parts = translator.translate("\n".join(texts)).split("\n")
        if len(parts) == len(texts):
            return parts
    return [translator.translate(text) for text in texts]


def _mymemory(texts, src: Language, dst: Language, context=None):
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


def _local(texts, src: Language, dst: Language, context=None):
    import torch

    tokenizer, model = _nllb()
    with _local_lock:  # the tokenizer's source language is shared state
        tokenizer.src_lang = src.nllb
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.inference_mode():
            output = model.generate(
                **batch,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(dst.nllb),
                num_beams=4,
                no_repeat_ngram_size=3,
            )
    results = tokenizer.batch_decode(output, skip_special_tokens=True)
    # NLLB tends to invent whole sentences for interjections ("Paaapiii!"):
    # drop outputs far longer than their source.
    return [
        result if len(result.split()) <= 2 * len(text.split()) + 2 else None
        for text, result in zip(texts, results)
    ]


CLAUDE_SYSTEM_PROMPT = """\
You translate the lettering of manga, manhwa and comics for a reader app. You get \
the text of every speech bubble and caption on one page, in reading order, as \
read by OCR, and translate each one into {target}.

- The OCR makes mistakes: wrong or missing letters and accents, words run \
together or split apart. Work out what the lettering most likely said, using the \
rest of the page and the earlier dialogue, and translate that.
- Write natural spoken {target}, the way a good localization would: short enough \
to fit a speech bubble, in character, keeping the tone (casual, rude, polite, \
childish, formal). Don't explain jokes or add notes.
- Keep character names consistent with the name list and earlier pages. Keep \
Japanese honorifics (-san, -kun, -chan, senpai) when the text uses them.
- Shouts, interjections and sound words get a natural {target} equivalent; keep \
stretched sounds stretched.
- If an entry is not readable text at all (OCR noise from artwork), return an \
empty string for it.
- Return exactly one translation for every bubble id you were given.
- In "names", list the character and place names on this page with how you \
wrote them in {target}.{extra}"""

CLAUDE_VIETNAMESE_NOTE = """
- Choose Vietnamese pronouns and forms of address (anh, em, chị, cậu, tớ, mày, \
tao, ông, bà, con...) that fit how the speakers relate to each other, as far as \
the context shows, and keep them consistent."""

CLAUDE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        },
        "names": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "target": {"type": "string"}},
                "required": ["source", "target"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["translations", "names"],
    "additionalProperties": False,
}


def claude_configured() -> bool:
    """Claude is used when credentials are in the environment, or when it's asked for by name."""
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or "claude" in os.environ.get("MANTRAN_TRANSLATORS", "")
    )


@lru_cache(maxsize=1)
def _claude_client():
    import anthropic

    return anthropic.Anthropic()


def _claude_prompt(texts, src: Language, dst: Language, context: TranslationContext | None) -> str:
    parts = [f"Source language: {src.name}\nTarget language: {dst.name}"]
    if context and context.names:
        names = "\n".join(f"- {source} -> {target}" for source, target in context.names.items())
        parts.append(f"Names used so far:\n{names}")
    if context and context.history:
        earlier = "\n".join(
            f"- {json.dumps(source, ensure_ascii=False)} -> {json.dumps(target, ensure_ascii=False)}"
            for source, target in context.history
        )
        parts.append(f"Earlier dialogue in this chapter, oldest first (context only, don't translate again):\n{earlier}")
    bubbles = [{"id": i + 1, "text": text} for i, text in enumerate(texts)]
    parts.append(f"Bubbles on this page:\n{json.dumps(bubbles, ensure_ascii=False, indent=1)}")
    return "\n\n".join(parts)


def _claude(texts, src: Language, dst: Language, context: TranslationContext | None = None):
    if not claude_configured():
        raise BackendUnavailable("no Anthropic credentials")
    try:
        import anthropic
    except ImportError as exc:
        raise BackendUnavailable("the anthropic package is not installed") from exc

    extra = CLAUDE_VIETNAMESE_NOTE if dst.code == "vi" else ""
    try:
        response = _claude_client().beta.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16000,
            # If a request is declined by a safety classifier, the API retries it
            # on a suitable fallback model instead of failing the page.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={
                "effort": CLAUDE_EFFORT,
                "format": {"type": "json_schema", "schema": CLAUDE_OUTPUT_SCHEMA},
            },
            system=CLAUDE_SYSTEM_PROMPT.format(target=dst.name, extra=extra),
            messages=[{"role": "user", "content": _claude_prompt(texts, src, dst, context)}],
        )
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
        raise BackendUnavailable(f"Anthropic rejected the credentials: {exc}") from exc

    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"Claude stopped early ({response.stop_reason})")
    text = next((block.text for block in response.content if block.type == "text"), "")
    data = json.loads(text)

    if context is not None:
        for name in data.get("names", []):
            if name.get("source") and name.get("target"):
                context.names[name["source"]] = name["target"]
    by_id = {item["id"]: item["text"] for item in data.get("translations", [])}
    return [by_id.get(i + 1) for i in range(len(texts))]


BACKENDS = {"claude": _claude, "google": _google, "mymemory": _mymemory, "local": _local}


def backend_order() -> list[str]:
    names = os.environ.get("MANTRAN_TRANSLATORS", DEFAULT_BACKENDS).split(",")
    return [name.strip() for name in names if name.strip() in BACKENDS]


# --- Source text clean-up ------------------------------------------------------------


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
    text = _sentence_case(text)
    if language.code == "en":
        text = re.sub(r"\bi\b", "I", text)
    return text


def _clean_result(result: str) -> str:
    return " ".join(html.unescape(result).split())


def translate_texts(texts, src="ja", dst="en", context: TranslationContext | None = None) -> list[str | None]:
    """Translate a batch (e.g. every bubble of a page).

    Returns one entry per input: the translation, "" for empty input or text a
    backend judged not to be real lettering, or ``None`` where no backend could
    translate it. ``context`` carries earlier pages of the same chapter.
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
            outputs = BACKENDS[name]([prepared[i] for i in pending], source, target, context)
        except (ImportError, BackendUnavailable) as exc:
            if name != "claude" or claude_configured():
                print(f"{name} translation unavailable: {exc}")
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
            if output is None:
                still_pending.append(i)
                continue
            results[i] = _clean_result(output)
            if results[i]:
                _cache[(prepared[i], source.translator, target.translator)] = results[i]
        pending = still_pending

    if pending:
        print(f"No translation backend could translate {len(pending)} text(s).")
    return results


def translate_text(text, src="ja", dst="en") -> str | None:
    """Translate one string; returns ``None`` if every backend fails."""
    return translate_texts([text], src, dst)[0]
