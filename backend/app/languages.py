"""The languages Mantran can read (OCR) and write (translate and letter)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str                      # MangaDex language code, also used across the app
    name: str
    translator: str                # code for Google Translate / MyMemory
    nllb: str                      # code for the local NLLB-200 model
    tesseract: str | None = None   # traineddata name; None means manga-ocr (Japanese)
    tesseract_vertical: str | None = None
    square_glyphs: bool = False    # CJK: every character takes a full square
    word_separator: str = " "      # "" for scripts written without spaces
    variant_of: str | None = None  # regional variant (e.g. es-la of es)


_ALL = (
    Language("ja", "Japanese", "ja", "jpn_Jpan", None, "jpn_vert", True, ""),
    Language("ko", "Korean", "ko", "kor_Hang", "kor", "kor_vert", True, " "),
    Language("zh", "Chinese (Simplified)", "zh-CN", "zho_Hans", "chi_sim", "chi_sim_vert", True, ""),
    Language("zh-hk", "Chinese (Traditional)", "zh-TW", "zho_Hant", "chi_tra", "chi_tra_vert", True, ""),
    Language("en", "English", "en", "eng_Latn", "eng"),
    Language("es", "Spanish", "es", "spa_Latn", "spa"),
    Language("es-la", "Spanish (Latin America)", "es", "spa_Latn", "spa", variant_of="es"),
    Language("pt", "Portuguese", "pt", "por_Latn", "por"),
    Language("pt-br", "Portuguese (Brazil)", "pt", "por_Latn", "por", variant_of="pt"),
    Language("id", "Indonesian", "id", "ind_Latn", "ind"),
    Language("fr", "French", "fr", "fra_Latn", "fra"),
    Language("it", "Italian", "it", "ita_Latn", "ita"),
    Language("de", "German", "de", "deu_Latn", "deu"),
    Language("ru", "Russian", "ru", "rus_Cyrl", "rus"),
    Language("tr", "Turkish", "tr", "tur_Latn", "tur"),
    Language("pl", "Polish", "pl", "pol_Latn", "pol"),
    Language("vi", "Vietnamese", "vi", "vie_Latn", "vie"),
)

LANGUAGES: dict[str, Language] = {language.code: language for language in _ALL}
TARGET_CODES = ("en", "vi")

# Names for MangaDex languages Mantran can't read, so they can still be listed.
OTHER_NAMES = {
    "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan", "cs": "Czech",
    "da": "Danish", "el": "Greek", "fa": "Persian", "fi": "Finnish", "he": "Hebrew",
    "hi": "Hindi", "hu": "Hungarian", "ja-ro": "Japanese (romanized)", "ko-ro": "Korean (romanized)",
    "zh-ro": "Chinese (romanized)", "lt": "Lithuanian", "mn": "Mongolian", "ms": "Malay",
    "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "no": "Norwegian", "ro": "Romanian",
    "sv": "Swedish", "th": "Thai", "tl": "Filipino", "uk": "Ukrainian",
}


def get_language(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"Unsupported language: {code}") from None


def language_name(code: str) -> str:
    if code in LANGUAGES:
        return LANGUAGES[code].name
    return OTHER_NAMES.get(code, code)


def source_languages(include_variants=False) -> list[Language]:
    return [lang for lang in _ALL if include_variants or lang.variant_of is None]


def target_languages() -> list[Language]:
    return [LANGUAGES[code] for code in TARGET_CODES]


def check_pair(source: str, target: str):
    """Raise ``ValueError`` unless ``source`` -> ``target`` is a supported translation."""
    get_language(source)
    if target not in TARGET_CODES:
        raise ValueError(f"Target language must be one of: {', '.join(TARGET_CODES)}")
    if LANGUAGES[source].translator == LANGUAGES[target].translator:
        raise ValueError("Source and target language are the same")
