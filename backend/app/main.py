"""Entry point, run from ``backend/``.

One page:      ``python -m app.main -i tests/image2.png``
A chapter:     ``python -m app.main --chapter https://mangadex.org/chapter/<id> --target-lang vi``
Website + API: ``uvicorn app.main:app``
"""

import argparse
import sys
from pathlib import Path

from .pipeline import process_image

try:
    from .api import app
except ImportError:  # pragma: no cover - fastapi is optional for CLI use
    app = None

TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"


def parse_args():
    parser = argparse.ArgumentParser(description="Translate the speech bubbles of a manga page or chapter")
    parser.add_argument("--input", "-i", default=str(TESTS_DIR / "image2.png"), help="Input manga image path")
    parser.add_argument("--output", "-o", default=str(TESTS_DIR / "image_inpainted.png"), help="Output image path")
    parser.add_argument("--chapter", "-c", help="MangaDex chapter link: translate the whole chapter instead")
    parser.add_argument(
        "--use-threshold-mask",
        action="store_true",
        help="Detect text by thresholding instead of the bubble detector model",
    )
    parser.add_argument(
        "--scale", type=int, default=0, help="Upscale factor before processing (0 = automatic for small images)"
    )
    parser.add_argument(
        "--source-lang",
        help="Language of the text on the page (default: ja for images, the chapter's language for chapters)",
    )
    parser.add_argument("--target-lang", default="en", choices=["en", "vi"], help="Language to translate into")
    return parser.parse_args()


def translate_one_image(args):
    print("Input image:", args.input)
    out_path, metadata = process_image(
        args.input,
        args.output,
        source_lang=args.source_lang or "ja",
        target_lang=args.target_lang,
        use_detector=not args.use_threshold_mask,
        scale=args.scale,
    )
    for region in metadata["regions"]:
        print(f"{region['text_box']}: {region['text']} -> {region['translation']}")
    print("Saved translated image:", out_path)
    return out_path


def translate_whole_chapter(args):
    from . import mangadex
    from .chapters import ChapterJob, build_zip, translate_chapter

    kind, chapter_id = mangadex.parse_link(args.chapter)
    if kind == "manga":
        sys.exit("That is a title link. Open a chapter on MangaDex and copy its link instead.")
    source_lang = args.source_lang or mangadex.get_chapter(chapter_id)["language"]

    def report(job):
        done = len(job.pages) + len(job.failed_pages)
        print(f"\r{job.chapter['label']}: page {done} of {job.total}", end="", flush=True)

    job = translate_chapter(ChapterJob(chapter_id, source_lang, args.target_lang), on_page=report)
    print()
    if job.failed_pages:
        print("Pages that failed:", ", ".join(str(i + 1) for i in sorted(job.failed_pages)))
    if job.untranslated:
        print(f"{job.untranslated} text region(s) could not be translated and were left as they were.")
    print("Translated pages:", job.folder)
    print("Zip:", build_zip(job))
    return job.folder


def main():
    # Japanese and Vietnamese text crash the default Windows console encoding.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.chapter:
        return translate_whole_chapter(args)
    return translate_one_image(args)


if __name__ == "__main__":
    main()
