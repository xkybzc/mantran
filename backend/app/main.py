"""Entry point, run from ``backend/``.

One page:      ``python -m app.main -i tests/image2.png``
A folder:      ``python -m app.main -i path/to/chapter_folder --source-lang es``
A chapter:     ``python -m app.main --chapter https://mangadex.org/chapter/<id> --target-lang vi``
Website + API: ``uvicorn app.main:app``
"""

import argparse
import shutil
import sys
from pathlib import Path

from .pipeline import process_image

try:
    from .api import app
except ImportError:  # pragma: no cover - fastapi is optional for CLI use
    app = None

TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Translate the speech bubbles of manga pages or a chapter")
    parser.add_argument(
        "--input", "-i", default=str(TESTS_DIR / "image2.png"),
        help="Manga page to translate, or a folder of pages (translated in file name order)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output image (for one page) or folder (for a folder; default: <folder>_<target-lang>)",
    )
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
        args.output or str(TESTS_DIR / "image_inpainted.png"),
        source_lang=args.source_lang or "ja",
        target_lang=args.target_lang,
        use_detector=not args.use_threshold_mask,
        scale=args.scale,
    )
    for region in metadata["regions"]:
        print(f"{region['text_box']}: {region['text']} -> {region['translation']}")
    print("Saved translated image:", out_path)
    return out_path


def _print_progress(job):
    done = len(job.pages) + len(job.failed_pages)
    print(f"\r{job.chapter['label']}: page {done} of {job.total}", end="", flush=True)


def _print_summary(job):
    print()
    if job.failed_pages:
        print("Pages that failed:", ", ".join(str(i + 1) for i in sorted(job.failed_pages)))
    if job.untranslated:
        print(f"{job.untranslated} text region(s) could not be translated and were left as they were.")


def translate_folder(args):
    from .jobs import natural_key, new_upload_job, translate_upload

    folder = Path(args.input)
    images = sorted(
        (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: natural_key(path.name),
    )
    if not images:
        sys.exit(f"No images found in {folder}")
    output = Path(args.output) if args.output else folder.with_name(f"{folder.name}_{args.target_lang}")
    print(f"Translating {len(images)} pages from {folder}")

    job = new_upload_job(
        [(path.name, path.read_bytes()) for path in images], args.source_lang or "ja", args.target_lang, folder.name
    )
    translate_upload(job, on_page=_print_progress)
    _print_summary(job)
    output.mkdir(parents=True, exist_ok=True)
    for index in sorted(job.pages):
        shutil.copy2(job.page_path(index), output / job.pages[index])
    shutil.rmtree(job.folder.parent, ignore_errors=True)
    print("Translated pages:", output)
    return output


def translate_whole_chapter(args):
    from . import mangadex
    from .jobs import TranslationJob, build_zip, translate_chapter

    kind, chapter_id = mangadex.parse_link(args.chapter)
    if kind == "manga":
        sys.exit("That is a title link. Open a chapter on MangaDex and copy its link instead.")
    source_lang = args.source_lang or mangadex.get_chapter(chapter_id)["language"]

    job = TranslationJob(source_lang, args.target_lang, chapter_id=chapter_id)
    translate_chapter(job, on_page=_print_progress)
    _print_summary(job)
    print("Translated pages:", job.folder)
    print("Zip:", build_zip(job))
    return job.folder


def main():
    # Japanese and Vietnamese text crash the default Windows console encoding.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.chapter:
        return translate_whole_chapter(args)
    if Path(args.input).is_dir():
        return translate_folder(args)
    return translate_one_image(args)


if __name__ == "__main__":
    main()
