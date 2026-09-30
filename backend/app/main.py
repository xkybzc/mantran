"""Entry point, run from ``backend/``.

CLI:  ``python -m app.main -i tests/image2.png``
API:  ``uvicorn app.main:app``
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
    parser = argparse.ArgumentParser(description="Translate the speech bubbles of a manga page")
    parser.add_argument("--input", "-i", default=str(TESTS_DIR / "image2.png"), help="Input manga image path")
    parser.add_argument("--output", "-o", default=str(TESTS_DIR / "image_inpainted.png"), help="Output image path")
    parser.add_argument(
        "--use-threshold-mask",
        action="store_true",
        help="Detect text by thresholding instead of the bubble detector model",
    )
    parser.add_argument(
        "--scale", type=int, default=0, help="Upscale factor before processing (0 = automatic for small images)"
    )
    parser.add_argument("--source-lang", default="ja", help="Source language for translation")
    parser.add_argument("--target-lang", default="en", help="Target language for translation")
    return parser.parse_args()


def main():
    # Japanese text crashes the default Windows console encoding.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    print("Input image:", args.input)
    out_path, metadata = process_image(
        args.input,
        args.output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        use_detector=not args.use_threshold_mask,
        scale=args.scale,
    )
    for region in metadata["regions"]:
        print(f"{region['text_box']}: {region['text']} -> {region['translation']}")
    print("Saved translated image:", out_path)
    return out_path


if __name__ == "__main__":
    main()
