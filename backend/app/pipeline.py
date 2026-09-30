"""End-to-end page translation: detect -> OCR -> translate -> erase -> typeset."""

from __future__ import annotations

import math
import os

import cv2
import numpy as np

from .cleaning import erase_region, find_bubble_interior
from .detection import TextRegion, detect_regions, pad_box
from .ocr import read_region
from .translation import translate_text
from .typesetting import layout_text, draw_layouts

# Small scans are upscaled first so the English lettering stays legible.
MIN_WORKING_SIZE = 512
# English lettering is set at about this fraction of the original glyph size.
FONT_TO_GLYPH_RATIO = 0.65


def auto_scale(image, min_size=MIN_WORKING_SIZE) -> int:
    longest = max(image.shape[:2])
    return max(1, math.ceil(min_size / longest)) if longest < min_size else 1


def _glyph_size(region: TextRegion, text: str) -> float:
    """Approximate size of one original character: text area shared per character."""
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in region.text_boxes)
    return math.sqrt(area / max(1, len(text)))


def _text_area(region: TextRegion, interior, image_shape):
    """Where the translation may go: the bubble's inside, or a box around the text."""
    if interior is not None:
        x0, y0, x1, y1 = interior.box
        margin = max(2, round(min(x1 - x0, y1 - y0) * 0.08))
        mask = cv2.erode(interior.mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=margin)
        if mask.any():
            return interior.box, mask.astype(bool)

    if region.bubble_box is not None:
        x0, y0, x1, y1 = region.bubble_box
        dx, dy = (x1 - x0) * 0.15, (y1 - y0) * 0.15
        return (round(x0 + dx), round(y0 + dy), round(x1 - dx), round(y1 - dy)), None

    # Free text: widen tall vertical columns so horizontal English has room.
    x0, y0, x1, y1 = region.text_box
    extra = max(0, round((y1 - y0) * 0.6 - (x1 - x0)) // 2)
    box = pad_box((x0 - extra, y0, x1 + extra, y1), 0, image_shape)
    return box, None


def translate_image(image, source_lang="ja", target_lang="en", use_detector=True, scale=0):
    """Translate a BGR page image. Returns ``(result_image, metadata)``.

    ``scale`` upsamples the page before processing; ``0`` picks it automatically.
    """
    scale = auto_scale(image) if scale <= 0 else scale
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    regions = detect_regions(image, use_model=use_detector)
    min_font = max(8, round(max(image.shape[:2]) * 0.012))

    result = image
    layouts = []
    region_info = []
    for region in regions:
        source = read_region(image, region, source_lang)
        translation = translate_text(source, src=source_lang, dst=target_lang) if source else ""
        region_info.append({
            "text_box": region.text_box,
            "bubble_box": region.bubble_box,
            "text": source,
            "translation": translation or "",
        })
        if not translation:
            # Nothing to put back: leave the original lettering untouched.
            continue

        interior = find_bubble_interior(image, region)
        result = erase_region(result, region, interior)
        area_box, area_mask = _text_area(region, interior, image.shape)
        max_font = max(min_font, round(_glyph_size(region, source) * FONT_TO_GLYPH_RATIO))
        layout = layout_text(translation, area_box, area_mask, max_size=max_font, min_size=min_font)
        if layout is None:
            layout = layout_text(translation, area_box, None, max_size=max_font, min_size=min_font // 2)
        if layout is not None:
            if region.bubble_box is None:
                layout.stroke_width = max(1, layout.font_size // 8)
            layouts.append(layout)

    result = draw_layouts(result, layouts)
    metadata = {
        "scale": scale,
        "regions": region_info,
        "boxes": [info["text_box"] for info in region_info],
        "extracted_text": "\n".join(info["text"] for info in region_info if info["text"]),
        "translated_text": "\n".join(info["translation"] for info in region_info if info["translation"]),
    }
    return result, metadata


def process_image(input_path, output_path, source_lang="ja", target_lang="en", use_detector=True, scale=0):
    """File-to-file wrapper around :func:`translate_image`."""
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load input image: {input_path}")

    result, metadata = translate_image(
        image, source_lang=source_lang, target_lang=target_lang, use_detector=use_detector, scale=scale
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not cv2.imwrite(output_path, result):
        raise RuntimeError(f"Failed to write image: {output_path}")
    return output_path, metadata
