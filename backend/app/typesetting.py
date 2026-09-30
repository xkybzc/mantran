"""Lay out and draw translated text inside speech bubbles."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .detection import Box

FONT_DIR = Path(__file__).resolve().parent / "fonts"
_WINDOWS_FONTS = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "Fonts"
# Tried in order; the first one that has every glyph of the text wins
# (Comic Neue has no Vietnamese, for example, so that falls through to Arial).
FONT_CANDIDATES = (
    FONT_DIR / "ComicNeue-Bold.ttf",
    _WINDOWS_FONTS / "comicbd.ttf",
    _WINDOWS_FONTS / "arialbd.ttf",
    Path("DejaVuSans-Bold.ttf"),
)
LINE_SPACING = 0.95


@dataclass
class TextLayout:
    lines: list[tuple[str, float, float]]  # text, centre x, top y (image coordinates)
    font_size: int
    font_path: str | None = None
    color: tuple[int, int, int] = (0, 0, 0)
    stroke_width: int = 0


@lru_cache(maxsize=None)
def _available_fonts() -> tuple[str, ...]:
    paths = []
    for candidate in FONT_CANDIDATES:
        try:
            ImageFont.truetype(str(candidate), 12)
        except OSError:
            continue
        paths.append(str(candidate))
    return tuple(paths)


@lru_cache(maxsize=4096)
def _has_glyph(path: str, char: str) -> bool:
    font = ImageFont.truetype(path, 24)
    # U+10FFFD is never mapped, so it renders as the font's "missing glyph" box.
    return bytes(font.getmask(char)) != bytes(font.getmask("\U0010FFFD"))


def resolve_font_path(text: str = "") -> str | None:
    """First candidate font that can draw every character of ``text``."""
    fonts = _available_fonts()
    chars = set(text) - set(" \t\n")
    for path in fonts:
        if all(_has_glyph(path, char) for char in chars):
            return path
    return fonts[0] if fonts else None


@lru_cache(maxsize=256)
def load_font(size: int, path: str | None = None):
    path = path or resolve_font_path()
    if path is None:
        return ImageFont.load_default(size)
    return ImageFont.truetype(path, size)


def _line_height(font) -> int:
    ascent, descent = font.getmetrics()
    return max(1, round((ascent + descent) * LINE_SPACING))


def _row_spans(mask: np.ndarray, cx: int):
    """For each row, the [left, right) run of the mask that contains column ``cx``."""
    height, width = mask.shape
    left = np.full(height, cx)
    right = np.full(height, cx)
    for y in np.flatnonzero(mask[:, cx]):
        row = mask[y]
        gaps_left = np.flatnonzero(~row[:cx])
        gaps_right = np.flatnonzero(~row[cx:])
        left[y] = gaps_left[-1] + 1 if gaps_left.size else 0
        right[y] = cx + gaps_right[0] if gaps_right.size else width
    return left, right


def _balanced_wrap(words, slots, measure, space_width):
    """Split ``words`` into exactly ``len(slots)`` lines that fit their slot widths.

    Among the splits that fit, picks the one whose lines fill their slots most
    evenly (least squared unused fraction), so no line is left dangling.
    """
    word_widths = [measure(word) for word in words]
    prefix = np.concatenate([[0.0], np.cumsum(word_widths)])
    count, total = len(slots), len(words)
    cost = np.full((count + 1, total + 1), np.inf)
    split = np.zeros((count + 1, total + 1), dtype=int)
    cost[0, 0] = 0.0
    for line, slot in enumerate(slots, start=1):
        if slot <= 0:
            return None
        for end in range(line, total + 1):
            for start in range(end - 1, line - 2, -1):
                width = prefix[end] - prefix[start] + space_width * (end - start - 1)
                if width > slot:
                    break
                candidate = cost[line - 1, start] + ((slot - width) / slot) ** 2
                if candidate < cost[line, end]:
                    cost[line, end], split[line, end] = candidate, start
    if not np.isfinite(cost[count, total]):
        return None

    lines, end = [], total
    for line in range(count, 0, -1):
        start = split[line, end]
        lines.append(" ".join(words[start:end]))
        end = start
    return lines[::-1]


def layout_text(text: str, area_box: Box, area_mask: np.ndarray | None = None,
                max_size=40, min_size=8, uppercase=True) -> TextLayout | None:
    """Pick the largest font size at which ``text`` fits inside the area.

    ``area_mask`` (same size as ``area_box``) describes the usable shape, e.g. an
    oval bubble; without it the whole box is usable. Lines share one centre axis
    and each line's width is limited by how wide the shape is at that height.
    """
    words = (text.upper() if uppercase else text).split()
    if not words:
        return None
    font_path = resolve_font_path(" ".join(words))

    x0, y0, x1, y1 = area_box
    if area_mask is None:
        area_mask = np.ones((y1 - y0, x1 - x0), dtype=bool)
    ys, xs = np.nonzero(area_mask)
    if ys.size == 0:
        return None
    cx, cy = int(np.median(xs)), float(ys.mean())
    if not area_mask[int(cy), cx]:
        cy = float(np.median(ys[xs == cx]))
    left, right = _row_spans(area_mask, cx)
    # Centre on the middle of the row through the text centre, then keep every
    # line symmetric about that axis so the block doesn't zig-zag.
    cx = int(left[int(cy)] + right[int(cy)]) // 2
    left, right = _row_spans(area_mask, cx)
    half_widths = np.minimum(cx - left, right - cx)
    height = area_mask.shape[0]

    for size in range(max(min_size, max_size), min_size - 1, -1):
        font = load_font(size, font_path)
        measure = font.getlength
        space = measure(" ")
        line_h = _line_height(font)
        max_lines = min(len(words), height // line_h)
        for count in range(1, max_lines + 1):
            top = cy - count * line_h / 2
            if top < 0 or top + count * line_h > height:
                break
            slots = [
                2 * int(half_widths[int(top + n * line_h):int(np.ceil(top + (n + 1) * line_h))].min())
                for n in range(count)
            ]
            lines = _balanced_wrap(words, slots, measure, space)
            if lines is None:
                continue
            placed = [(line, x0 + cx, y0 + top + n * line_h) for n, line in enumerate(lines)]
            return TextLayout(placed, size, font_path)
    return None


def draw_layouts(image, layouts: list[TextLayout]):
    """Render all layouts onto a copy of the BGR ``image``."""
    if not layouts:
        return image
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    for layout in layouts:
        font = load_font(layout.font_size, layout.font_path)
        for line, cx, top in layout.lines:
            draw.text(
                (cx, top), line, font=font, fill=layout.color, anchor="ma",
                stroke_width=layout.stroke_width, stroke_fill=(255, 255, 255),
            )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
