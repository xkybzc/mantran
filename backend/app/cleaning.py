"""Erase the original lettering from detected text regions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .detection import Box, TextRegion, pad_box


@dataclass
class BubbleInterior:
    """The light inside of a speech bubble (lettering included), in bubble-box coordinates."""

    box: Box
    mask: np.ndarray
    color: tuple[int, int, int]
    is_flat: bool


def find_bubble_interior(image, region: TextRegion) -> BubbleInterior | None:
    """Segment the bubble's inside so it can be repainted and used as the lettering area.

    Returns ``None`` when the region has no bubble or the segmentation looks
    unreliable (e.g. the bubble outline is open and the inside leaks into the art).
    """
    if region.bubble_box is None:
        return None

    x0, y0, x1, y1 = region.bubble_box
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    crop_h, crop_w = gray.shape

    # Faint or broken outlines (low-res scans) must still separate the inside from
    # white artwork around the bubble: use a strict "light" level and thicken the
    # dark strokes to seal small gaps. The inside is grown back afterwards.
    paper = float(np.median(gray[gray > threshold])) if (gray > threshold).any() else 255.0
    light_level = (threshold + paper) / 2
    light = (gray > light_level).astype(np.uint8)
    seal = max(1, round(min(crop_h, crop_w) * 0.01))
    dark = cv2.dilate(1 - light, np.ones((3, 3), np.uint8), iterations=seal)
    light = 1 - dark

    # Dark blobs that don't reach the box edge are lettering, so count them as inside.
    # The outline and the surrounding artwork always touch the edge.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    sx, sy, sw, sh = (stats[:, i] for i in range(4))
    enclosed = (sx > 0) & (sy > 0) & (sx + sw < crop_w) & (sy + sh < crop_h)
    enclosed[0] = False
    candidate = (light.astype(bool) | enclosed[labels]).astype(np.uint8)

    # The inside is the light component that holds most of the text.
    count, labels = cv2.connectedComponents(candidate, connectivity=4)
    tx0, ty0, tx1, ty1 = region.text_box
    text_labels = labels[max(0, ty0 - y0):ty1 - y0, max(0, tx0 - x0):tx1 - x0]
    votes = np.bincount(text_labels.ravel(), minlength=count)
    votes[0] = 0
    best = int(np.argmax(votes))
    if best == 0 or votes[best] < 0.4 * text_labels.size:
        return None
    component = cv2.dilate((labels == best).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=seal)

    # Fill holes (e.g. lettering glued to nothing but itself).
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(component)
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)

    if mask.sum() < 0.2 * mask.size or _leaks_out(mask, region.bubble_box, image.shape):
        return None

    inside_light = mask.astype(bool) & (gray > light_level)
    background = crop[inside_light]
    color = tuple(int(c) for c in np.median(background, axis=0))
    is_flat = float(gray[inside_light].std()) < 20 and min(color) > 150
    return BubbleInterior(region.bubble_box, mask.astype(bool), color, is_flat)


def _leaks_out(mask, box: Box, image_shape, max_contact=0.2) -> bool:
    """True if the mask runs along box edges that aren't also image edges."""
    height, width = image_shape[:2]
    edges = []
    if box[1] > 0:
        edges.append(mask[0, :])
    if box[3] < height:
        edges.append(mask[-1, :])
    if box[0] > 0:
        edges.append(mask[:, 0])
    if box[2] < width:
        edges.append(mask[:, -1])
    if not edges:
        return False
    border = np.concatenate(edges)
    return border.mean() > max_contact


def text_stroke_mask(image, text_box: Box, bubble_box: Box | None = None) -> np.ndarray:
    """Mask of the dark glyph strokes inside ``text_box`` (full-image size)."""
    height, width = image.shape[:2]
    tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
    x0, y0, x1, y1 = pad_box(text_box, max(2, int(min(tw, th) * 0.08)), image.shape)
    if bubble_box is not None:
        x0, y0 = max(x0, bubble_box[0]), max(y0, bubble_box[1])
        x1, y1 = min(x1, bubble_box[2]), min(y1, bubble_box[3])

    gray = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Drop long strokes that run into the crop edge: bubble outlines and art lines.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    crop_h, crop_w = gray.shape
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        touches_edge = x == 0 or y == 0 or x + w == crop_w or y + h == crop_h
        if touches_edge and (w > 0.6 * crop_w or h > 0.6 * crop_h):
            dark[labels == label] = 0

    radius = max(2, round(min(tw, th) * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = cv2.dilate(dark, kernel)
    return mask


def remove_text_regions(image, mask, radius=3):
    """Inpaint the masked pixels. Only the mask's bounding rectangle is processed."""
    if image is None:
        raise ValueError("Image cannot be None")

    result = image.copy()
    mask = mask.astype(np.uint8)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    points = cv2.findNonZero(mask)
    if points is None:
        return result

    x, y, w, h = cv2.boundingRect(points)
    pad = radius * 2 + 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    result[y0:y1, x0:x1] = cv2.inpaint(result[y0:y1, x0:x1], mask[y0:y1, x0:x1], radius, cv2.INPAINT_TELEA)
    return result


def erase_region(image, region: TextRegion, interior: BubbleInterior | None):
    """Remove the region's lettering from ``image`` and return the cleaned image."""
    if interior is not None and interior.is_flat:
        # Repaint the whole inside with the bubble colour, keeping a thin band
        # next to the outline so its anti-aliased edge isn't eaten.
        x0, y0, x1, y1 = interior.box
        band = max(1, round(min(x1 - x0, y1 - y0) * 0.01))
        fill = cv2.erode(interior.mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=band)
        image = image.copy()
        image[y0:y1, x0:x1][fill.astype(bool)] = interior.color
        return image

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in region.text_boxes:
        mask |= text_stroke_mask(image, box, region.bubble_box)
    return remove_text_regions(image, mask)
