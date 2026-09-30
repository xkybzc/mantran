"""Locate speech bubbles and the blocks of text inside them."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

DETECTOR_MODEL = "ogkalu/comic-text-and-bubble-detector"
DETECTOR_INPUT_SIZE = 640
LABEL_BUBBLE = 0

Box = tuple[int, int, int, int]


@dataclass
class TextRegion:
    """A block of text to translate and, if it sits in one, its speech bubble."""

    text_boxes: list[Box]
    bubble_box: Box | None = None

    @property
    def text_box(self) -> Box:
        return union_box(self.text_boxes)


def box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection_area(box_a: Box, box_b: Box) -> int:
    return box_area((
        max(box_a[0], box_b[0]),
        max(box_a[1], box_b[1]),
        min(box_a[2], box_b[2]),
        min(box_a[3], box_b[3]),
    ))


def overlap_ratio(box_a: Box, box_b: Box) -> float:
    """Intersection over the area of the smaller box."""
    smaller_area = min(box_area(box_a), box_area(box_b))
    if smaller_area <= 0:
        return 0.0
    return intersection_area(box_a, box_b) / smaller_area


def union_box(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def clip_box(box, image_shape) -> Box | None:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def pad_box(box: Box, pad: int, image_shape) -> Box:
    x0, y0, x1, y1 = box
    return clip_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), image_shape) or box


def detect_regions(image, use_model=True, score_threshold=0.35) -> list[TextRegion]:
    """Find text regions, preferring the learned detector over thresholding."""
    if use_model:
        try:
            detections = detect_with_model(image, score_threshold=score_threshold)
        except Exception as exc:
            print(f"Bubble detector unavailable ({exc}); falling back to threshold detection.")
        else:
            return group_detections(detections)
    return [TextRegion([box]) for box in get_text_boxes_from_threshold(image)]


@lru_cache(maxsize=1)
def _load_detector():
    from transformers import AutoModelForObjectDetection

    return AutoModelForObjectDetection.from_pretrained(DETECTOR_MODEL).eval()


def detect_with_model(image, score_threshold=0.35):
    """Run the RT-DETR comic detector. Returns ``(label, score, box)`` tuples.

    The model's preprocessing is only resize + rescale, so it is done here
    directly instead of through ``AutoImageProcessor`` (which needs torchvision).
    """
    import torch

    model = _load_detector()
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    interpolation = cv2.INTER_AREA if max(height, width) > DETECTOR_INPUT_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(rgb, (DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE), interpolation=interpolation)
    pixels = torch.from_numpy(resized).permute(2, 0, 1).float().div(255).unsqueeze(0)

    with torch.inference_mode():
        outputs = model(pixel_values=pixels)

    scores, labels = outputs.logits.sigmoid()[0].max(dim=-1)
    keep = scores > score_threshold
    detections = []
    for (cx, cy, bw, bh), label, score in zip(
        outputs.pred_boxes[0][keep].tolist(), labels[keep].tolist(), scores[keep].tolist()
    ):
        box = clip_box(
            ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height),
            image.shape,
        )
        if box:
            detections.append((label, score, box))
    return _suppress_duplicates(detections)


def _suppress_duplicates(detections, max_overlap=0.7):
    kept = []
    for label, score, box in sorted(detections, key=lambda d: -d[1]):
        if any(label == k[0] and overlap_ratio(box, k[2]) > max_overlap for k in kept):
            continue
        kept.append((label, score, box))
    return kept


def _reading_order(boxes: list[Box]) -> list[Box]:
    # Vertical Japanese columns read right to left; horizontal lines top to bottom.
    if all(b[3] - b[1] >= b[2] - b[0] for b in boxes):
        return sorted(boxes, key=lambda b: -b[2])
    return sorted(boxes, key=lambda b: b[1])


def group_detections(detections) -> list[TextRegion]:
    """Attach each detected text box to the bubble that contains it."""
    bubbles = [box for label, _, box in detections if label == LABEL_BUBBLE]
    grouped: dict[int, list[Box]] = {}
    free_boxes = []
    for label, _, box in detections:
        if label == LABEL_BUBBLE:
            continue
        overlaps = [intersection_area(box, bubble) for bubble in bubbles]
        best = int(np.argmax(overlaps)) if overlaps else -1
        if best >= 0 and overlaps[best] > 0.5 * box_area(box):
            grouped.setdefault(best, []).append(box)
        else:
            free_boxes.append(box)

    regions = [TextRegion(_reading_order(boxes), bubbles[i]) for i, boxes in grouped.items()]
    regions += [TextRegion([box]) for box in free_boxes]
    return sorted(regions, key=lambda r: (r.text_box[1], -r.text_box[2]))


# --- Threshold-based fallback (no model required) ---------------------------------


def _prune_boxes(boxes, image_shape):
    height, width = image_shape[:2]
    min_area = max(25, int(width * height * 0.00035))
    max_area = max(100, int(width * height * 0.45))

    filtered = []
    for x0, y0, x1, y1 in boxes:
        width_px = x1 - x0
        height_px = y1 - y0
        if width_px < 6 or height_px < 6:
            continue
        area = width_px * height_px
        if area < min_area or area > max_area:
            continue
        if max(width_px, height_px) / max(1, min(width_px, height_px)) > 12:
            continue
        filtered.append((x0, y0, x1, y1))

    unique_boxes = []
    for box in sorted(filtered, key=lambda item: (item[1], item[0])):
        if any(overlap_ratio(box, existing) > 0.75 for existing in unique_boxes):
            continue
        unique_boxes.append(box)
    return unique_boxes


def _extract_component_boxes(mask, image_shape):
    height, width = image_shape[:2]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 20:
            continue
        pad = max(4, int(min(w, h) * 0.25))
        boxes.append((max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)))
    return boxes


def get_text_boxes_from_threshold(img, threshold=180) -> list[Box]:
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Could not load image for threshold detection")

    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    masks = [cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)]
    for value in (threshold, max(140, threshold - 25), max(110, threshold - 60)):
        masks.append(cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)[1])

    boxes = []
    for mask in masks:
        boxes.extend(_extract_component_boxes(mask, gray.shape))
    return _prune_boxes(boxes, gray.shape)
