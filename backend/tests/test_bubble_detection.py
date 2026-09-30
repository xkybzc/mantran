import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.cleaning import erase_region, find_bubble_interior, remove_text_regions, text_stroke_mask
from app.detection import LABEL_BUBBLE, TextRegion, get_text_boxes_from_threshold, group_detections
from app.typesetting import layout_text


def _bubble_page():
    """A white page with one outlined oval bubble holding some lettering."""
    img = np.full((300, 300, 3), 255, dtype=np.uint8)
    img[:, :60] = 90  # dark artwork beside the bubble
    cv2.ellipse(img, (160, 150), (100, 120), 0, 0, 360, (0, 0, 0), 3)
    for y in (90, 130, 170):
        cv2.putText(img, "TEXT", (125, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    region = TextRegion([(115, 60, 215, 185)], bubble_box=(57, 27, 263, 273))
    return img, region


def test_detects_multiple_bubble_regions():
    img = np.full((400, 400, 3), 255, dtype=np.uint8)

    bubble_specs = [
        ((50, 50), (180, 140)),
        ((210, 60), (330, 160)),
        ((80, 220), (320, 340)),
    ]

    for (x0, y0), (x1, y1) in bubble_specs:
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), 2)
        cv2.rectangle(img, (x0 + 5, y0 + 5), (x1 - 5, y1 - 5), (240, 240, 240), -1)
        cv2.putText(img, "Hello", (x0 + 15, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    boxes = get_text_boxes_from_threshold(img, threshold=210)

    assert len(boxes) >= 3
    assert all(x1 - x0 > 20 and y1 - y0 > 20 for x0, y0, x1, y1 in boxes[:3])


def test_group_detections_attaches_text_to_its_bubble():
    detections = [
        (LABEL_BUBBLE, 0.9, (0, 0, 100, 100)),
        (LABEL_BUBBLE, 0.9, (200, 0, 300, 100)),
        (1, 0.9, (220, 10, 240, 90)),
        (1, 0.9, (250, 10, 270, 90)),
        (2, 0.9, (120, 150, 180, 190)),
    ]

    regions = group_detections(detections)

    in_bubble = [r for r in regions if r.bubble_box == (200, 0, 300, 100)]
    assert len(in_bubble) == 1
    # Vertical columns read right to left.
    assert in_bubble[0].text_boxes == [(250, 10, 270, 90), (220, 10, 240, 90)]
    assert any(r.bubble_box is None and r.text_box == (120, 150, 180, 190) for r in regions)


def test_stroke_mask_stays_tight_around_text():
    img, region = _bubble_page()

    mask = text_stroke_mask(img, region.text_box, region.bubble_box)

    assert mask.shape == img.shape[:2]
    assert mask.any()
    x0, y0, x1, y1 = region.text_box
    assert np.count_nonzero(mask) < (x1 - x0) * (y1 - y0) * 0.6
    # The bubble outline is not part of the mask.
    assert not mask[150, 58:64].any()


def test_erase_region_clears_text_and_keeps_outline():
    img, region = _bubble_page()

    interior = find_bubble_interior(img, region)
    assert interior is not None and interior.is_flat

    result = erase_region(img, region, interior)

    x0, y0, x1, y1 = region.text_box
    assert result[y0:y1, x0:x1].min() > 240
    assert result[150, 58:64].min() < 50  # left side of the outline
    assert (result[:, :50] == 90).all()  # artwork untouched


def test_remove_text_regions_inpaints_masked_area():
    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    cv2.putText(img, "abc", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[60:120, 20:140] = 255

    result = remove_text_regions(img, mask, radius=3)

    assert result.shape == img.shape
    assert result[60:120, 20:140].min() > 200


def test_layout_fits_inside_bubble_shape():
    area = np.zeros((200, 160), dtype=np.uint8)
    cv2.ellipse(area, (80, 100), (70, 90), 0, 0, 360, 1, -1)
    area = area.astype(bool)

    layout = layout_text("Don't let go even if you die!", (10, 20, 170, 220), area, max_size=40, min_size=8)

    assert layout is not None
    assert len(layout.lines) >= 2
    assert all(line == line.upper() for line, _, _ in layout.lines)
    assert " ".join(line for line, _, _ in layout.lines) == "DON'T LET GO EVEN IF YOU DIE!"
    # Every line shares the same centre axis.
    assert len({round(cx) for _, cx, _ in layout.lines}) == 1
