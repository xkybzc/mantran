import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main


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

    boxes = main.get_text_boxes_from_threshold(img, threshold=210)

    assert len(boxes) >= 3
    assert all(x1 - x0 > 20 and y1 - y0 > 20 for x0, y0, x1, y1 in boxes[:3])


def test_mask_stays_tight_around_text():
    img = np.full((240, 240, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 50), (180, 140), (0, 0, 0), 2)
    cv2.rectangle(img, (45, 55), (175, 135), (240, 240, 240), -1)
    cv2.putText(img, "Hi", (60, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

    mask, boxes = main.build_mask_from_threshold(img, threshold=220)

    assert mask.shape == img.shape[:2]
    assert boxes

    bubble_area = (180 - 40) * (140 - 50)
    mask_area = int(np.count_nonzero(mask[50:140, 40:180]))
    assert mask_area < bubble_area * 0.35
