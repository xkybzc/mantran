import argparse
import base64
import os
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

try:
    from deep_translator import GoogleTranslator
except ImportError:  # pragma: no cover - optional dependency
    GoogleTranslator = None

try:
    from manga_ocr import MangaOcr
except ImportError:  # pragma: no cover - optional dependency
    MangaOcr = None

try:
    from translate import Translator
except ImportError:  # pragma: no cover - optional dependency
    Translator = None

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None
    File = None
    Form = None
    HTTPException = None
    UploadFile = None


def _configure_tesseract():
    if os.name == "nt":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
            os.path.join(os.environ.get("ProgramFiles", ""), "Tesseract-OCR", "tesseract.exe"),
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                return True
    return False


def _tesseract_available():
    if _configure_tesseract():
        return True
    try:
        pytesseract.get_tesseract_version()
        return True
    except (pytesseract.TesseractNotFoundError, OSError):
        return False


def _box_overlap_ratio(box_a, box_b):
    x0 = max(box_a[0], box_b[0])
    y0 = max(box_a[1], box_b[1])
    x1 = min(box_a[2], box_b[2])
    y1 = min(box_a[3], box_b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0

    overlap_area = (x1 - x0) * (y1 - y0)
    smaller_area = min(
        (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]),
        (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]),
    )
    if smaller_area <= 0:
        return 0.0
    return overlap_area / smaller_area


def _prune_boxes(boxes, image_shape):
    height, width = image_shape[:2]
    min_area = max(25, int(width * height * 0.00035))
    max_area = max(100, int(width * height * 0.45))

    filtered = []
    for x0, y0, x1, y1 in boxes:
        if x1 <= x0 or y1 <= y0:
            continue
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
        if any(_box_overlap_ratio(box, existing) > 0.75 for existing in unique_boxes):
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
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(width, x + w + pad)
        y1 = min(height, y + h + pad)
        boxes.append((x0, y0, x1, y1))
    return boxes


def get_text_boxes_from_threshold(img, threshold=180):
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Could not load image for threshold mask")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    masks = [adaptive]
    for value in (threshold, max(140, threshold - 25), max(110, threshold - 60)):
        _, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY_INV)
        masks.append(binary)

    boxes = []
    for mask in masks:
        boxes.extend(_extract_component_boxes(mask, gray.shape))

    if not boxes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        _, fallback = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 3 and h > 3:
                boxes.append((x, y, x + w, y + h))

    return _prune_boxes(boxes, gray.shape)


def build_mask_from_threshold(img, threshold=180):
    boxes = get_text_boxes_from_threshold(img, threshold=threshold)
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Could not load image for threshold mask")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        pad = max(2, int(min(x1 - x0, y1 - y0) * 0.08))
        x0 = max(0, x0 + pad)
        y0 = max(0, y0 + pad)
        x1 = min(img.shape[1], x1 - pad)
        y1 = min(img.shape[0], y1 - pad)
        if x1 > x0 and y1 > y0:
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask, boxes


def build_mask_from_tesseract(img, conf_threshold=40):
    if not _tesseract_available():
        print("Tesseract is not available; using a threshold-based mask fallback.")
        return build_mask_from_threshold(img)

    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Could not load image for mask creation")

    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not available; using a threshold-based mask fallback.")
        return build_mask_from_threshold(img)
    except Exception as exc:
        print(f"OCR mask failed ({exc}); using a threshold-based mask fallback.")
        return build_mask_from_threshold(img)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    boxes = []
    for i, conf in enumerate(data["conf"]):
        try:
            if int(conf) <= conf_threshold:
                continue
        except (ValueError, TypeError):
            continue

        text = str(data["text"][i]).strip()
        if not text:
            continue

        x, y, w, h = (
            int(data["left"][i]),
            int(data["top"][i]),
            int(data["width"][i]),
            int(data["height"][i]),
        )
        if w > 0 and h > 0:
            pad = max(6, int(min(w, h) * 0.3))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(img.shape[1], x + w + pad)
            y1 = min(img.shape[0], y + h + pad)
            boxes.append((x0, y0, x1, y1))
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

    threshold_mask, threshold_boxes = build_mask_from_threshold(img)
    mask = cv2.bitwise_or(mask, threshold_mask)
    boxes.extend(threshold_boxes)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask, boxes


def translate_text(text, src="ja", dst="en"):
    if not text or not text.strip():
        return ""

    if Translator is not None:
        try:
            translator = Translator(from_lang=src, to_lang=dst)
            return translator.translate(text)
        except Exception as exc:  # pragma: no cover - network fallback
            print(f"translate package translation failed: {exc}")

    if GoogleTranslator is not None:
        try:
            return GoogleTranslator(source=src, target=dst).translate(text)
        except Exception as exc:  # pragma: no cover - network fallback
            print(f"deep-translator translation failed: {exc}")

    print("No translation backend is available; returning the original text.")
    return text


def _get_font(size):
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "Fonts", "arial.ttf"),
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            continue
    return ImageFont.load_default()


def overlay_translated_text(image, text, boxes):
    if not text or not boxes:
        return image

    img = image.copy()
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    width = max(1, x1 - x0 - 20)
    height = max(1, y1 - y0 - 20)

    pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    text_color = (0, 0, 0)

    chosen_font = None
    chosen_lines = None
    for size in [26, 22, 18, 16, 14, 12]:
        font = _get_font(size)
        wrap_width = max(8, int(width / max(8, size // 2)))
        lines = textwrap.wrap(text, width=wrap_width)
        line_heights = []
        max_line_width = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_h = bbox[3] - bbox[1]
            line_heights.append(line_h)
            max_line_width = max(max_line_width, bbox[2] - bbox[0])
        total_height = sum(line_heights) + max(0, len(lines) - 1) * 4
        if max_line_width <= width - 12 and total_height <= height - 12:
            chosen_font = font
            chosen_lines = lines
            break

    if chosen_font is None:
        chosen_font = _get_font(12)
        chosen_lines = textwrap.wrap(text, width=max(8, int(width / 10)))

    y_cursor = y0 + 12
    for line in chosen_lines:
        bbox = draw.textbbox((0, 0), line, font=chosen_font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = x0 + max(4, int((width - text_w) / 2))
        draw.text((text_x, y_cursor), line, font=chosen_font, fill=text_color)
        y_cursor += text_h + 4

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _normalize_mask(mask, image_shape):
    mask = mask.astype(np.uint8)
    if mask.ndim == 2:
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    if mask.shape != image_shape[:2]:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def inpaint_ns(input_path, output_path, mask=None, radius=3, translated_text="", boxes=None):
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load input image: {input_path}")

    if mask is None:
        mask, boxes = build_mask_from_tesseract(img)
    elif isinstance(mask, str):
        mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise FileNotFoundError(f"Could not load mask image: {mask}")
        mask = mask_img

    mask = _normalize_mask(mask, img.shape)
    if translated_text and boxes:
        result = img.copy()
        result = overlay_translated_text(result, translated_text, boxes)
    else:
        flags = cv2.INPAINT_TELEA if radius <= 0 else cv2.INPAINT_NS
        result = cv2.inpaint(img, mask, radius, flags=flags)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    success = cv2.imwrite(output_path, result)
    if not success:
        raise RuntimeError(f"Failed to write image: {output_path}")
    return output_path


def extract_text_from_image(image_path, boxes=None):
    if MangaOcr is not None:
        try:
            mocr = MangaOcr()
            return mocr(image_path)
        except Exception as exc:
            print(f"manga OCR failed: {exc}")

    if boxes:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is not None:
            texts = []
            for x0, y0, x1, y1 in boxes:
                crop = image[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                for lang in ["jpn", "eng"]:
                    try:
                        text = pytesseract.image_to_string(crop, lang=lang, config="--psm 6")
                    except Exception:
                        text = ""
                    text = text.strip()
                    if text:
                        texts.append(text)
                        break
            if texts:
                return "\n".join(texts)

    try:
        data = pytesseract.image_to_data(
            image_path,
            output_type=pytesseract.Output.DICT,
            lang="eng",
        )
        texts = [text.strip() for text in data.get("text", []) if text and text.strip()]
        if texts:
            return " ".join(texts)
    except Exception as exc:
        print(f"Tesseract OCR failed: {exc}")

    return None


def process_image(input_path, output_path, source_lang="ja", target_lang="en", use_threshold_mask=False):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if use_threshold_mask:
        mask, boxes = build_mask_from_threshold(input_path)
    else:
        mask, boxes = build_mask_from_tesseract(input_path)

    extracted = extract_text_from_image(input_path, boxes=boxes)
    translated_text = ""
    if extracted is not None:
        translated_text = translate_text(extracted, src=source_lang, dst=target_lang)

    out_path = inpaint_ns(
        input_path,
        output_path,
        mask=mask,
        radius=3,
        translated_text=translated_text,
        boxes=boxes,
    )
    return out_path, {"boxes": boxes, "extracted_text": extracted, "translated_text": translated_text}


def parse_args():
    parser = argparse.ArgumentParser(description="Inpaint manga text with Navier–Stokes")
    parser.add_argument(
        "--input",
        "-i",
        default="backend/tests/image.png",
        help="Input manga image path (for example: backend/tests/image.png or backend/tests/image1.png)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="backend/tests/image_inpainted.png",
        help="Inpainted output image",
    )
    parser.add_argument("--mask", "-m", default=None, help="Optional mask image path")
    parser.add_argument(
        "--radius",
        "-r",
        type=int,
        default=3,
        help="Inpainting radius",
    )
    parser.add_argument(
        "--use-threshold-mask",
        action="store_true",
        help="Build the inpainting mask from thresholding instead of OCR boxes",
    )
    parser.add_argument("--source-lang", default="ja", help="Source language for translation")
    parser.add_argument("--target-lang", default="en", help="Target language for translation")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Input image:", args.input)
    out_path, metadata = process_image(
        args.input,
        args.output,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        use_threshold_mask=args.use_threshold_mask,
    )
    print("Extracted text:\n", metadata["extracted_text"])
    print("Translated text:\n", metadata["translated_text"])
    print("Saved inpainted image:", out_path)
    return out_path


if FastAPI is not None:
    app = FastAPI(title="Mantran", version="0.1.0")

    @app.get("/health")
    def health_check():
        return {"status": "ok"}

    @app.post("/api/translate")
    async def translate_image(
        file: UploadFile = File(...),
        source_lang: str = Form("ja"),
        target_lang: str = Form("en"),
    ):
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        suffix = Path(file.filename).suffix or ".png"
        with tempfile.TemporaryDirectory(prefix="mantran_", dir="backend") as temp_dir:
            input_path = os.path.join(temp_dir, f"upload{suffix}")
            output_path = os.path.join(temp_dir, f"translated{suffix}")
            with open(input_path, "wb") as handle:
                handle.write(await file.read())

            out_path, metadata = process_image(
                input_path,
                output_path,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            with open(out_path, "rb") as handle:
                image_bytes = handle.read()

        return {
            "output_image": base64.b64encode(image_bytes).decode("ascii"),
            "mime_type": "image/png",
            "extracted_text": metadata["extracted_text"],
            "translated_text": metadata["translated_text"],
        }


if __name__ == "__main__":
    main()
