import argparse
import os
import textwrap

import cv2
import numpy as np
import pytesseract
from deep_translator import GoogleTranslator

try:
    from manga_ocr import MangaOcr
except ImportError:
    MangaOcr = None


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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 5000 and w > 3 and h > 3:
            if w < gray.shape[1] * 0.8 and h < gray.shape[0] * 0.8:
                pad = max(2, int(min(w, h) * 0.2))
                x0 = max(0, x - pad)
                y0 = max(0, y - pad)
                x1 = min(gray.shape[1], x + w + pad)
                y1 = min(gray.shape[0], y + h + pad)
                boxes.append((x0, y0, x1, y1))

    if not boxes:
        _, fallback = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        fallback = cv2.morphologyEx(fallback, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 3 and h > 3:
                boxes.append((x, y, x + w, y + h))

    return boxes


def build_mask_from_threshold(img, threshold=180):
    boxes = get_text_boxes_from_threshold(img, threshold=threshold)
    if isinstance(img, str):
        img = cv2.imread(img, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Could not load image for threshold mask")

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
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

        text = data["text"][i].strip()
        if not text:
            continue

        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        if w > 0 and h > 0:
            boxes.append((x, y, x + w, y + h))
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    if not boxes:
        return build_mask_from_threshold(img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask, boxes


def translate_text(text, src="ja", dst="en"):
    if not text or not text.strip():
        return ""
    try:
        return GoogleTranslator(source=src, target=dst).translate(text)
    except Exception as exc:
        print(f"Translation failed: {exc}")
        return text


def overlay_translated_text(image, text, boxes):
    if not text or not boxes:
        return image

    img = image.copy()
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x0, y0, x1, y1 = box
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.45, min(0.9, width / 220))
    thickness = 2
    lines = textwrap.wrap(text, width=max(8, int(width / 8)))

    y_cursor = y0 + 20
    for line in lines:
        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = x0 + max(5, int((width - text_w) / 2))
        cv2.putText(
            img,
            line,
            (text_x, y_cursor),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
        y_cursor += text_h + 8
        if y_cursor > y1 - 10:
            break

    return img


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

    result = cv2.inpaint(img, mask, radius, flags=cv2.INPAINT_NS)
    if translated_text and boxes:
        result = overlay_translated_text(result, translated_text, boxes)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, result)
    return output_path


def extract_text_from_image(image_path):
    if MangaOcr is not None:
        try:
            mocr = MangaOcr()
            return mocr(image_path)
        except Exception as exc:
            print(f"manga OCR failed: {exc}")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Inpaint manga text with Navier–Stokes")
    parser.add_argument("--input", "-i", default="backend/tests/image.png", help="Input manga image")
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
        help="Navier–Stokes inpainting radius",
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

    if args.use_threshold_mask:
        mask, boxes = build_mask_from_threshold(args.input)
    else:
        mask, boxes = build_mask_from_tesseract(args.input)

    extracted = extract_text_from_image(args.input)
    translated_text = ""
    if extracted is not None:
        print("Extracted text:\n", extracted)
        translated_text = translate_text(extracted, src=args.source_lang, dst=args.target_lang)
        print("Translated text:\n", translated_text)

    out_path = inpaint_ns(
        args.input,
        args.output,
        mask=mask,
        radius=args.radius,
        translated_text=translated_text,
        boxes=boxes,
    )
    print("Saved inpainted image:", out_path)

    return out_path


if __name__ == "__main__":
    main()
