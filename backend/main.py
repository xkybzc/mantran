#Read more on manga-ocr documentation
import cv2
import numpy as np
import pytesseract

from manga_ocr import MangaOcr


def main():
    mocr = MangaOcr()
    text = mocr("backend/tests/image.png")
    return text

if __name__ == "__main__":
    main()
