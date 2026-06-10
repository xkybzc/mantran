#Read more on manga-ocr documentation

from manga_ocr import MangaOcr

def main():
    mocr = MangaOcr()
    text = mocr("/Users/lkbm/Documents/mantran/backend/tests/image.png")
    print(text)

if __name__ == "__main__":
    main()