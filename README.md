# Mantran

Translates manga into English or Vietnamese: finds the speech bubbles, reads the
original text, erases it and letters the translation back in. Works on a single
uploaded page or on a whole chapter from MangaDex. See [ROADMAP.md](ROADMAP.md)
for the plan.

Source languages: Japanese, Korean, Chinese (simplified and traditional), English,
Spanish, Portuguese, Indonesian, French, Italian, German, Russian, Turkish, Polish
and Vietnamese.

## Layout

```
backend/
  app/
    main.py          command line entry point (+ exports the API app)
    api.py           FastAPI endpoints; also serves the frontend
    pipeline.py      one page: detect -> OCR -> translate -> erase -> letter
    detection.py     find speech bubbles and text blocks
    ocr.py           read the original text (manga-ocr for Japanese, Tesseract otherwise)
    translation.py   translate it (Google, MyMemory, or a local model)
    cleaning.py      erase the original lettering
    typesetting.py   fit and draw the translation in the bubble
    languages.py     supported languages and their OCR / translator codes
    mangadex.py      MangaDex API client
    chapters.py      whole-chapter jobs, page cache and ZIP export
    fonts/           bundled lettering font (OFL)
  data/              downloaded OCR data and translated chapters (not in git)
  tests/             pytest tests and sample pages
  requirements.txt
frontend/
  index.html         Single page and MangaDex chapter views
  styles.css
  app.js
```

## Setup

1. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (needed for
   every source language except Japanese). Language data is downloaded automatically
   into `backend/data/tessdata` the first time a language is used.
2. From the `backend/` folder, with the project venv: `pip install -r requirements.txt`

The first run also downloads the bubble detector and OCR models from Hugging Face.

## Running

From the `backend/` folder:

```
python -m app.main -i tests/image2.png -o tests/image_inpainted.png
python -m app.main -i page.png --source-lang es --target-lang vi
python -m app.main --chapter https://mangadex.org/chapter/<id> --target-lang en
uvicorn app.main:app          # website at http://127.0.0.1:8000
pytest tests
```

Translated chapters are cached in `backend/data/chapters/`, so translating the same
chapter again is instant and an interrupted chapter picks up where it stopped.

## Translation backends

Tried in order until one works:

| Name       | Notes                                                                  |
|------------|------------------------------------------------------------------------|
| `google`   | Best quality. Free web endpoint, rate limited per IP.                  |
| `mymemory` | 5,000 characters a day, 50,000 with the `MYMEMORY_EMAIL` env var set.  |
| `local`    | NLLB-200 on your machine: offline and unlimited, ~2.5 GB download.     |

Set `MANTRAN_TRANSLATORS` to change the order, e.g. `set MANTRAN_TRANSLATORS=local`.

## MangaDex

Chapters are fetched through the public MangaDex API, following its usage rules
(identifying User-Agent, staying under the rate limit, reporting image fetches to
MangaDex@Home). Scanlation groups are credited in the chapter list. For personal
reading only; chapters hosted on external sites (e.g. official publishers) can't be
downloaded.
