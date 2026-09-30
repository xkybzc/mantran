# Mantran

Translates manga pages: finds the speech bubbles, reads the Japanese, erases it and
letters the English translation back in. See [ROADMAP.md](ROADMAP.md) for the plan.

## Layout

```
backend/
  app/
    main.py          command line entry point (+ exports the API app)
    api.py           FastAPI endpoints
    pipeline.py      runs the steps below in order
    detection.py     find speech bubbles and text blocks
    ocr.py           read the original text
    translation.py   translate it
    cleaning.py      erase the original lettering
    typesetting.py   fit and draw the translation in the bubble
    fonts/           bundled lettering font (OFL)
  tests/             pytest tests and sample pages
  requirements.txt
frontend/
  index.html         upload on the left, translated page on the right
  styles.css
  app.js             talks to /api/translate
```

## Running

From the `backend/` folder, with the project venv:

```
pip install -r requirements.txt
python -m app.main -i tests/image2.png -o tests/image_inpainted.png
uvicorn app.main:app          # website at http://127.0.0.1:8000
pytest tests
```

The backend serves the `frontend/` folder, so the website and the API run from the
one `uvicorn` command. API docs are at http://127.0.0.1:8000/docs.

The first run downloads the bubble detector and OCR models from Hugging Face.
