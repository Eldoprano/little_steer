# Viewer

Flask web app for inspecting labeled reasoning traces. Renders color-coded annotation spans with filtering by model, dataset, trajectory, and alignment.

## Quick start

```bash
cd 2_labeling/2b_sentence
uv run python viewer/app.py
```

Then open `http://localhost:5050` in a browser.

Reads JSONL files from `data/labeled/` (configurable via `DATA_DIR` in `data_loader.py`).

## Files

| File | Purpose |
|---|---|
| `app.py` | Flask routes: entry list, entry detail, filter API |
| `data_loader.py` | Loads and processes labeled JSONL files, renders annotation spans as HTML |
| `templates/` | Jinja2 HTML templates |
| `static/` | CSS and JS for the viewer UI |
