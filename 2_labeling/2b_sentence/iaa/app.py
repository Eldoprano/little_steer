"""IAA viewer Flask app.

Run with:
    uv run python iaa/app.py
from the 2b_sentence directory, or via the CLI.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

# Allow running as `python iaa/app.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, render_template, request

from iaa.compute import (
    MODES,
    compute_agreement_matrix,
    count_shared_all,
    load_iaa_data,
)

DATA_PATH = Path(__file__).parents[3] / "data" / "dataset.jsonl"

app = Flask(
    __name__,
    template_folder="templates",
    # Share CSS with the existing viewer
    static_folder=str(Path(__file__).parent.parent / "viewer" / "static"),
    static_url_path="/static",
)

_state: dict = {
    "entry_data": None,
    "annotator_counts": None,
    "loading": True,
    "error": None,
}
_pair_cache: dict = {}
_lock = threading.Lock()


def _load() -> None:
    try:
        ed, ac = load_iaa_data(DATA_PATH)
        with _lock:
            _state["entry_data"] = ed
            _state["annotator_counts"] = ac
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
    finally:
        with _lock:
            _state["loading"] = False


@app.route("/")
def index():
    return render_template("iaa.html")


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "loading": _state["loading"],
            "error": _state["error"],
            "ready": not _state["loading"] and _state["error"] is None,
        })


@app.route("/api/annotators")
def api_annotators():
    with _lock:
        if _state["loading"]:
            return jsonify({"loading": True}), 202
        if _state["error"]:
            return jsonify({"error": _state["error"]}), 500
        counts = dict(_state["annotator_counts"])

    annotators = sorted(
        [{"name": n, "count": c} for n, c in counts.items()],
        key=lambda x: -x["count"],
    )
    return jsonify({"annotators": annotators})


@app.route("/api/matrix", methods=["POST"])
def api_matrix():
    with _lock:
        if _state["loading"]:
            return jsonify({"loading": True}), 202
        if _state["error"]:
            return jsonify({"error": _state["error"]}), 500
        ed = _state["entry_data"]

    body = request.json or {}
    selected: list[str] = body.get("annotators", [])
    mode: str = body.get("mode", "score")
    if mode not in MODES:
        mode = "score"

    if len(selected) < 2:
        return jsonify({"annotators": selected, "pairs": {}, "mode": mode})

    with _lock:
        result = compute_agreement_matrix(ed, selected, mode, _pair_cache)

    return jsonify(result)


@app.route("/api/shared", methods=["POST"])
def api_shared():
    with _lock:
        if _state["loading"]:
            return jsonify({"loading": True}), 202
        ed = _state["entry_data"]

    body = request.json or {}
    selected: list[str] = body.get("annotators", [])
    count = count_shared_all(ed, selected)
    return jsonify({"shared": count})


if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        sys.exit(1)
    print(f"IAA viewer  →  http://localhost:5051")
    print(f"Dataset:       {DATA_PATH}")
    t = threading.Thread(target=_load, daemon=True)
    t.start()
    app.run(debug=False, port=5051)
