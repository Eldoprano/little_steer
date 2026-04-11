"""Flask viewer for labeled reasoning traces.

Run with:
    uv run python viewer/app.py
Or:
    uv run flask --app viewer/app run --debug
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as `python viewer/app.py` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, abort, jsonify, render_template

from viewer.data_loader import (
    DATA_DIR,
    LABEL_GROUPS,
    entry_summary,
    get_entry,
    get_filter_options,
    load_all_entries,
    render_reasoning_html,
)

app = Flask(__name__, template_folder="templates", static_folder="static")


def _load() -> list[dict]:
    return load_all_entries(DATA_DIR)


@app.route("/")
def index():
    entries = _load()
    summaries = [entry_summary(e) for e in entries]
    filters = get_filter_options(entries)
    return render_template(
        "index.html",
        summaries_json=json.dumps(summaries),
        filters=filters,
        total=len(summaries),
        label_groups=LABEL_GROUPS,
    )


@app.route("/entry/<entry_id>")
def entry_detail(entry_id: str):
    entry = get_entry(entry_id, DATA_DIR)
    if entry is None:
        abort(404)

    # Find messages by role
    messages = entry.get("messages") or []
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    reasoning_msg = next((m["content"] for m in messages if m["role"] == "reasoning"), None)
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")

    # Find reasoning message index
    reasoning_idx = next(
        (i for i, m in enumerate(messages) if m["role"] == "reasoning"), None
    )

    # Filter annotations for reasoning message
    annotations = entry.get("annotations") or []
    reasoning_annotations = (
        [a for a in annotations if a.get("message_idx") == reasoning_idx]
        if reasoning_idx is not None
        else []
    )

    reasoning_html = ""
    if reasoning_msg is not None:
        reasoning_html = render_reasoning_html(reasoning_msg, reasoning_annotations)

    metadata = entry.get("metadata") or {}
    assessment = metadata.get("assessment") or {}

    # Which groups are actually used in this entry, and how many sentences each
    used_groups: set[str] = set()
    group_counts: dict[str, int] = {}
    from viewer.data_loader import label_group
    for ann in reasoning_annotations:
        for lbl in (ann.get("labels") or []):
            gid = label_group(lbl)
            used_groups.add(gid)
            group_counts[gid] = group_counts.get(gid, 0) + 1

    # Average safety score across annotated sentences
    scores = [a.get("score", 0) for a in reasoning_annotations if a.get("score") is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    return render_template(
        "entry.html",
        entry=entry,
        user_msg=user_msg,
        reasoning_html=reasoning_html,
        assistant_msg=assistant_msg,
        assessment=assessment,
        metadata=metadata,
        annotations=reasoning_annotations,
        used_groups=sorted(used_groups),
        group_counts=group_counts,
        avg_score=avg_score,
        label_groups=LABEL_GROUPS,
    )


@app.route("/api/entries")
def api_entries():
    entries = _load()
    return jsonify([entry_summary(e) for e in entries])


if __name__ == "__main__":
    if not DATA_DIR.exists():
        print(f"Data dir not found: {DATA_DIR}")
        print("Label some files first, then run the viewer.")
    else:
        n = sum(1 for _ in DATA_DIR.glob("*.jsonl"))
        print(f"Serving {n} JSONL file(s) from {DATA_DIR}")
    app.run(debug=True, port=5050)
