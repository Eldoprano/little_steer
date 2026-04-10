"""
Label Evolution Explorer — Flask backend
Run: uv run python serve.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs

from flask import Flask, jsonify, request, send_from_directory

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT.parent.parent.parent / "1_generating/data"
STATIC_DIR = ROOT / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

# ── Helpers ───────────────────────────────────────────────────────────────────

MIN_PARA = 250  # characters; mirrors data_loader default


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    filtered = [p.strip() for p in parts if len(p.strip()) >= MIN_PARA]
    return filtered if filtered else [p.strip() for p in parts if p.strip()]


def load_state(run_name: str) -> dict | None:
    p = RUNS_DIR / run_name / "state.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def load_responses(run_name: str) -> list[dict]:
    p = RUNS_DIR / run_name / "responses.jsonl"
    if not p.exists():
        return []
    results = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entry["_raw"] = json.loads(entry["raw"])
                results.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def get_record(composite_id: str) -> tuple[str, str, str] | None:
    """Return (user_prompt, reasoning_text, stem) for a composite_id."""
    if "::" not in composite_id:
        return None
    stem, sample_id = composite_id.split("::", 1)
    path = DATA_DIR / f"{stem}.jsonl"
    if not path.exists():
        return None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("id") != sample_id:
                continue
            user_prompt = ""
            reasoning = ""
            for msg in rec.get("messages", []):
                if msg.get("role") == "user":
                    user_prompt = msg.get("content", "").strip()
                elif msg.get("role") == "reasoning":
                    reasoning = msg.get("content", "").strip()
            return user_prompt, reasoning, stem
    return None


def get_paragraph(composite_id: str, sub_text_idx: int) -> dict | None:
    """Return paragraph info dict or None."""
    result = get_record(composite_id)
    if result is None:
        return None
    user_prompt, reasoning, stem = result
    paras = split_paragraphs(reasoning)
    if sub_text_idx >= len(paras):
        return None
    context_before = paras[sub_text_idx - 1] if sub_text_idx > 0 else None
    return {
        "composite_id": composite_id,
        "sub_text_idx": sub_text_idx,
        "stem": stem,
        "user_prompt": user_prompt,
        "paragraph": paras[sub_text_idx],
        "context_before": context_before,
    }


def list_all_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        if d.is_dir() and (d / "state.json").exists():
            runs.append(d.name)
    return runs


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/runs")
def api_runs():
    """List all runs with basic metadata."""
    results = []
    for run_name in list_all_runs():
        state = load_state(run_name)
        if state is None:
            continue
        active = state.get("taxonomy", {}).get("active", {})
        label_count = len(active)
        total_usage = sum(v.get("usage_count", 0) for v in active.values())
        processed_count = len(state.get("processed", {}))
        is_empty = "empty" in run_name.lower()
        results.append({
            "name": run_name,
            "label_count": label_count,
            "total_usage": total_usage,
            "processed_count": processed_count,
            "is_empty": is_empty,
            "created_at": state.get("created_at", ""),
        })
    return jsonify(results)


@app.route("/api/labels")
def api_labels():
    """
    Return merged label info for selected runs.
    Query: ?run=r1&run=r2  (repeatable)
    Response: list of label objects, each with per-run usage stats.
    """
    selected = request.args.getlist("run")
    if not selected:
        selected = list_all_runs()

    # label_name -> {description, runs: {run_name: {usage_count, created_at_step, label_id}}}
    merged: dict[str, dict] = {}

    for run_name in selected:
        state = load_state(run_name)
        if state is None:
            continue
        active = state.get("taxonomy", {}).get("active", {})
        for name, info in active.items():
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "description": info.get("description", ""),
                    "runs": {},
                }
            merged[name]["runs"][run_name] = {
                "usage_count": info.get("usage_count", 0),
                "created_at_step": info.get("created_at_step", -1),
                "label_id": info.get("label_id", ""),
            }

    # Sort by total usage desc
    labels = list(merged.values())
    labels.sort(key=lambda x: -sum(v["usage_count"] for v in x["runs"].values()))
    return jsonify(labels)


@app.route("/api/label/<label_name>/samples")
def api_label_samples(label_name: str):
    """
    Return sample paragraphs for a label across selected runs.
    Query: ?run=r1&run=r2&limit=5
    Response: {creation_samples: [...], usage_samples: [...]}
    """
    selected = request.args.getlist("run")
    limit = int(request.args.get("limit", 5))
    if not selected:
        selected = list_all_runs()

    creation_samples = []  # paragraphs where label was CREATEd
    usage_samples = []  # paragraphs where label was applied

    for run_name in selected:
        responses = load_responses(run_name)
        for entry in responses:
            raw = entry.get("_raw", {})
            cid = entry.get("composite_id", "")
            idx = entry.get("sub_text_idx", 0)

            # Check creation
            for op in raw.get("operations", []):
                if op.get("type") == "CREATE" and op.get("name") == label_name:
                    para = get_paragraph(cid, idx)
                    if para:
                        para["run"] = run_name
                        para["justification"] = raw.get("justification", "")
                        creation_samples.append(para)
                    break

            # Check usage
            if label_name in raw.get("labels", []):
                para = get_paragraph(cid, idx)
                if para:
                    para["run"] = run_name
                    para["justification"] = raw.get("justification", "")
                    usage_samples.append(para)

    # Cap to limit
    creation_samples = creation_samples[:limit]
    usage_samples = usage_samples[:limit]

    return jsonify({
        "label_name": label_name,
        "creation_samples": creation_samples,
        "usage_samples": usage_samples,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5050
    print(f"Label Evolution Explorer → http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
