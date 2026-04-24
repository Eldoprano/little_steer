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

from flask import Flask, abort, jsonify, render_template, request

from viewer.data_loader import (
    ALL_LABELS,
    DATA_DIR,
    LABEL_GROUPS,
    compute_stats,
    get_grouped_summaries,
    get_all_versions,
    get_entry,
    get_filter_options,
    load_all_entries,
    render_reasoning_html,
)

import argparse

app = Flask(__name__, template_folder="templates", static_folder="static")

# Globals that can be overridden by CLI
CURRENT_DATA_DIR = DATA_DIR
CLI_WORK_ORDER_FILTER: dict[str, set[str]] | None = None
CLI_WORK_ORDER_NAME: str | None = None


def _get_available_work_orders() -> list[str]:
    """List .json files in the project root that look like work orders."""
    return [p.name for p in Path(__file__).parent.parent.glob("work_order*.json")]


def _load_work_order(name: str) -> dict[str, set[str]] | None:
    path = Path(__file__).parent.parent / name
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            wo = json.load(f)
            if "per_file" in wo:
                return {fname: set(ids) for fname, ids in wo["per_file"].items()}
            if "flat_order" in wo:
                fmap = {}
                for item in wo["flat_order"]:
                    fname, eid = item["file"], item["id"]
                    fmap.setdefault(fname, set()).add(eid)
                return fmap
    except Exception:
        pass
    return None


def _load() -> list[dict]:
    # Use query param if present, otherwise fallback to CLI override
    wo_name = request.args.get("work_order")
    if wo_name:
        fmap = _load_work_order(wo_name)
        if fmap is not None:
            return load_all_entries(CURRENT_DATA_DIR, filter_map=fmap)
    
    return load_all_entries(CURRENT_DATA_DIR, filter_map=CLI_WORK_ORDER_FILTER)


@app.context_processor
def inject_work_orders():
    return {
        "available_work_orders": _get_available_work_orders(),
        "current_work_order": request.args.get("work_order") or CLI_WORK_ORDER_NAME
    }


@app.route("/")
def index():
    entries = _load()
    summaries = get_grouped_summaries(entries)
    filters = get_filter_options(entries)
    return render_template(
        "index.html",
        summaries_json=json.dumps(summaries),
        filters=filters,
        total=len(summaries),
        label_groups=LABEL_GROUPS,
        label_groups_json=json.dumps(LABEL_GROUPS),
        all_labels=ALL_LABELS,
        all_labels_json=json.dumps(ALL_LABELS),
    )


@app.route("/entry/<entry_id>/<reasoning_hash>")
def entry_detail(entry_id: str, reasoning_hash: str):
    versions = get_all_versions(entry_id, reasoning_hash, CURRENT_DATA_DIR)
    if not versions:
        abort(404)

    requested_run_key = request.args.get("run_key")
    if requested_run_key:
        entry = next((v for v in versions if v.get("run_key") == requested_run_key), versions[0])
    else:
        entry = versions[0]

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

    # Which labels/groups are actually used in this entry
    used_labels_ordered: list[str] = []
    used_groups: set[str] = set()
    group_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    from viewer.data_loader import label_group
    for ann in reasoning_annotations:
        for lbl in (ann.get("labels") or []):
            gid = label_group(lbl)
            used_groups.add(gid)
            group_counts[gid] = group_counts.get(gid, 0) + 1
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            if lbl not in used_labels_ordered:
                used_labels_ordered.append(lbl)

    # Average safety score across annotated sentences
    scores = [a.get("score", 0) for a in reasoning_annotations if a.get("score") is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    return render_template(
        "entry.html",
        entry=entry,
        versions=versions,
        user_msg=user_msg,
        reasoning_html=reasoning_html,
        assistant_msg=assistant_msg,
        assessment=assessment,
        metadata=metadata,
        annotations=reasoning_annotations,
        used_groups=sorted(used_groups),
        used_labels=used_labels_ordered,
        label_counts=label_counts,
        group_counts=group_counts,
        avg_score=avg_score,
        label_groups=LABEL_GROUPS,
        all_labels=ALL_LABELS,
        reasoning_hash=reasoning_hash,
    )


@app.route("/stats")
def stats():
    entries = _load()
    exclude_safe = request.args.get("exclude_safe", "0") == "1"
    if exclude_safe:
        entries = [
            e for e in entries
            if (e.get("metadata") or {}).get("prompt_safety", "").lower() != "safe"
        ]
    s = compute_stats(entries)
    return render_template(
        "stats.html",
        stats=s,
        label_groups=LABEL_GROUPS,
        exclude_safe=exclude_safe,
    )


@app.route("/api/entries")
def api_entries():
    entries = _load()
    return jsonify(get_grouped_summaries(entries))


@app.route("/api/entry/<entry_id>/<reasoning_hash>/compare")
def api_entry_compare(entry_id: str, reasoning_hash: str):
    """Return all labeled versions of an entry (same id, all source files)."""
    versions = get_all_versions(entry_id, reasoning_hash, CURRENT_DATA_DIR)
    result = []
    for entry in versions:
        messages = entry.get("messages") or []
        reasoning_msg = next((m["content"] for m in messages if m["role"] == "reasoning"), None)
        reasoning_idx = next((i for i, m in enumerate(messages) if m["role"] == "reasoning"), None)
        annotations = entry.get("annotations") or []
        reasoning_annotations = (
            [a for a in annotations if a.get("message_idx") == reasoning_idx]
            if reasoning_idx is not None
            else []
        )
        reasoning_html = render_reasoning_html(reasoning_msg or "", reasoning_annotations) if reasoning_msg else ""
        metadata = entry.get("metadata") or {}
        assessment = metadata.get("assessment") or {}
        scores = [a.get("score", 0) for a in reasoning_annotations if a.get("score") is not None]
        avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0
        result.append({
            "source_file": entry.get("source_file", ""),
            "run_key": entry.get("run_key", ""),
            "dataset_name": entry.get("_dataset_name", entry.get("source_file", "")),
            "judge": entry.get("judge", ""),
            "model": entry.get("model", ""),
            "reasoning_html": reasoning_html,
            "n_annotations": len(reasoning_annotations),
            "avg_score": avg_score,
            "trajectory": assessment.get("trajectory", ""),
            "alignment": assessment.get("alignment", ""),
        })
    return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labeled reasoning trace viewer.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help=f"Directory containing JSONL files (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--work-order",
        type=Path,
        help="Path to a work order JSON file to filter entries.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5050,
        help="Port to run the server on (default: 5050)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()
    CURRENT_DATA_DIR = args.data_dir

    if args.work_order:
        if not args.work_order.exists():
            print(f"Work order file not found: {args.work_order}")
            sys.exit(1)
        CLI_WORK_ORDER_NAME = args.work_order.name
        with open(args.work_order, encoding="utf-8") as f:
            wo = json.load(f)
            # Use 'per_file' if available, otherwise 'flat_order'
            if "per_file" in wo:
                CLI_WORK_ORDER_FILTER = {
                    fname: set(ids) for fname, ids in wo["per_file"].items()
                }
                print(f"Loaded CLI work order filter for {len(CLI_WORK_ORDER_FILTER)} file(s).")
            elif "flat_order" in wo:
                CLI_WORK_ORDER_FILTER = {}
                for item in wo["flat_order"]:
                    fname = item["file"]
                    eid = item["id"]
                    if fname not in CLI_WORK_ORDER_FILTER:
                        CLI_WORK_ORDER_FILTER[fname] = set()
                    CLI_WORK_ORDER_FILTER[fname].add(eid)
                print(f"Loaded CLI work order filter for {len(wo['flat_order'])} entries.")
            else:
                print("Invalid work order format: missing 'per_file' or 'flat_order'.")
                sys.exit(1)

    if not CURRENT_DATA_DIR.exists():
        print(f"Data dir not found: {CURRENT_DATA_DIR}")
        print("Label some files first, then run the viewer.")
    else:
        n = (
            sum(1 for _ in CURRENT_DATA_DIR.glob("*.jsonl"))
            if CURRENT_DATA_DIR.is_dir()
            else 1
        )
        print(f"Serving {n} JSONL file(s) from {CURRENT_DATA_DIR}")

    app.run(debug=args.debug, port=args.port)
