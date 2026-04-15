"""data_loader.py — Load and process labeled JSONL entries for the viewer."""

from __future__ import annotations

import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any

_DEFAULT_DATA_DIR = Path(__file__).parents[3] / "data" / "2b_labeled" / "v5"
DATA_DIR: Path = Path(os.environ["VIEWER_DATA_DIR"]) if "VIEWER_DATA_DIR" in os.environ else _DEFAULT_DATA_DIR

# ── Label group metadata (loaded from taxonomy.json) ─────────────────────────

_TAXONOMY_PATH = Path(__file__).parents[2] / "taxonomy.json"


def _build_label_groups() -> dict[str, dict[str, Any]]:
    with open(_TAXONOMY_PATH, encoding="utf-8") as f:
        tax = json.load(f)
    result: dict[str, dict[str, Any]] = {}
    for group in tax["groups"]:
        colors = group["colors"]
        dark = colors.get("dark", colors["light"])
        result[group["id"]] = {
            "name": group["name"],
            "synthetic": group.get("synthetic", False),
            "color": colors["light"]["bg"],
            "border": colors["light"]["border"],
            "text": colors["light"]["text"],
            "dark_color": dark["bg"],
            "dark_border": dark["border"],
            "dark_text": dark["text"],
            "labels": [label["id"] for label in group["labels"]],
        }
    return result


LABEL_GROUPS: dict[str, dict[str, Any]] = _build_label_groups()

# Build label → group lookup
LABEL_TO_GROUP: dict[str, str] = {}
for gid, gdata in LABEL_GROUPS.items():
    for lbl in gdata["labels"]:
        LABEL_TO_GROUP[lbl] = gid

# Fallback is the last non-synthetic group
_FALLBACK_GROUP = next(
    gid for gid in reversed(list(LABEL_GROUPS.keys()))
    if not LABEL_GROUPS[gid].get("synthetic", False)
)


def label_group(label: str) -> str:
    """Return the group ID for a label, or the last group as fallback."""
    return LABEL_TO_GROUP.get(label, _FALLBACK_GROUP)


def group_color(group_id: str) -> dict[str, Any]:
    return LABEL_GROUPS.get(group_id, LABEL_GROUPS[_FALLBACK_GROUP])


# ── JSONL loading ──────────────────────────────────────────────────────────────

def _source_name(path: Path) -> str:
    """Return a clean dataset name from the filename."""
    return path.stem


def load_all_entries(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load all JSONL files, return list of dicts with 'source_file' added."""
    entries: list[dict[str, Any]] = []
    for path in sorted(data_dir.glob("*.jsonl")):
        source = _source_name(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entry["source_file"] = source
                    entries.append(entry)
                except json.JSONDecodeError:
                    pass
    return entries


def get_entry(entry_id: str, data_dir: Path = DATA_DIR) -> dict[str, Any] | None:
    """Find a single entry by id (scans all files)."""
    for path in sorted(data_dir.glob("*.jsonl")):
        source = _source_name(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("id") == entry_id:
                        entry["source_file"] = source
                        return entry
                except json.JSONDecodeError:
                    pass
    return None


def get_filter_options(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    models: set[str] = set()
    datasets: set[str] = set()
    trajectories: set[str] = set()
    alignments: set[str] = set()
    judges: set[str] = set()

    for e in entries:
        if e.get("model"):
            models.add(e["model"])
        if e.get("source_file"):
            datasets.add(e["source_file"])
        assessment = (e.get("metadata") or {}).get("assessment") or {}
        if assessment.get("trajectory"):
            trajectories.add(assessment["trajectory"])
        if assessment.get("alignment"):
            alignments.add(assessment["alignment"])
        if e.get("judge"):
            judges.add(e["judge"])

    return {
        "models": sorted(models),
        "datasets": sorted(datasets),
        "trajectories": sorted(trajectories),
        "alignments": sorted(alignments),
        "judges": sorted(judges),
    }


def entry_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """Produce a lightweight summary dict for the index page."""
    annotations = entry.get("annotations") or []
    metadata = entry.get("metadata") or {}
    assessment = metadata.get("assessment") or {}

    # Average safety score
    scores = [a.get("score", 0) for a in annotations if a.get("score") is not None]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0

    # User prompt preview
    user_prompt = ""
    for msg in (entry.get("messages") or []):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "")[:200]
            break

    return {
        "id": entry.get("id", ""),
        "model": entry.get("model", ""),
        "judge": entry.get("judge", ""),
        "source_file": entry.get("source_file", ""),
        "dataset_name": metadata.get("dataset_name", entry.get("source_file", "")),
        "trajectory": assessment.get("trajectory", ""),
        "alignment": assessment.get("alignment", ""),
        "turning_point": assessment.get("turning_point", -1),
        "avg_score": avg_score,
        "n_annotations": len(annotations),
        "has_reasoning": metadata.get("has_reasoning", bool(annotations)),
        "user_prompt": user_prompt,
        "labeled_at": metadata.get("labeled_at", ""),
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def compute_stats(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute dataset-wide statistics across all entries."""
    label_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    judge_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    trajectory_counts: dict[str, int] = {}
    alignment_counts: dict[str, int] = {}
    score_buckets: list[int] = [0, 0, 0]  # -1, 0, +1

    total_annotations = 0
    entries_with_annotations = 0

    for entry in entries:
        annotations = entry.get("annotations") or []
        metadata = entry.get("metadata") or {}
        assessment = metadata.get("assessment") or {}

        if annotations:
            entries_with_annotations += 1

        for ann in annotations:
            total_annotations += 1
            score = ann.get("score")
            if score is not None:
                s = int(float(score))
                if s == -1:
                    score_buckets[0] += 1
                elif s == 0:
                    score_buckets[1] += 1
                elif s == 1:
                    score_buckets[2] += 1

            for lbl in (ann.get("labels") or []):
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
                gid = label_group(lbl)
                group_counts[gid] = group_counts.get(gid, 0) + 1

        model = entry.get("model", "")
        if model:
            model_counts[model] = model_counts.get(model, 0) + 1
        judge = entry.get("judge", "")
        if judge:
            judge_counts[judge] = judge_counts.get(judge, 0) + 1
        ds = entry.get("source_file", "")
        if ds:
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
        traj = assessment.get("trajectory", "")
        if traj:
            trajectory_counts[traj] = trajectory_counts.get(traj, 0) + 1
        aln = assessment.get("alignment", "")
        if aln:
            alignment_counts[aln] = alignment_counts.get(aln, 0) + 1

    # Build per-group label breakdown (ordered by group definition)
    group_label_counts: dict[str, dict[str, int]] = {}
    for gid, gdata in LABEL_GROUPS.items():
        group_label_counts[gid] = {lbl: label_counts.get(lbl, 0) for lbl in gdata["labels"]}

    score_labels = ["-1", "0", "1"]

    return {
        "total_entries": len(entries),
        "total_annotations": total_annotations,
        "entries_with_annotations": entries_with_annotations,
        "label_counts": dict(sorted(label_counts.items(), key=lambda x: -x[1])),
        "group_counts": group_counts,
        "group_label_counts": group_label_counts,
        "model_counts": dict(sorted(model_counts.items(), key=lambda x: -x[1])),
        "judge_counts": dict(sorted(judge_counts.items(), key=lambda x: -x[1])),
        "dataset_counts": dict(sorted(dataset_counts.items(), key=lambda x: -x[1])),
        "trajectory_counts": dict(sorted(trajectory_counts.items(), key=lambda x: -x[1])),
        "alignment_counts": dict(sorted(alignment_counts.items(), key=lambda x: -x[1])),
        "score_buckets": score_buckets,
        "score_labels": score_labels,
    }


# ── Reasoning HTML renderer ────────────────────────────────────────────────────

def render_reasoning_html(reasoning_text: str, annotations: list[dict]) -> str:
    """Render reasoning text as HTML with colored annotation spans.

    Annotations are sorted and de-overlapped. Each span gets a colored
    background based on its primary label's group. Data attributes store all
    info needed for client-side color-mode toggling and behavior filtering.
    """
    if not reasoning_text:
        return ""

    # Filter annotations that target the reasoning message (we've already
    # selected the right message upstream, but double-check bounds)
    valid = []
    for ann in annotations:
        start = ann.get("char_start", 0)
        end = ann.get("char_end", 0)
        if 0 <= start < end <= len(reasoning_text):
            valid.append(ann)

    # Sort by start position; resolve overlaps greedily (first wins)
    valid.sort(key=lambda a: a["char_start"])
    non_overlapping: list[dict] = []
    cursor = 0
    for ann in valid:
        if ann["char_start"] >= cursor:
            non_overlapping.append(ann)
            cursor = ann["char_end"]

    def _escape_plain(text: str) -> str:
        """Escape plain text and convert newlines to <br>."""
        return escape(text).replace("\n", "<br>\n")

    # Build HTML segments
    parts: list[str] = []
    pos = 0
    for ann in non_overlapping:
        start, end = ann["char_start"], ann["char_end"]
        # Plain text before this annotation
        if pos < start:
            parts.append(_escape_plain(reasoning_text[pos:start]))

        labels = ann.get("labels") or ["neutralFiller"]
        primary = labels[0]
        gid = label_group(primary)
        gc = group_color(gid)
        score = ann.get("score", 0)

        # All group IDs referenced by labels in this span (comma-sep, no quotes)
        all_gids = ",".join(dict.fromkeys(label_group(l) for l in labels))
        # Labels as comma-sep string (label names are safe ASCII)
        labels_csv = ",".join(labels)

        style = (
            f"background:{gc['color']};"
            f"border:1px solid {gc['border']};"
            f"color:{gc['text']};"
            f"border-radius:3px;"
            f"padding:1px 3px;"
            f"cursor:default;"
        )
        span_text = _escape_plain(reasoning_text[start:end])
        parts.append(
            f'<span class="ann-span"'
            f' data-group="{gid}"'
            f' data-groups="{all_gids}"'
            f' data-labels="{labels_csv}"'
            f' data-score="{score}"'
            f' style="{style}">'
            f'{span_text}</span>'
        )
        pos = end

    # Remaining plain text
    if pos < len(reasoning_text):
        parts.append(_escape_plain(reasoning_text[pos:]))

    return "".join(parts)
