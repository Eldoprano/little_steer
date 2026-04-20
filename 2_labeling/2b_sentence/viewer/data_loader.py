"""data_loader.py — Load and process labeled JSONL entries for the viewer."""

from __future__ import annotations

import json
import os
import re
from html import escape
from pathlib import Path
from typing import Any

_TAXONOMY_PATH = Path(__file__).parents[2] / "taxonomy.json"

from sentence_labeler.taxonomy_loader import get_taxonomy_version

_DEFAULT_DATA_DIR = Path(__file__).parents[3] / "data" / "2b_labeled" / get_taxonomy_version()
DATA_DIR: Path = Path(os.environ["VIEWER_DATA_DIR"]) if "VIEWER_DATA_DIR" in os.environ else _DEFAULT_DATA_DIR

# ── Label group metadata (loaded from taxonomy.json) ─────────────────────────


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

# Flat dict of all label ids → {name, group_id, group_name, colors…}
def _build_all_labels() -> dict[str, dict[str, Any]]:
    with open(_TAXONOMY_PATH, encoding="utf-8") as f:
        tax = json.load(f)
    result: dict[str, dict[str, Any]] = {}
    for group in tax["groups"]:
        colors = group["colors"]
        dark = colors.get("dark", colors["light"])
        for label in group["labels"]:
            result[label["id"]] = {
                "name": label.get("display") or label.get("name") or label["id"],
                "group_id": group["id"],
                "group_name": group["name"],
                "color": colors["light"]["bg"],
                "border": colors["light"]["border"],
                "text": colors["light"]["text"],
                "dark_color": dark["bg"],
                "dark_border": dark["border"],
                "dark_text": dark["text"],
            }
    return result

ALL_LABELS: dict[str, dict[str, Any]] = _build_all_labels()

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


def _extract_dataset_name(path: Path, entry: dict[str, Any]) -> str:
    """Return human-readable dataset name: prefer metadata.dataset_name, fallback to stem."""
    return (entry.get("metadata") or {}).get("dataset_name") or path.stem


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
                    entry.setdefault("_dataset_name", _extract_dataset_name(path, entry))
                    entries.append(entry)
                except json.JSONDecodeError:
                    pass
    return entries


def get_all_versions(entry_id: str, data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    """Return all labeled versions of an entry (same id, across all source files)."""
    versions: list[dict[str, Any]] = []
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
                        entry["_dataset_name"] = _extract_dataset_name(path, entry)
                        versions.append(entry)
                        break
                except json.JSONDecodeError:
                    pass
    return versions


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
                        entry["_dataset_name"] = _extract_dataset_name(path, entry)
                        return entry
                except json.JSONDecodeError:
                    pass
    return None


def get_filter_options(entries: list[dict[str, Any]]) -> dict[str, list[str]]:
    models: set[str] = set()
    datasets: set[str] = set()
    dataset_names: dict[str, str] = {}  # source_file → dataset_name
    trajectories: set[str] = set()
    alignments: set[str] = set()
    judges: set[str] = set()

    for e in entries:
        if e.get("model"):
            models.add(e["model"])
        if e.get("source_file"):
            sf = e["source_file"]
            datasets.add(sf)
            dn = (e.get("_dataset_name") or
                  (e.get("metadata") or {}).get("dataset_name") or sf)
            dataset_names[sf] = dn
        assessment = (e.get("metadata") or {}).get("assessment") or {}
        if assessment.get("trajectory"):
            trajectories.add(assessment["trajectory"])
        if assessment.get("alignment"):
            alignments.add(assessment["alignment"])
        if e.get("judge"):
            judges.add(e["judge"])

    # Collect all behaviour group IDs that actually appear in any entry
    behaviour_group_ids: list[str] = list(dict.fromkeys(
        gid
        for gid in LABEL_GROUPS
        if any(
            gid in (
                label_group(lbl)
                for ann in (e.get("annotations") or [])
                for lbl in (ann.get("labels") or [])
            )
            for e in entries
        )
    ))

    return {
        "models": sorted(models),
        "datasets": sorted(datasets),
        "dataset_names": dataset_names,
        "trajectories": sorted(trajectories),
        "alignments": sorted(alignments),
        "judges": sorted(judges),
        "behaviours": behaviour_group_ids,
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

    # Collect individual labels and behaviour groups present in this entry
    label_ids: list[str] = list(dict.fromkeys(
        lbl
        for ann in annotations
        for lbl in (ann.get("labels") or [])
    ))
    behaviour_groups: list[str] = list(dict.fromkeys(label_group(l) for l in label_ids))

    dataset_name = entry.get("_dataset_name") or metadata.get("dataset_name") or entry.get("source_file", "")

    return {
        "id": entry.get("id", ""),
        "model": entry.get("model", ""),
        "judge": entry.get("judge", ""),
        "source_file": entry.get("source_file", ""),
        "dataset_name": dataset_name,
        "trajectory": assessment.get("trajectory", ""),
        "alignment": assessment.get("alignment", ""),
        "turning_point": assessment.get("turning_point", -1),
        "avg_score": avg_score,
        "n_annotations": len(annotations),
        "has_reasoning": metadata.get("has_reasoning", bool(annotations)),
        "user_prompt": user_prompt,
        "labeled_at": metadata.get("labeled_at", ""),
        "behaviours": behaviour_groups,
        "label_ids": label_ids,
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
    model_dataset_counts: dict[str, dict[str, int]] = {}  # model → dataset_name → count

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
        ds = (entry.get("_dataset_name") or
              (entry.get("metadata") or {}).get("dataset_name") or
              entry.get("source_file", ""))
        if ds:
            dataset_counts[ds] = dataset_counts.get(ds, 0) + 1

        model_key = model or "unknown"
        ds_key = ds or "unknown"
        if model_key not in model_dataset_counts:
            model_dataset_counts[model_key] = {}
        model_dataset_counts[model_key][ds_key] = (
            model_dataset_counts[model_key].get(ds_key, 0) + 1
        )
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

    # Collect sorted unique models and datasets for heatmap axes
    all_heatmap_models = sorted(model_dataset_counts.keys())
    all_heatmap_datasets = sorted({ds for m in model_dataset_counts.values() for ds in m})

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
        "model_dataset_counts": model_dataset_counts,
        "heatmap_models": all_heatmap_models,
        "heatmap_datasets": all_heatmap_datasets,
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
