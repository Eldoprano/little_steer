"""data_loader.py — Load and process labeled JSONL entries for the viewer."""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data" / "labeled"

# ── Label group metadata ───────────────────────────────────────────────────────

LABEL_GROUPS: dict[str, dict[str, str]] = {
    "I": {
        "name": "Prompt Processing",
        "color": "#dbeafe",
        "border": "#93c5fd",
        "text": "#1e40af",
        "labels": ["I_REPHRASE_PROMPT", "I_SPECULATE_USER_MOTIVE", "I_FLAG_EVALUATION_AWARENESS"],
    },
    "II": {
        "name": "Safety Assessment",
        "color": "#fee2e2",
        "border": "#fca5a5",
        "text": "#991b1b",
        "labels": [
            "II_FLAG_PROMPT_AS_HARMFUL", "II_STATE_SAFETY_CONCERN", "II_STATE_LEGAL_CONCERN",
            "II_STATE_ETHICAL_MORAL_CONCERN", "II_CHECK_POLICY_SCOPE", "II_CAUTIOUS_FRAMING",
        ],
    },
    "III": {
        "name": "Deliberation",
        "color": "#fef9c3",
        "border": "#fde047",
        "text": "#713f12",
        "labels": ["III_WEIGH_DECISION", "III_REFRAME_TOWARD_SAFETY", "III_REFRAME_TOWARD_COMPLIANCE"],
    },
    "IV": {
        "name": "Intent Declaration",
        "color": "#ede9fe",
        "border": "#c4b5fd",
        "text": "#4c1d95",
        "labels": ["IV_INTEND_REFUSAL", "IV_INTEND_HARMFUL_COMPLIANCE"],
    },
    "V": {
        "name": "Knowledge & Content",
        "color": "#ffedd5",
        "border": "#fdba74",
        "text": "#7c2d12",
        "labels": ["V_STATE_FACT_OR_KNOWLEDGE", "V_DETAIL_HARMFUL_METHOD"],
    },
    "VI": {
        "name": "Meta-Cognition",
        "color": "#dcfce7",
        "border": "#86efac",
        "text": "#14532d",
        "labels": [
            "VI_EXPRESS_UNCERTAINTY", "VI_SELF_CORRECT",
            "VI_PLAN_REASONING_STEP", "VI_SUMMARIZE_REASONING",
        ],
    },
    "VII": {
        "name": "Filler",
        "color": "#f3f4f6",
        "border": "#d1d5db",
        "text": "#374151",
        "labels": ["VII_NEUTRAL_FILLER"],
    },
}

# Build label → group lookup
LABEL_TO_GROUP: dict[str, str] = {}
for gid, gdata in LABEL_GROUPS.items():
    for lbl in gdata["labels"]:
        LABEL_TO_GROUP[lbl] = gid


def label_group(label: str) -> str:
    """Return the group ID for a label (e.g. 'II'), or 'VII' as fallback."""
    return LABEL_TO_GROUP.get(label, "VII")


def group_color(group_id: str) -> dict[str, str]:
    return LABEL_GROUPS.get(group_id, LABEL_GROUPS["VII"])


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

        labels = ann.get("labels") or ["VII_NEUTRAL_FILLER"]
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
