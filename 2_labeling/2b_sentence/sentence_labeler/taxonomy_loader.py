"""taxonomy_loader.py — Load taxonomy.json and build derived structures.

Single source of truth: ../../taxonomy.json (relative to this file's location,
which places it at 2_labeling/taxonomy.json).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_TAXONOMY_PATH = Path(__file__).parents[2] / "taxonomy.json"


@lru_cache(maxsize=1)
def load_taxonomy() -> dict[str, Any]:
    """Load and cache the taxonomy JSON. Raises FileNotFoundError if missing."""
    with open(_TAXONOMY_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_valid_labels() -> frozenset[str]:
    """Return all valid label IDs, including synthetic ones (e.g. 'none')."""
    tax = load_taxonomy()
    return frozenset(
        label["id"]
        for group in tax["groups"]
        for label in group["labels"]
    )


def _real_groups(tax: dict) -> list:
    """Return non-synthetic groups only."""
    return [g for g in tax["groups"] if not g.get("synthetic", False)]


def get_valid_trajectories() -> set[str]:
    return {t["value"] for t in load_taxonomy()["trajectories"]}


def get_valid_alignments() -> set[str]:
    return {a["value"] for a in load_taxonomy()["alignments"]}


def get_fallback_label() -> str:
    """Return the last label of the last non-synthetic group (conventionally 'neutralFiller')."""
    tax = load_taxonomy()
    return _real_groups(tax)[-1]["labels"][-1]["id"]


def build_prompt_sections() -> dict[str, str]:
    """Build the taxonomy-derived sections for prompt.md injection.

    Returns a dict with keys:
      label_groups_section, label_ids_section,
      safety_categories_section, trajectories_section, alignments_section
    """
    tax = load_taxonomy()
    groups = _real_groups(tax)  # skip synthetic groups (e.g. none_group) in the prompt

    # ── Label groups (full, with descriptions) ────────────────────────────────
    group_lines: list[str] = []
    for group in groups:
        group_lines.append(f"### {group['name']}")
        group_lines.append("")
        for label in group["labels"]:
            group_lines.append(f"- `{label['id']}`: {label['description']}")
        group_lines.append("")
    # Strip trailing blank line
    while group_lines and not group_lines[-1]:
        group_lines.pop()

    # ── Label IDs only (refresher, no descriptions) ───────────────────────────
    id_lines: list[str] = []
    for group in groups:
        ids = ", ".join(f"{label['id']}" for label in group["labels"])
        id_lines.append(f"- **{group['name']}**: {ids}")

    # ── Safety categories ─────────────────────────────────────────────────────
    cat_lines: list[str] = []
    for cat in tax["safety_categories"]:
        code = cat["code"]
        code_str = f"+{code}" if code > 0 else str(code)
        cat_lines.append(f"- **{cat['name']}** ({code_str}): {cat['description']}")

    # ── Trajectories ──────────────────────────────────────────────────────────
    traj_lines: list[str] = []
    for t in tax["trajectories"]:
        traj_lines.append(f"   - `{t['value']}`: {t['description']}")

    # ── Alignments ────────────────────────────────────────────────────────────
    align_lines: list[str] = []
    for a in tax["alignments"]:
        align_lines.append(f"   - `{a['value']}`: {a['description']}")

    return {
        "label_groups_section": "\n".join(group_lines),
        "label_ids_section": "\n".join(id_lines),
        "safety_categories_section": "\n".join(cat_lines),
        "trajectories_section": "\n".join(traj_lines),
        "alignments_section": "\n".join(align_lines),
    }


def inject_taxonomy(template: str) -> str:
    """Replace taxonomy placeholders in a prompt template string.

    Uses str.replace rather than str.format so that literal braces
    elsewhere in the template (e.g. JSON examples) are never interpreted.
    """
    sections = build_prompt_sections()
    for key, value in sections.items():
        template = template.replace("{" + key + "}", value)
    return template
