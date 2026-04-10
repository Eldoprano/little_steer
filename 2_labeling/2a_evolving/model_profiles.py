"""
Model profiles: split strategy registry for each model pattern.

All models are expected to have a role:'reasoning' message.
Records without one are skipped by data_loader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


MIN_PARAGRAPH_LEN = 15  # characters; shorter paragraphs are dropped


def split_by_blank_lines(text: str) -> list[str]:
    """Split text on blank lines, strip, and filter short/empty paragraphs."""
    paragraphs = re.split(r"\n\s*\n", text)
    result = []
    for p in paragraphs:
        p = p.strip()
        if len(p) >= MIN_PARAGRAPH_LEN and not all(c in " \t\n.,;:!?-–—" for c in p):
            result.append(p)
    return result


@dataclass
class ModelProfile:
    pattern: str               # substring matched against filename stem (case-insensitive)
    split_fn: Callable[[str], list[str]]  # receives reasoning text, returns paragraphs
    description: str           # human-readable note for inspect command


# Ordered: more-specific patterns first; empty string = fallback
PROFILES: list[ModelProfile] = [
    ModelProfile(
        pattern="deepseek",
        split_fn=split_by_blank_lines,
        description="Conversational reasoning in role:'reasoning', split by blank lines",
    ),
    ModelProfile(
        pattern="qwen",
        split_fn=split_by_blank_lines,
        description="Structured reasoning in role:'reasoning', split by blank lines",
    ),
    ModelProfile(
        pattern="phi",
        split_fn=split_by_blank_lines,
        description="Extended deliberative reasoning in role:'reasoning', split by blank lines",
    ),
    ModelProfile(
        pattern="ministral",
        split_fn=split_by_blank_lines,
        description="Reasoning expected in role:'reasoning'; records without it are skipped",
    ),
    ModelProfile(
        pattern="",
        split_fn=split_by_blank_lines,
        description="Generic fallback: reasoning in role:'reasoning', split by blank lines",
    ),
]


def get_profile(filename_stem: str) -> ModelProfile:
    """Return the first profile whose pattern is a substring of filename_stem (case-insensitive)."""
    stem_lower = filename_stem.lower()
    for profile in PROFILES:
        if profile.pattern and profile.pattern in stem_lower:
            return profile
    return PROFILES[-1]  # fallback
