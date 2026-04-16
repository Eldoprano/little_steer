"""schema.py — Pydantic models for the sentence labeler.

LLM output schema (what we parse from the judge response) plus config models.
Valid labels, trajectories, and alignments are loaded from taxonomy.json via
taxonomy_loader — do not hardcode them here.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from .taxonomy_loader import (
    get_fallback_label,
    get_valid_alignments,
    get_valid_labels,
    get_valid_trajectories,
)

# ── Valid label set (derived from taxonomy.json) ───────────────────────────────

VALID_LABELS: frozenset[str] = get_valid_labels()
_VALID_LABELS_LIST: list[str] = sorted(VALID_LABELS)
_FALLBACK_LABEL: str = get_fallback_label()


def coerce_label(raw: str) -> str | None:
    """Try to map a raw label string to a valid label.

    Tries in order:
    1. Exact match (case-sensitive)
    2. Case-insensitive exact match
    3. Fuzzy match via difflib (cutoff 0.7)

    Returns the matched label, or None if no match found.
    """
    if raw in VALID_LABELS:
        return raw
    lower = raw.lower()
    for lbl in VALID_LABELS:
        if lbl.lower() == lower:
            return lbl
    matches = difflib.get_close_matches(raw, _VALID_LABELS_LIST, n=1, cutoff=0.7)
    if matches:
        return matches[0]
    return None


# ── LLM output schema ─────────────────────────────────────────────────────────

# Trajectory and alignment values are validated against the taxonomy at runtime.
# The Literal types below are kept for IDE autocompletion only; the actual
# validation uses the sets loaded from taxonomy.json.
TrajectoryType = Literal[
    "concern_then_refuse",
    "concern_then_comply",
    "comply_no_deliberation",
    "refuse_no_deliberation",
    "deliberation_then_refuse",
    "deliberation_then_comply",
]

AlignmentType = Literal["aligned", "misaligned", "ambiguous"]

_VALID_TRAJECTORIES: set[str] = get_valid_trajectories()
_VALID_ALIGNMENTS: set[str] = get_valid_alignments()


class SentenceAnnotation(BaseModel):
    """A single sentence with behavioral labels and a safety score.

    Labels are coerced from whatever the model returns: exact → case-insensitive
    → fuzzy (cutoff 0.7). Labels that cannot be resolved are dropped with a
    warning. If all labels are invalid, falls back to VII_NEUTRAL_FILLER.
    """

    text: str
    labels: list[str]
    safety_score: int

    @field_validator("labels", mode="before")
    @classmethod
    def coerce_labels(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("labels must be a list")
        if len(v) == 0:
            return ["none"]  # model explicitly found no matching behavior
        coerced: list[str] = []
        for raw in v:
            resolved = coerce_label(str(raw))
            if resolved is not None:
                coerced.append(resolved)
            # silently drop unresolvable labels (caller can log warnings)
        if not coerced:
            coerced = [_FALLBACK_LABEL]
        if len(coerced) > 3:
            coerced = coerced[:3]
        return coerced

    @field_validator("safety_score", mode="before")
    @classmethod
    def clamp_score(cls, v: Any) -> int:
        try:
            score = int(v)
        except (TypeError, ValueError):
            return 0
        return max(-1, min(1, score))

    @model_validator(mode="after")
    def text_not_empty(self) -> SentenceAnnotation:
        if not self.text.strip():
            raise ValueError("sentence text must not be empty")
        return self


class Assessment(BaseModel):
    """Overall assessment of the reasoning block."""

    trajectory: str
    turning_point: int = -1
    alignment: str

    @field_validator("trajectory", mode="before")
    @classmethod
    def coerce_trajectory(cls, v: Any) -> str:
        s = str(v).lower().strip()
        if s in _VALID_TRAJECTORIES:
            return s
        # fuzzy fallback
        matches = difflib.get_close_matches(s, list(_VALID_TRAJECTORIES), n=1, cutoff=0.6)
        if matches:
            return matches[0]
        return "comply_no_deliberation"

    @field_validator("alignment", mode="before")
    @classmethod
    def coerce_alignment(cls, v: Any) -> str:
        s = str(v).lower().strip()
        if s in _VALID_ALIGNMENTS:
            return s
        matches = difflib.get_close_matches(s, list(_VALID_ALIGNMENTS), n=1, cutoff=0.6)
        if matches:
            return matches[0]
        return "ambiguous"

    @field_validator("turning_point", mode="before")
    @classmethod
    def coerce_turning_point(cls, v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1


class LabelingOutput(BaseModel):
    """Complete output from the LLM judge for one entry."""

    sentences: list[SentenceAnnotation]
    assessment: Assessment
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sentences")
    @classmethod
    def sentences_not_empty(cls, v: list[SentenceAnnotation]) -> list[SentenceAnnotation]:
        if not v:
            raise ValueError("sentences list must not be empty")
        return v


# ── Config models ─────────────────────────────────────────────────────────────

class JudgeConfig(BaseModel):
    name: str
    model_id: str
    backend: str = "openai"
    base_url: str | None = None
    api_key_source: str = "openai-api-key"
    temperature: float = 0.2
    max_tokens: int | None = 8192
    max_completion_tokens: int | None = None
    timeout: int = 120
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    service_tier: Literal["auto", "default", "flex"] | None = None
    # When True, run.py will pause before this judge and ask the user to
    # load the model in LMStudio before continuing.
    lmstudio_prompt: bool = False
    rpm: int | None = None  # requests per minute limit for this judge; null = no limit
    rpd: int | None = None  # requests per day limit for this judge; null = no limit


class PipelineConfig(BaseModel):
    max_retries: int = 3
    retry_delay: float = 2.0
    checkpoint_every: int = 10
    max_workers: int = 4
    response_sentences: int = 5
    skip_no_reasoning: bool = True
    skip_no_response: bool = False
    overwrite_existing: bool = False
    max_reasoning_chars: int | None = None  # crop reasoning before this many chars (at sentence boundary)
    token_budget: int | None = None  # daily token limit (tokens); null = no limit
    budget_state_file: str = "token_budget_state.json"  # path relative to config file
    request_state_file: str = "request_budget_state.json"  # path relative to config file
    max_entries: int | None = None  # stop after labeling this many entries (across all files); null = no limit
    max_entries_per_file: int | None = None  # stop after labeling this many entries per file; null = no limit
    shuffle_files: bool = True  # randomise file order each run for breadth-first coverage
    work_order_file: str | None = None  # shared work order JSON; when set, overrides shuffle/max_entries_per_file


class MapperConfig(BaseModel):
    fuzzy_threshold: float = 0.85
    max_lookahead_multiplier: int = 5
    max_lookahead_min: int = 500


class OutputConfig(BaseModel):
    dir: str = "../../data/2b_labeled"
    suffix: str = ""
    in_place: bool = False
    taxonomy_version: str = ""  # when set, output goes to {dir}/{taxonomy_version}/


class LabelerConfig(BaseModel):
    """Top-level labeler configuration.

    Supports both the old singular ``judge:`` key and the new ``judges:`` list.
    A model_validator normalises the old form into ``judges`` so the rest of the
    code only needs to deal with the list.
    """

    # New preferred form — one or more judges.
    judges: list[JudgeConfig] = Field(default_factory=list)
    # Old singular form kept for backward-compatibility; always normalised away.
    judge: JudgeConfig | None = Field(default=None, exclude=True)

    secrets_file: str = "../../.secrets.json"
    prompt_file: str = "prompt.md"
    system_prompt_file: str | None = "system_prompt.md"
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    mapper: MapperConfig = Field(default_factory=MapperConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _normalise_judges(self) -> "LabelerConfig":
        """Merge singular judge into the list, ensuring judges is never empty."""
        if self.judge is not None and self.judge not in self.judges:
            self.judges = [self.judge] + list(self.judges)
        if not self.judges:
            raise ValueError(
                "Config must define at least one judge via 'judge:' or 'judges:' key."
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "LabelerConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def resolve_path(self, config_path: Path, relative: str) -> Path:
        """Resolve a path that is relative to the config file's directory."""
        return (config_path.parent / relative).resolve()
