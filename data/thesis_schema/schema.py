"""
little_steer.data.schema

Pydantic models for the little_steer dataset format.

Dataset format: one entry per conversation (ConversationEntry).
Each conversation has:
  - messages: HuggingFace-style list of {"role": ..., "content": ...}
    Supported roles: "system", "user", "assistant", "reasoning"
    The "reasoning" role stores chain-of-thought content independently of
    model-specific tokens (<think>, etc).
  - annotations: list of AnnotatedSpan, each pointing to a labeled span
    within a specific message.

Span positions use message_idx to identify the target message:
  - message_idx >= 0: char_start/char_end are relative to that message's content
  - message_idx == -1: absolute positions computed by concatenating all
    message contents in order (content only, no role prefixes)
"""

from __future__ import annotations

import hashlib
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class AnnotatedSpan(BaseModel):
    """A labeled annotation within a conversation message.

    Positions are relative to the message at message_idx.
    Use message_idx=-1 for spans that cross message boundaries (absolute position).
    """

    text: str
    """The annotated text segment."""

    message_idx: int
    """
    Index into the parent ConversationEntry.messages list.
    -1 means absolute position: char_start/char_end are offsets computed
    by concatenating all message contents (in order) without any separator.
    """

    char_start: int
    """Character offset within the message content (or absolute if message_idx=-1)."""

    char_end: int
    """End character offset (exclusive) within the message content."""

    labels: list[str]
    """Category labels (e.g., ["I_REPHRASE_PROMPT", "III_PLAN_IMMEDIATE_REASONING_STEP"])."""

    score: float = 0.0
    """Optional severity / confidence score from the judge."""

    meta: dict[str, Any] = Field(default_factory=dict)
    """Extensible metadata (e.g., judge sentence text, start_phrase, end_phrase)."""

    @field_validator("char_end")
    @classmethod
    def end_after_start(cls, v: int, info: Any) -> int:
        if "char_start" in info.data and v <= info.data["char_start"]:
            raise ValueError(
                f"char_end ({v}) must be greater than char_start ({info.data['char_start']})"
            )
        return v

    @field_validator("labels")
    @classmethod
    def labels_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("labels must not be empty")
        return v

    def validate_against_message(self, message_content: str) -> bool:
        """Verify that text matches the resolved message content slice."""
        extracted = message_content[self.char_start : self.char_end]
        return extracted == self.text

    def __repr__(self) -> str:
        labels_str = ", ".join(self.labels)
        return (
            f"AnnotatedSpan(msg={self.message_idx}, "
            f"[{self.char_start}:{self.char_end}], "
            f"labels=[{labels_str}], "
            f"text={self.text[:40]!r}{'...' if len(self.text) > 40 else ''})"
        )


class LabelRun(BaseModel):
    """One labeling pass over the current generated content."""

    judge_name: str
    judge_model_id: str = ""
    taxonomy_version: str = ""
    labeled_at: str = ""
    generation_hash: str = ""
    reasoning_truncated: bool = False
    assessment: dict[str, Any] = Field(default_factory=dict)
    sentence_annotations: list[dict[str, Any]] = Field(default_factory=list)
    spans: list[AnnotatedSpan] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str = ""
    status: str = "completed"
    error: str | None = None

    @property
    def key(self) -> str:
        return label_run_key(self.judge_name, self.taxonomy_version, self.generation_hash)


class SafetyRun(BaseModel):
    """One safety-scoring pass over the current generated content."""

    guard_name: str
    guard_model_id: str = ""
    scored_at: str = ""
    generation_hash: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    status: str = "completed"
    error: str | None = None

    @property
    def key(self) -> str:
        return safety_run_key(self.guard_name, self.generation_hash)


class ConversationEntry(BaseModel):
    """A single conversation with all its annotations.

    messages uses HuggingFace conversational format:
      [{"role": "user", "content": "..."}, {"role": "reasoning", "content": "..."}, ...]

    Supported roles:
      - "system": system prompt
      - "user": human turn
      - "assistant": model final response
      - "reasoning": model chain-of-thought (model-agnostic, no <think> tokens here)
    """

    id: str
    """Unique identifier for this conversation."""

    messages: list[dict[str, str]]
    """
    List of messages in HuggingFace conversational format.
    Each dict must have "role" and "content" keys.
    """

    annotations: list[AnnotatedSpan] = Field(default_factory=list)
    """All labeled annotation spans in this conversation."""

    model: str
    """Name/ID of the model that generated the response."""

    judge: str = ""
    """Name/ID of the model that performed the labeling."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """
    Extensible metadata: prompt_type, dataset_source, split, task_id,
    safety_decision_category, reasoning_usage_score, etc.
    """

    label_runs: list[LabelRun] = Field(default_factory=list)
    """All labeler outputs kept for this generated sample."""

    safety_runs: list[SafetyRun] = Field(default_factory=list)
    """All safety-scoring outputs kept for this generated sample."""

    @field_validator("messages")
    @classmethod
    def messages_valid(cls, v: list[dict]) -> list[dict]:
        valid_roles = {"system", "user", "assistant", "reasoning"}
        for i, msg in enumerate(v):
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message {i} must have 'role' and 'content' keys")
            if msg["role"] not in valid_roles:
                raise ValueError(
                    f"Message {i} has invalid role '{msg['role']}'. "
                    f"Valid roles: {valid_roles}"
                )
        return v

    @model_validator(mode="after")
    def annotations_in_bounds(self) -> ConversationEntry:
        """Verify all annotations reference valid message indices."""
        for ann in self.annotations:
            if ann.message_idx == -1:
                # Absolute position — just check it's non-negative
                continue
            if ann.message_idx < 0 or ann.message_idx >= len(self.messages):
                raise ValueError(
                    f"Annotation references message_idx={ann.message_idx}, "
                    f"but conversation only has {len(self.messages)} messages."
                )
        return self

    def get_message_content(self, message_idx: int) -> str:
        """Get the content string for a specific message."""
        if message_idx < 0 or message_idx >= len(self.messages):
            raise IndexError(f"message_idx {message_idx} out of range")
        return self.messages[message_idx]["content"]

    def get_absolute_content(self) -> str:
        """Concatenate all message contents (for message_idx=-1 span resolution)."""
        return "".join(msg["content"] for msg in self.messages)

    def get_message_absolute_offset(self, message_idx: int) -> int:
        """Get the char offset where message_idx starts in the absolute concatenation."""
        offset = 0
        for i, msg in enumerate(self.messages):
            if i == message_idx:
                return offset
            offset += len(msg["content"])
        raise IndexError(f"message_idx {message_idx} out of range")

    def get_reasoning_content(self) -> str | None:
        """Return the content of the first 'reasoning' message, or None."""
        for msg in self.messages:
            if msg["role"] == "reasoning":
                return msg["content"]
        return None

    def generation_hash(self) -> str:
        """Stable 16-char hash of the current reasoning content."""
        reasoning = self.get_reasoning_content() or ""
        return hashlib.md5(reasoning.encode("utf-8")).hexdigest()[:16]

    def active_label_run(self) -> LabelRun | None:
        """Return the currently-selected canonical label run, if any."""
        key = self.metadata.get("active_label_run")
        if key:
            for run in self.label_runs:
                if run.key == key:
                    return run
        if self.label_runs:
            return self.label_runs[-1]
        return None

    def set_active_label_run(self, run: LabelRun) -> None:
        """Mirror an active run onto legacy top-level fields."""
        self.metadata["active_label_run"] = run.key
        self.metadata["assessment"] = dict(run.assessment)
        self.metadata["labeled_at"] = run.labeled_at
        self.metadata["taxonomy_version"] = run.taxonomy_version
        if run.reasoning_truncated:
            self.metadata["reasoning_truncated"] = True
        else:
            self.metadata.pop("reasoning_truncated", None)
        self.annotations = list(run.spans)
        self.judge = run.judge_name

    def upsert_label_run(self, run: LabelRun, activate: bool = True) -> None:
        """Insert or replace a label run by canonical run key."""
        self.label_runs = [existing for existing in self.label_runs if existing.key != run.key]
        self.label_runs.append(run)
        if activate:
            self.set_active_label_run(run)

    def activate_judge(self, judge_name: str) -> bool:
        """Set entry.annotations to the most recent run from a specific judge.

        Must be called before passing this entry to the representation extractor
        (Step 3), which reads entry.annotations.  With multiple concurrent judges
        upsert_label_run activates whichever judge wrote last — non-deterministic.
        Call this explicitly to pin a preferred judge (default: gpt-mini).

        Matching is by prefix so 'gpt-mini' matches 'gpt-mini_pass2' etc.
        Returns True if a matching run was found and activated, False otherwise.
        """
        matching = [r for r in self.label_runs if r.judge_name == judge_name
                    or r.judge_name.startswith(judge_name + "_")]
        if not matching:
            return False
        best = max(matching, key=lambda r: r.labeled_at)
        self.set_active_label_run(best)
        return True

    def upsert_safety_run(self, run: SafetyRun) -> None:
        """Insert or replace a safety run by canonical run key."""
        self.safety_runs = [existing for existing in self.safety_runs if existing.key != run.key]
        self.safety_runs.append(run)

    def reset_after_regeneration(
        self,
        *,
        messages: list[dict[str, str]],
        generation_metadata: dict[str, Any],
    ) -> None:
        """Replace generated content and clear all derived downstream state."""
        self.messages = messages
        self.annotations = []
        self.judge = ""
        self.label_runs = []
        self.safety_runs = []
        self.metadata = dict(generation_metadata)
        self.metadata.pop("active_label_run", None)
        self.metadata.pop("assessment", None)
        self.metadata.pop("labeled_at", None)
        self.metadata.pop("taxonomy_version", None)
        self.metadata.pop("reasoning_truncated", None)

    def get_annotations_for_message(self, message_idx: int) -> list[AnnotatedSpan]:
        """Return annotations targeting a specific message."""
        return [a for a in self.annotations if a.message_idx == message_idx]

    def summary(self) -> str:
        roles = [m["role"] for m in self.messages]
        label_counts: dict[str, int] = {}
        for ann in self.annotations:
            for lbl in ann.labels:
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
        top_labels = sorted(label_counts, key=lambda k: -label_counts[k])[:5]
        return (
            f"ConversationEntry(id={self.id!r}, model={self.model!r}, "
            f"turns={roles}, annotations={len(self.annotations)}, "
            f"top_labels={top_labels})"
        )

    def __repr__(self) -> str:
        return self.summary()


def label_run_key(judge_name: str, taxonomy_version: str, generation_hash: str) -> str:
    """Canonical identifier for one label run."""
    return f"{judge_name}::{taxonomy_version}::{generation_hash}"


def safety_run_key(guard_name: str, generation_hash: str) -> str:
    """Canonical identifier for one safety run."""
    return f"{guard_name}::{generation_hash}"
