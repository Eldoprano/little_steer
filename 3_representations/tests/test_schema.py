"""
Tests for little_steer.data.schema — AnnotatedSpan and ConversationEntry.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
from thesis_schema import AnnotatedSpan, ConversationEntry, LabelRun, SafetyRun


# ── AnnotatedSpan ────────────────────────────────────────────────────────────

def test_annotated_span_basic():
    span = AnnotatedSpan(
        text="Hello world",
        message_idx=0,
        char_start=0,
        char_end=11,
        labels=["TEST_LABEL"],
    )
    assert span.text == "Hello world"
    assert span.char_start == 0
    assert span.char_end == 11
    assert span.labels == ["TEST_LABEL"]


def test_annotated_span_end_before_start_raises():
    with pytest.raises(Exception):
        AnnotatedSpan(
            text="oops",
            message_idx=0,
            char_start=10,
            char_end=5,  # end < start
            labels=["TEST"],
        )


def test_annotated_span_empty_labels_raises():
    with pytest.raises(Exception):
        AnnotatedSpan(
            text="hi",
            message_idx=0,
            char_start=0,
            char_end=2,
            labels=[],  # empty
        )


def test_annotated_span_absolute_position():
    """message_idx=-1 should be accepted without a bound check."""
    span = AnnotatedSpan(
        text="across messages",
        message_idx=-1,
        char_start=95,
        char_end=110,
        labels=["CROSS_MSG"],
    )
    assert span.message_idx == -1


def test_annotated_span_validates_text():
    span = AnnotatedSpan(
        text="Hello",
        message_idx=0,
        char_start=0,
        char_end=5,
        labels=["L"],
    )
    assert span.validate_against_message("Hello world") is True
    assert span.validate_against_message("Xello world") is False


# ── ConversationEntry ────────────────────────────────────────────────────────

SAMPLE_MESSAGES = [
    {"role": "user", "content": "Can you help with X?"},
    {"role": "reasoning", "content": "I need to think about this. The user wants X."},
    {"role": "assistant", "content": "Sure, here is how to do X."},
]


def make_entry(**kwargs) -> ConversationEntry:
    defaults = dict(
        id="test_001",
        messages=SAMPLE_MESSAGES,
        annotations=[],
        model="test-model",
        judge="test-judge",
    )
    defaults.update(kwargs)
    return ConversationEntry(**defaults)


def test_conversation_entry_basic():
    entry = make_entry()
    assert entry.id == "test_001"
    assert len(entry.messages) == 3
    assert entry.get_reasoning_content() == "I need to think about this. The user wants X."


def test_conversation_entry_no_reasoning():
    entry = make_entry(messages=[
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ])
    assert entry.get_reasoning_content() is None


def test_conversation_entry_invalid_role_raises():
    with pytest.raises(Exception):
        make_entry(messages=[{"role": "robot", "content": "beep"}])


def test_conversation_entry_missing_content_raises():
    with pytest.raises(Exception):
        make_entry(messages=[{"role": "user"}])  # no content


def test_conversation_entry_annotation_in_bounds():
    """Valid annotation pointing to message 1."""
    span = AnnotatedSpan(
        text="I need to think",
        message_idx=1,
        char_start=0,
        char_end=15,
        labels=["III_PLAN_IMMEDIATE_REASONING_STEP"],
    )
    entry = make_entry(annotations=[span])
    assert len(entry.annotations) == 1


def test_conversation_entry_annotation_out_of_bounds_raises():
    """Annotation pointing to non-existent message index should fail."""
    span = AnnotatedSpan(
        text="x",
        message_idx=99,  # way out of bounds
        char_start=0,
        char_end=1,
        labels=["L"],
    )
    with pytest.raises(Exception):
        make_entry(annotations=[span])


def test_conversation_entry_absolute_annotation_allowed():
    """message_idx=-1 should be accepted (absolute position)."""
    span = AnnotatedSpan(
        text="can help with",
        message_idx=-1,
        char_start=4,
        char_end=17,
        labels=["L"],
    )
    entry = make_entry(annotations=[span])
    assert entry.annotations[0].message_idx == -1


def test_get_message_content():
    entry = make_entry()
    assert entry.get_message_content(0) == "Can you help with X?"
    assert "think about this" in entry.get_message_content(1)
    with pytest.raises(IndexError):
        entry.get_message_content(99)


def test_get_absolute_content():
    entry = make_entry()
    absolute = entry.get_absolute_content()
    assert "Can you help with X?" in absolute
    assert "I need to think" in absolute
    assert "Sure, here is how" in absolute


def test_get_message_absolute_offset():
    entry = make_entry()
    # Message 0 starts at 0
    assert entry.get_message_absolute_offset(0) == 0
    # Message 1 starts after message 0
    msg0_len = len(SAMPLE_MESSAGES[0]["content"])
    assert entry.get_message_absolute_offset(1) == msg0_len


def test_get_annotations_for_message():
    span0 = AnnotatedSpan(text="hi", message_idx=0, char_start=0, char_end=2, labels=["L1"])
    span1 = AnnotatedSpan(text="x", message_idx=1, char_start=0, char_end=1, labels=["L2"])
    entry = make_entry(annotations=[span0, span1])
    assert len(entry.get_annotations_for_message(0)) == 1
    assert len(entry.get_annotations_for_message(1)) == 1
    assert len(entry.get_annotations_for_message(2)) == 0


def test_json_roundtrip():
    span = AnnotatedSpan(
        text="I need to think",
        message_idx=1,
        char_start=0,
        char_end=15,
        labels=["III_PLAN"],
        score=2.5,
    )
    entry = make_entry(annotations=[span])
    json_str = entry.model_dump_json()
    restored = ConversationEntry.model_validate_json(json_str)
    assert restored.id == entry.id
    assert len(restored.annotations) == 1
    assert restored.annotations[0].text == "I need to think"
    assert restored.annotations[0].score == 2.5


def test_generation_hash_changes_with_reasoning():
    entry_a = make_entry()
    entry_b = make_entry(messages=[
        {"role": "user", "content": "Can you help with X?"},
        {"role": "reasoning", "content": "Different reasoning."},
        {"role": "assistant", "content": "Sure, here is how to do X."},
    ])
    assert entry_a.generation_hash() != entry_b.generation_hash()


def test_upsert_label_run_updates_active_fields():
    entry = make_entry()
    run = LabelRun(
        judge_name="judge-v6",
        judge_model_id="judge-model",
        taxonomy_version="v6",
        labeled_at="2026-04-21T00:00:00Z",
        generation_hash=entry.generation_hash(),
        assessment={"trajectory": "comply_no_deliberation", "turning_point": -1, "alignment": "ambiguous"},
        sentence_annotations=[],
        spans=[
            AnnotatedSpan(
                text="I need to think",
                message_idx=1,
                char_start=0,
                char_end=15,
                labels=["III_PLAN"],
            )
        ],
    )
    entry.upsert_label_run(run, activate=True)
    assert entry.judge == "judge-v6"
    assert entry.metadata["active_label_run"] == run.key
    assert entry.annotations == run.spans


def test_reset_after_regeneration_clears_derived_state():
    entry = make_entry(
        annotations=[
            AnnotatedSpan(text="I need to think", message_idx=1, char_start=0, char_end=15, labels=["III_PLAN"])
        ],
        metadata={"active_label_run": "x", "assessment": {"trajectory": "a"}},
        label_runs=[
            LabelRun(judge_name="judge-v5", taxonomy_version="v5", generation_hash="abc")
        ],
        safety_runs=[
            SafetyRun(guard_name="wildguard", generation_hash="abc")
        ],
        judge="judge-v5",
    )
    entry.reset_after_regeneration(
        messages=[
            {"role": "user", "content": "Hi"},
            {"role": "reasoning", "content": "Fresh reasoning"},
            {"role": "assistant", "content": "Fresh answer"},
        ],
        generation_metadata={"prompt_id": "p1", "generation_hash": "newhash"},
    )
    assert entry.annotations == []
    assert entry.judge == ""
    assert entry.label_runs == []
    assert entry.safety_runs == []
    assert entry.metadata["generation_hash"] == "newhash"
    assert "active_label_run" not in entry.metadata
