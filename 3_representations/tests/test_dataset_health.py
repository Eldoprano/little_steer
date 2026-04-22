import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))

from thesis_schema import AnnotatedSpan, ConversationEntry, LabelRun
from dataset_health import check_dataset


def _write_entries(tmp_path: Path, entries: list[ConversationEntry]) -> Path:
    path = tmp_path / "dataset.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.model_dump_json() + "\n")
    return path


def _make_entry(entry_id: str = "entry-1") -> ConversationEntry:
    entry = ConversationEntry(
        id=entry_id,
        messages=[
            {"role": "user", "content": "Prompt"},
            {"role": "reasoning", "content": "Reasoning sentence."},
            {"role": "assistant", "content": "Answer"},
        ],
        model="model-a",
        metadata={"prompt_id": "prompt-1", "generation_hash": "placeholder", "quality": {"approved": True}, "approved": True},
    )
    entry.metadata["generation_hash"] = entry.generation_hash()
    return entry


def test_dataset_health_accepts_clean_v6_entry(tmp_path: Path):
    entry = _make_entry()
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
                text="Reasoning sentence.",
                message_idx=1,
                char_start=0,
                char_end=len("Reasoning sentence."),
                labels=["III_PLAN"],
            )
        ],
    )
    entry.upsert_label_run(run, activate=True)
    path = _write_entries(tmp_path, [entry])

    counts, _ = check_dataset(path, allowed_taxonomy="v6")
    assert counts == {}


def test_dataset_health_flags_stale_and_noncanonical_runs(tmp_path: Path):
    entry = _make_entry()
    run = LabelRun(
        judge_name="judge-v5",
        taxonomy_version="v5",
        generation_hash="stalehash",
        spans=[
            AnnotatedSpan(
                text="wrong",
                message_idx=1,
                char_start=0,
                char_end=5,
                labels=["III_PLAN"],
            )
        ],
    )
    entry.upsert_label_run(run, activate=True)
    path = _write_entries(tmp_path, [entry, _make_entry(entry_id="entry-1")])

    counts, _ = check_dataset(path, allowed_taxonomy="v6")
    assert counts["duplicate_id"] == 1
    assert counts["stale_label_run"] == 1
    assert counts["noncanonical_taxonomy"] == 1
    assert counts["span_text_mismatch"] == 1
