import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data"))

from thesis_schema import ConversationEntry  # noqa: E402


def _load_fix_quality_module():
    path = ROOT / "1_generating" / "fix_quality.py"
    spec = importlib.util.spec_from_file_location("fix_quality", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_entry(*, assistant: str = "Answer", finish_reason: str = "eos") -> ConversationEntry:
    return ConversationEntry(
        id="entry-1",
        messages=[
            {"role": "user", "content": "Prompt"},
            {"role": "reasoning", "content": "Reasoning"},
            {"role": "assistant", "content": assistant},
        ],
        model="model-a",
        judge="",
        metadata={
            "prompt_id": "prompt-1",
            "generation": {"finish_reason": finish_reason},
            "generation_hash": "abc123",
        },
    )


def test_quality_payload_reads_nested_generation_finish_reason():
    mod = _load_fix_quality_module()
    entry = _make_entry(finish_reason="max_length")

    quality = mod._quality_payload(entry)

    assert "max_length" in quality["issues"]
    assert quality["approved"] is False


def test_fix_artifacts_updates_quality_metadata(tmp_path: Path):
    mod = _load_fix_quality_module()
    path = tmp_path / "dataset.jsonl"
    entry = _make_entry(assistant="[/THINK] Cleaned answer")
    path.write_text(entry.model_dump_json() + "\n", encoding="utf-8")

    fixed = mod.fix_artifacts_file(path, dry_run=False)
    updated = ConversationEntry.model_validate_json(path.read_text(encoding="utf-8").strip())

    assert fixed == 1
    assert updated.messages[-1]["content"] == "Cleaned answer"
    assert updated.metadata["quality"]["approved"] is True
    assert "think_artifact" not in updated.metadata["quality"]["issues"]
    assert updated.metadata["approved"] is True
