import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "2_labeling" / "2b_sentence"))


def _load_module():
    path = ROOT / "2_labeling" / "2b_sentence" / "viewer" / "data_loader.py"
    spec = importlib.util.spec_from_file_location("viewer_data_loader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_all_entries_expands_one_record_per_label_run_and_hides_unlabeled(tmp_path: Path):
    mod = _load_module()
    path = tmp_path / "dataset.jsonl"
    entry = {
        "id": "entry-1",
        "messages": [
            {"role": "user", "content": "Prompt"},
            {"role": "reasoning", "content": "Reasoning"},
            {"role": "assistant", "content": "Answer"},
        ],
        "annotations": [],
        "model": "model-a",
        "judge": "",
        "metadata": {
            "prompt_id": "prompt-1",
            "dataset_name": "dataset-a",
            "generation_hash": "hash-1",
        },
        "label_runs": [
            {
                "judge_name": "gpt-5.4-mini",
                "judge_model_id": "gpt-5.4-mini",
                "taxonomy_version": "v6",
                "labeled_at": "2026-04-21T00:00:00Z",
                "generation_hash": "hash-1",
                "reasoning_truncated": False,
                "assessment": {"trajectory": "comply_no_deliberation", "alignment": "ambiguous", "turning_point": -1},
                "sentence_annotations": [{"text": "Reasoning", "labels": ["III_PLAN"], "safety_score": 0}],
                "spans": [{"text": "Reasoning", "message_idx": 1, "char_start": 0, "char_end": 9, "labels": ["III_PLAN"], "score": 0, "meta": {}}],
                "usage": {},
                "finish_reason": "stop",
                "status": "completed",
                "error": None,
            },
            {
                "judge_name": "gpt-5.4-2026-03-05",
                "judge_model_id": "gpt-5.4-2026-03-05",
                "taxonomy_version": "v6",
                "labeled_at": "2026-04-21T00:00:01Z",
                "generation_hash": "hash-1",
                "reasoning_truncated": False,
                "assessment": {"trajectory": "concern_then_refuse", "alignment": "aligned", "turning_point": 0},
                "sentence_annotations": [],
                "spans": [],
                "usage": {},
                "finish_reason": "stop",
                "status": "completed",
                "error": None,
            },
        ],
        "safety_runs": [
            {
                "guard_name": "wildguard",
                "guard_model_id": "wildguard",
                "scored_at": "2026-04-21T01:00:00Z",
                "generation_hash": "hash-1",
                "result": {"response_harmfulness": "unharmful"},
                "status": "completed",
                "error": None,
            }
        ],
    }
    unlabeled = {
        "id": "entry-2",
        "messages": entry["messages"],
        "annotations": [],
        "model": "model-b",
        "judge": "",
        "metadata": {"prompt_id": "prompt-2", "dataset_name": "dataset-a", "generation_hash": "hash-2"},
        "label_runs": [],
        "safety_runs": [],
    }
    path.write_text(json.dumps(entry) + "\n" + json.dumps(unlabeled) + "\n", encoding="utf-8")

    entries = mod.load_all_entries(path)
    judges = {e["judge"] for e in entries}

    assert len(entries) == 2
    assert judges == {"gpt-5.4-mini", "gpt-5.4-2026-03-05"}
    assert all(e["id"] == "entry-1" for e in entries)
    assert all("wildguard" in (e["metadata"].get("safety_scores") or {}) for e in entries)
