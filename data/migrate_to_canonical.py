#!/usr/bin/env python3
"""Migrate existing generated + labeled data into the canonical dataset store."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from thesis_schema import ConversationEntry, LabelRun, SafetyRun

console = Console()


def _hash_text(text: str | None) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()[:16]


def _canonical_entry_id(dataset_name: str, prompt_text: str, model_id: str) -> str:
    return hashlib.md5(f"{dataset_name}::{prompt_text}::{model_id}".encode("utf-8")).hexdigest()[:16]


def _message_content(entry: ConversationEntry, role: str) -> str:
    for msg in entry.messages:
        if msg["role"] == role:
            return msg["content"]
    return ""


def _normalize_generation_metadata(entry: ConversationEntry) -> dict[str, Any]:
    metadata = dict(entry.metadata)
    generation_hash = metadata.get("generation_hash") or entry.generation_hash()
    approved = metadata.get("approved")
    quality = dict(metadata.get("quality") or {})
    if approved is not None and "approved" not in quality:
        quality["approved"] = approved

    return {
        "prompt_id": metadata.get("prompt_id"),
        "dataset_name": metadata.get("dataset_name"),
        "dataset_source": metadata.get("dataset_source"),
        "dataset_path": metadata.get("dataset_path"),
        "dataset_row": metadata.get("dataset_row", {}),
        "prompt_safety": metadata.get("prompt_safety", "unknown"),
        "generation_hash": generation_hash,
        "generation": {
            "run_id": metadata.get("run_id"),
            "generated_at": metadata.get("generated_at"),
            "model_name": metadata.get("model_name"),
            "model_id": entry.model,
            "backend": metadata.get("backend"),
            "quantization": metadata.get("quantization"),
            "max_new_tokens": metadata.get("max_new_tokens"),
            "temperature": metadata.get("temperature"),
            "top_p": metadata.get("top_p"),
            "top_k": metadata.get("top_k"),
            "min_p": metadata.get("min_p"),
            "presence_penalty": metadata.get("presence_penalty"),
            "repetition_penalty": metadata.get("repetition_penalty"),
            "system_prompt": metadata.get("system_prompt"),
            "gpu_memory_utilization": metadata.get("gpu_memory_utilization"),
            "actual_max_model_len": metadata.get("actual_max_model_len"),
            "actual_enforce_eager": metadata.get("actual_enforce_eager"),
            "finish_reason": metadata.get("finish_reason"),
            "n_new_tokens": metadata.get("n_new_tokens"),
            "has_reasoning": metadata.get("has_reasoning"),
        },
        "quality": quality,
        # Keep these mirrors for legacy readers.
        "approved": approved,
        "safety_scores": dict(metadata.get("safety_scores") or {}),
    }


def _label_run_from_legacy(entry: ConversationEntry) -> LabelRun:
    metadata = entry.metadata
    generation_hash = metadata.get("generation_hash") or entry.generation_hash()
    return LabelRun(
        judge_name=entry.judge or metadata.get("judge_name", ""),
        judge_model_id=metadata.get("judge_model_id") or metadata.get("judge_name") or entry.judge or "",
        taxonomy_version=metadata.get("taxonomy_version", ""),
        labeled_at=metadata.get("labeled_at", ""),
        generation_hash=generation_hash,
        reasoning_truncated=bool(metadata.get("reasoning_truncated", False)),
        assessment=dict(metadata.get("assessment") or {}),
        sentence_annotations=[
            {
                "text": ann.meta.get("judge_text", ann.text),
                "labels": ann.labels,
                "safety_score": int(ann.score),
            }
            for ann in entry.annotations
        ],
        spans=list(entry.annotations),
        usage=dict((metadata.get("usage") or {})),
        finish_reason=metadata.get("finish_reason", ""),
        status="completed",
        error=None,
    )


def _safety_runs_from_legacy(entry: ConversationEntry) -> list[SafetyRun]:
    metadata = entry.metadata
    generation_hash = metadata.get("generation_hash") or entry.generation_hash()
    runs: list[SafetyRun] = []
    for guard_name, result in (metadata.get("safety_scores") or {}).items():
        payload = dict(result or {})
        runs.append(
            SafetyRun(
                guard_name=guard_name,
                guard_model_id=guard_name,
                scored_at=payload.pop("scored_at", ""),
                generation_hash=generation_hash,
                result=payload,
                status="completed",
                error=None,
            )
        )
    return runs


@dataclass(frozen=True)
class MatchKey:
    dataset_name: str
    model_id: str
    prompt_hash: str
    reasoning_hash: str
    assistant_hash: str


def _match_key(entry: ConversationEntry) -> MatchKey:
    metadata = entry.metadata
    return MatchKey(
        dataset_name=metadata.get("dataset_name", ""),
        model_id=entry.model,
        prompt_hash=_hash_text(_message_content(entry, "user")),
        reasoning_hash=_hash_text(_message_content(entry, "reasoning")),
        assistant_hash=_hash_text(_message_content(entry, "assistant")),
    )


def build_generated_index(generated_dir: Path) -> tuple[dict[str, ConversationEntry], dict[MatchKey, list[str]], Counter]:
    store: dict[str, ConversationEntry] = {}
    key_index: dict[MatchKey, list[str]] = defaultdict(list)
    counts: Counter = Counter()

    for path in sorted(generated_dir.glob("*.jsonl")):
        counts["generated_files"] += 1
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = ConversationEntry.model_validate_json(line)
                counts["generated_entries"] += 1

                prompt_text = _message_content(entry, "user")
                dataset_name = (entry.metadata or {}).get("dataset_name", "")
                canonical_id = _canonical_entry_id(dataset_name, prompt_text, entry.model)
                canonical = ConversationEntry(
                    id=canonical_id,
                    messages=entry.messages,
                    annotations=[],
                    model=entry.model,
                    judge="",
                    metadata=_normalize_generation_metadata(entry),
                    label_runs=[],
                    safety_runs=[],
                )
                for run in _safety_runs_from_legacy(entry):
                    canonical.upsert_safety_run(run)

                store[canonical_id] = canonical
                key_index[_match_key(canonical)].append(canonical_id)

    return store, key_index, counts


def merge_v6_labels(
    labeled_dir: Path,
    store: dict[str, ConversationEntry],
    key_index: dict[MatchKey, list[str]],
) -> tuple[Counter, dict[str, Any]]:
    counts: Counter = Counter()
    report: dict[str, Any] = {
        "merged": [],
        "no_match": [],
        "ambiguous_match": [],
        "non_v6_skipped": [],
        "content_mismatch": [],
    }

    for path in sorted(labeled_dir.rglob("*.jsonl")):
        counts["labeled_files_seen"] += 1
        in_v6_dir = "/v6/" in path.as_posix()
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                entry = ConversationEntry.model_validate_json(line)
                metadata = entry.metadata or {}
                taxonomy_version = metadata.get("taxonomy_version") or ("v6" if in_v6_dir else "")
                if taxonomy_version != "v6":
                    counts["non_v6_skipped"] += 1
                    report["non_v6_skipped"].append({"file": str(path), "line": line_no, "taxonomy_version": taxonomy_version})
                    continue

                counts["v6_entries_seen"] += 1
                key = _match_key(entry)
                matches = key_index.get(key, [])
                if not matches:
                    counts["no_match"] += 1
                    report["no_match"].append(
                        {
                            "file": str(path),
                            "line": line_no,
                            "legacy_id": entry.id,
                            "dataset_name": metadata.get("dataset_name"),
                            "model_id": entry.model,
                            "generation_hash": metadata.get("generation_hash"),
                        }
                    )
                    continue
                if len(matches) > 1:
                    counts["ambiguous_match"] += 1
                    report["ambiguous_match"].append(
                        {
                            "file": str(path),
                            "line": line_no,
                            "legacy_id": entry.id,
                            "candidate_ids": matches,
                        }
                    )
                    continue

                canonical_id = matches[0]
                canonical = store[canonical_id]
                if canonical.metadata.get("generation_hash") != (metadata.get("generation_hash") or entry.generation_hash()):
                    counts["content_mismatch"] += 1
                    report["content_mismatch"].append(
                        {
                            "file": str(path),
                            "line": line_no,
                            "canonical_id": canonical_id,
                            "legacy_id": entry.id,
                            "canonical_generation_hash": canonical.metadata.get("generation_hash"),
                            "legacy_generation_hash": metadata.get("generation_hash") or entry.generation_hash(),
                        }
                    )
                    continue

                run = _label_run_from_legacy(entry)
                canonical.upsert_label_run(run, activate=True)
                counts["v6_merged"] += 1
                report["merged"].append(
                    {
                        "file": str(path),
                        "line": line_no,
                        "canonical_id": canonical_id,
                        "judge": run.judge_name,
                    }
                )

    return counts, report


def write_store(path: Path, store: dict[str, ConversationEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for entry_id in sorted(store):
            f.write(store[entry_id].model_dump_json() + "\n")
    tmp.replace(path)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate existing dataset into canonical dataset.jsonl.")
    parser.add_argument("--generated-dir", default="data/1_generated")
    parser.add_argument("--labeled-dir", default="data/2b_labeled")
    parser.add_argument("--output", default="data/dataset.jsonl")
    parser.add_argument("--report", default="data/dataset_migration_report.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir).resolve()
    labeled_dir = Path(args.labeled_dir).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()

    console.print("[bold]Building canonical store from generated data[/bold]")
    store, key_index, gen_counts = build_generated_index(generated_dir)
    console.print("[bold]Merging v6 label runs conservatively[/bold]")
    label_counts, report = merge_v6_labels(labeled_dir, store, key_index)

    report["summary"] = {**gen_counts, **label_counts}
    report["output"] = str(output_path)

    table = Table(title="Migration Summary", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    for key, value in sorted(report["summary"].items()):
        table.add_row(key, str(value))
    table.add_row("canonical_entries", str(len(store)))
    console.print(table)

    if args.dry_run:
        console.print("[yellow]Dry run only[/yellow]")
        return

    write_store(output_path, store)
    write_report(report_path, report)
    console.print(f"[green]Wrote canonical dataset:[/green] {output_path}")
    console.print(f"[green]Wrote migration report:[/green] {report_path}")


if __name__ == "__main__":
    main()
