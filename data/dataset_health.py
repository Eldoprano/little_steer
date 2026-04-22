#!/usr/bin/env python3
"""Check the canonical dataset for stale or inconsistent entries."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from thesis_schema import ConversationEntry

console = Console()
DEFAULT_DATASET = Path(__file__).resolve().parent / "dataset.jsonl"


def _load_lines(path: Path) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            text = line.strip()
            if text:
                lines.append((idx, text))
    return lines


def check_dataset(path: Path, allowed_taxonomy: str | None = "v6") -> tuple[Counter, dict[str, list[str]]]:
    counts: Counter = Counter()
    details: dict[str, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()

    for line_no, raw in _load_lines(path):
        try:
            entry = ConversationEntry.model_validate_json(raw)
        except Exception as exc:
            counts["schema_invalid"] += 1
            details["schema_invalid"].append(f"line {line_no}: {exc}")
            continue

        entry_ref = f"{entry.id} (line {line_no})"

        if entry.id in seen_ids:
            counts["duplicate_id"] += 1
            details["duplicate_id"].append(entry_ref)
        seen_ids.add(entry.id)

        prompt_id = entry.metadata.get("prompt_id")
        if not prompt_id:
            counts["missing_prompt_id"] += 1
            details["missing_prompt_id"].append(entry_ref)

        active_key = entry.metadata.get("active_label_run")
        active_run = entry.active_label_run()
        if active_key and active_run is None:
            counts["missing_active_run"] += 1
            details["missing_active_run"].append(entry_ref)

        if active_run is not None:
            if entry.annotations != active_run.spans:
                counts["annotations_mismatch"] += 1
                details["annotations_mismatch"].append(entry_ref)
            if entry.judge != active_run.judge_name:
                counts["judge_mismatch"] += 1
                details["judge_mismatch"].append(entry_ref)

        current_hash = entry.metadata.get("generation_hash") or entry.generation_hash()
        for run in entry.label_runs:
            if run.generation_hash != current_hash:
                counts["stale_label_run"] += 1
                details["stale_label_run"].append(f"{entry_ref} [{run.key}]")
            if allowed_taxonomy and run.taxonomy_version != allowed_taxonomy:
                counts["noncanonical_taxonomy"] += 1
                details["noncanonical_taxonomy"].append(f"{entry_ref} [{run.taxonomy_version}]")
            for span in run.spans:
                if span.message_idx < 0 or span.message_idx >= len(entry.messages):
                    counts["span_out_of_bounds"] += 1
                    details["span_out_of_bounds"].append(entry_ref)
                    continue
                msg = entry.messages[span.message_idx]["content"]
                if span.char_start < 0 or span.char_end > len(msg):
                    counts["span_out_of_bounds"] += 1
                    details["span_out_of_bounds"].append(entry_ref)
                    continue
                if msg[span.char_start:span.char_end] != span.text:
                    counts["span_text_mismatch"] += 1
                    details["span_text_mismatch"].append(entry_ref)

        for run in entry.safety_runs:
            if run.generation_hash != current_hash:
                counts["stale_safety_run"] += 1
                details["stale_safety_run"].append(f"{entry_ref} [{run.key}]")

        quality = entry.metadata.get("quality") or {}
        approved = entry.metadata.get("approved")
        quality_approved = quality.get("approved")
        if approved is not None and quality_approved is not None and approved != quality_approved:
            counts["quality_mismatch"] += 1
            details["quality_mismatch"].append(entry_ref)

        if entry.get_reasoning_content() is None:
            counts["missing_reasoning"] += 1
            details["missing_reasoning"].append(entry_ref)

    return counts, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Check canonical dataset health.")
    parser.add_argument("dataset", nargs="?", default=str(DEFAULT_DATASET))
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--taxonomy-version", default="v6")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found:[/red] {dataset_path}")
        raise SystemExit(1)

    counts, details = check_dataset(dataset_path, allowed_taxonomy=args.taxonomy_version)

    table = Table(title="Dataset Health", show_lines=True)
    table.add_column("Issue", style="cyan")
    table.add_column("Count", justify="right")

    total_issues = 0
    for issue, count in sorted(counts.items()):
        table.add_row(issue, str(count))
        total_issues += count

    if total_issues == 0:
        table.add_row("ok", "0")
    console.print(table)

    if args.details and details:
        for issue, items in sorted(details.items()):
            console.rule(f"[bold blue]{issue}[/bold blue]")
            for item in items:
                console.print(item)

    raise SystemExit(1 if total_issues else 0)


if __name__ == "__main__":
    main()
