#!/usr/bin/env python3
"""
One-time backfill: propagate already-computed safety scores from data/1_generated/
to all data/2b_labeled/v6/*.jsonl files.

Matches entries by metadata.generation_hash (MD5 of reasoning content), which is
unique per model generation and avoids the cross-model ID collision problem that
entry.id (hash of prompt only) has.

Usage:
    cd 1_generating
    uv run python safety_scoring/backfill_labeled.py           # dry-run by default
    uv run python safety_scoring/backfill_labeled.py --apply   # actually write changes
    uv run python safety_scoring/backfill_labeled.py --guard wildguard --apply
"""

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

_SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = _SCRIPT_DIR.parent.parent / "data"
GENERATED_DIR = DATA_ROOT / "1_generated"
LABELED_DIR = DATA_ROOT / "2b_labeled" / "v6"

console = Console()


def read_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_jsonl_atomic(path: Path, entries: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp.rename(path)


def build_source_index(guards: list[str]) -> dict[str, dict]:
    """
    Builds a dict: generation_hash -> {scores: {...}}
    generation_hash is unique per model generation, so it never collides
    across different model files the way entry.id (prompt hash) does.
    Only includes entries that have at least one of the requested guards scored.
    """
    index: dict[str, dict] = {}
    for src_file in sorted(GENERATED_DIR.glob("*.jsonl")):
        for entry in read_jsonl(src_file):
            ghash = entry.get("metadata", {}).get("generation_hash", "")
            if not ghash:
                continue
            scores = entry.get("metadata", {}).get("safety_scores", {})
            relevant = {g: scores[g] for g in guards if g in scores}
            if not relevant:
                continue
            index[ghash] = {"scores": relevant}
    return index


def backfill(guards: list[str], apply: bool) -> None:
    console.print(f"[bold]Guards:[/] {', '.join(guards)}")
    console.print(f"[bold]Mode:[/] {'[green]APPLY[/]' if apply else '[yellow]DRY RUN[/]'}\n")

    console.print("Building source index from data/1_generated/ ...")
    source_index = build_source_index(guards)
    console.print(f"  {len(source_index)} source entries with scores.\n")

    table = Table(show_header=True, show_lines=False)
    table.add_column("File", style="cyan", no_wrap=True, max_width=60)
    table.add_column("Entries", justify="right")
    table.add_column("Updated", justify="right", style="green")
    table.add_column("Skipped (no src)", justify="right", style="dim")

    total_updated = 0

    for labeled_file in sorted(LABELED_DIR.rglob("*.jsonl")):
        stem = labeled_file.stem
        # Skip checkpoint files and non-data files
        if stem.startswith(".") or not any(c == "_" for c in stem):
            continue

        entries = read_jsonl(labeled_file)
        updated = 0
        skipped_no_src = 0
        changed = False

        for entry in entries:
            ghash = entry.get("metadata", {}).get("generation_hash", "")
            if not ghash or ghash not in source_index:
                skipped_no_src += 1
                continue

            src = source_index[ghash]

            scores_block = entry.setdefault("metadata", {}).setdefault("safety_scores", {})
            for guard, score in src["scores"].items():
                if guard not in scores_block:
                    scores_block[guard] = score
                    updated += 1
                    changed = True

        if apply and changed:
            write_jsonl_atomic(labeled_file, entries)

        rel = labeled_file.relative_to(DATA_ROOT)
        table.add_row(
            str(rel),
            str(len(entries)),
            str(updated),
            str(skipped_no_src) if skipped_no_src else "",
        )
        total_updated += updated

    console.print(table)
    console.print(f"\n[bold]Total:[/] [green]{total_updated}[/] score fields added")
    if not apply:
        console.print("\n[yellow]--dry-run mode: no files written. Pass --apply to apply changes.[/]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill safety scores from data/1_generated/ into data/2b_labeled/**."
    )
    parser.add_argument(
        "--guard",
        choices=["wildguard", "qwen3guard", "both"],
        default="both",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes (default is dry-run).",
    )
    args = parser.parse_args()
    guards = ["wildguard", "qwen3guard"] if args.guard == "both" else [args.guard]
    backfill(guards, apply=args.apply)


if __name__ == "__main__":
    main()
