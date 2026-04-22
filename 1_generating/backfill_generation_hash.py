#!/usr/bin/env python3
"""
backfill_generation_hash.py — Add generation_hash to existing JSONL files.

Adds metadata.generation_hash (16-char MD5 of the reasoning content) to any
entry that is missing it.  Safe to re-run: entries that already have the field
are left untouched.

Covers both 1_generated and 2b_labeled directories.

Usage:
    uv run python backfill_generation_hash.py               # dry-run (default)
    uv run python backfill_generation_hash.py --write       # actually write
    uv run python backfill_generation_hash.py --write --data-dir PATH --labeled-dir PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

console = Console(highlight=False)


def _make_generation_hash(reasoning: str | None) -> str:
    return hashlib.md5((reasoning or "").encode("utf-8")).hexdigest()[:16]


def _extract_reasoning(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "reasoning":
            return m.get("content") or ""
    return None


def backfill_file(path: Path, dry_run: bool) -> tuple[int, int]:
    """Add generation_hash to entries that lack it. Returns (total, updated)."""
    lines: list[str] = path.read_text(encoding="utf-8").splitlines()
    updated_lines: list[str] = []
    total = 0
    updated = 0

    for raw in lines:
        raw = raw.strip()
        if not raw:
            updated_lines.append("")
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            updated_lines.append(raw)
            continue

        total += 1
        meta = entry.get("metadata") or {}

        if "generation_hash" not in meta:
            reasoning = _extract_reasoning(entry.get("messages") or [])
            meta["generation_hash"] = _make_generation_hash(reasoning)
            entry["metadata"] = meta
            updated_lines.append(json.dumps(entry, ensure_ascii=False))
            updated += 1
        else:
            updated_lines.append(raw)

    if updated and not dry_run:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        tmp.rename(path)

    return total, updated


def scan_dir(root: Path, dry_run: bool) -> list[tuple[str, int, int]]:
    results: list[tuple[str, int, int]] = []
    for path in sorted(root.rglob("*.jsonl")):
        # Skip checkpoint files that happen to end in .jsonl (none exist, but be safe)
        if path.name.endswith(".checkpoint.jsonl"):
            continue
        total, updated = backfill_file(path, dry_run=dry_run)
        if updated:
            results.append((str(path.relative_to(root.parent)), total, updated))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Actually write changes (default: dry-run)")
    parser.add_argument("--data-dir", default="../data/1_generated",
                        help="Path to 1_generated directory")
    parser.add_argument("--labeled-dir", default="../data/2b_labeled",
                        help="Path to 2b_labeled directory")
    args = parser.parse_args()

    dry_run = not args.write
    data_dir = (Path(__file__).parent / args.data_dir).resolve()
    labeled_dir = (Path(__file__).parent / args.labeled_dir).resolve()

    if dry_run:
        console.print("[bold yellow]DRY RUN — pass --write to apply changes[/bold yellow]\n")

    all_results: list[tuple[str, int, int]] = []

    for root, label in [(data_dir, "1_generated"), (labeled_dir, "2b_labeled")]:
        if not root.exists():
            console.print(f"[yellow]Not found: {root}[/yellow]")
            continue
        console.print(f"[bold]Scanning {label}:[/bold] {root}")
        results = scan_dir(root, dry_run=dry_run)
        all_results.extend(results)
        if results:
            for rel, total, upd in results:
                console.print(f"  {'(would update)' if dry_run else 'Updated'} {upd}/{total} entries in {rel}")
        else:
            console.print(f"  [green]All entries already have generation_hash[/green]")

    total_entries = sum(t for _, t, _ in all_results)
    total_updated = sum(u for _, _, u in all_results)

    console.print()
    if dry_run:
        console.print(
            f"[bold]Would add generation_hash to [cyan]{total_updated}[/cyan] entries "
            f"across [cyan]{len(all_results)}[/cyan] files.[/bold]"
        )
        console.print("Run with [bold]--write[/bold] to apply.")
    else:
        console.print(
            f"[bold green]Done.[/bold green] Added generation_hash to "
            f"[cyan]{total_updated}[/cyan] entries across [cyan]{len(all_results)}[/cyan] files."
        )


if __name__ == "__main__":
    main()
