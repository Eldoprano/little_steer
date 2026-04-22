#!/usr/bin/env python3
"""
check_generation_consistency.py — Detect labelers that labeled different generations.

For each (model, dataset, entry_id) triple, checks whether all labeled versions
share the same generation_hash.  A mismatch means at least two labelers labeled
different generations of the same prompt.

Output modes:
  --summary    Print aggregated counts (default)
  --detail     Print each affected entry
  --export CSV Write affected entries to a CSV file

Usage:
    uv run python check_generation_consistency.py
    uv run python check_generation_consistency.py --detail
    uv run python check_generation_consistency.py --export mismatches.csv
    uv run python check_generation_consistency.py --labeled-dir PATH
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

console = Console(highlight=False)


def _stem_to_model_dataset(stem: str) -> tuple[str, str]:
    """Split 'modelname_datasetname_judgename' into (model_dataset_prefix, judge).

    The labeler name is the last underscore-separated token; the rest is
    model+dataset (which may itself contain underscores).
    """
    parts = stem.split("_")
    if len(parts) < 2:
        return stem, ""
    return "_".join(parts[:-1]), parts[-1]


def load_labeled_dir(labeled_dir: Path) -> dict[str, dict[str, list[tuple[str, str]]]]:
    """Load all JSONL files.

    Returns: {model_dataset_stem: {entry_id: [(judge, generation_hash), ...]}}
    """
    data: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))

    for path in sorted(labeled_dir.rglob("*.jsonl")):
        if ".checkpoint" in path.name or path.name.endswith(".tmp"):
            continue
        stem = path.stem
        # Identify the model+dataset prefix (everything except the last _<judge>)
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        # The judge is identified by the labeler suffix.  We can't always split
        # unambiguously, but the model+dataset pair always appears as a prefix
        # of the 1_generated filename, so we try longest-match against known
        # generated files (skipped here — we just use all-but-last convention).
        # NOTE: if model/dataset names themselves contain underscores, the last
        # underscore-separated token is still the judge (that's our naming rule).
        judge = parts[-1]
        model_dataset = "_".join(parts[:-1])

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eid = entry.get("id")
                if not eid:
                    continue
                ghash = (entry.get("metadata") or {}).get("generation_hash", "")
                data[model_dataset][eid].append((judge, ghash))

    return data


def find_mismatches(
    data: dict[str, dict[str, list[tuple[str, str]]]]
) -> list[dict]:
    """Return list of mismatch records."""
    mismatches: list[dict] = []
    for model_dataset, entries in sorted(data.items()):
        for eid, versions in sorted(entries.items()):
            if len(versions) < 2:
                continue  # Only one labeler — can't compare
            hashes = {ghash for _, ghash in versions if ghash}
            if len(hashes) <= 1:
                continue  # All agree (or all empty)
            # Mismatch found
            mismatches.append({
                "model_dataset": model_dataset,
                "entry_id": eid,
                "versions": versions,
                "unique_hashes": sorted(hashes),
            })
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled-dir", default="../data/2b_labeled",
                        help="Path to 2b_labeled directory (searches recursively)")
    parser.add_argument("--detail", action="store_true",
                        help="Print each affected entry")
    parser.add_argument("--export", metavar="CSV",
                        help="Write affected entries to CSV file")
    args = parser.parse_args()

    labeled_dir = (Path(__file__).parent / args.labeled_dir).resolve()
    if not labeled_dir.exists():
        console.print(f"[red]Labeled dir not found: {labeled_dir}[/red]")
        sys.exit(1)

    console.print(f"[bold]Scanning:[/bold] {labeled_dir}\n")
    data = load_labeled_dir(labeled_dir)

    total_entries = sum(len(v) for v in data.values())
    total_multi = sum(1 for entries in data.values() for vs in entries.values() if len(vs) >= 2)
    console.print(f"Loaded {total_entries:,} (entry_id, judge) pairs across {len(data)} model+dataset prefixes")
    console.print(f"Entries labeled by ≥2 judges: {total_multi:,}\n")

    mismatches = find_mismatches(data)

    # Summary by model_dataset
    by_prefix: dict[str, int] = defaultdict(int)
    for m in mismatches:
        by_prefix[m["model_dataset"]] += 1

    table = Table(title="Generation Consistency Check", box=box.SIMPLE, show_lines=True)
    table.add_column("Model+Dataset", style="cyan")
    table.add_column("Mismatches", justify="right", style="red")

    for prefix, count in sorted(by_prefix.items(), key=lambda x: -x[1]):
        table.add_row(prefix, str(count))

    if by_prefix:
        table.add_section()
        table.add_row("[bold]TOTAL[/bold]", f"[bold red]{len(mismatches)}[/bold red]")
        console.print(table)
    else:
        console.print("[bold green]No mismatches found — all labelers labeled the same generation.[/bold green]")
        return

    if args.detail:
        console.print()
        detail = Table(title="Mismatch Details", box=box.SIMPLE)
        detail.add_column("Model+Dataset")
        detail.add_column("Entry ID")
        detail.add_column("Judge")
        detail.add_column("Hash")
        for m in mismatches:
            for judge, ghash in m["versions"]:
                detail.add_row(m["model_dataset"], m["entry_id"], judge, ghash or "(missing)")
            detail.add_section()
        console.print(detail)

    if args.export:
        out = Path(args.export)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model_dataset", "entry_id", "judge", "generation_hash"])
            for m in mismatches:
                for judge, ghash in m["versions"]:
                    writer.writerow([m["model_dataset"], m["entry_id"], judge, ghash])
        console.print(f"\n[green]Exported to {out}[/green]")

    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[red]{len(mismatches)}[/red] entries have generation mismatches across judges."
    )
    if mismatches:
        console.print(
            "[yellow]Resolution:[/yellow] Run fix_quality.py --sync-labeled to purge "
            "stale labeled entries, then re-run the labeling pipeline."
        )


if __name__ == "__main__":
    main()
