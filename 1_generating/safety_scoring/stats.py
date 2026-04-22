#!/usr/bin/env python3
"""
Print safety score statistics from data/1_generated/*.jsonl.

Usage:
    cd 1_generating
    uv run python safety_scoring/stats.py
    uv run python safety_scoring/stats.py --guard wildguard
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

_SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = _SCRIPT_DIR.parent.parent / "data"
GENERATED_DIR = DATA_ROOT / "1_generated"

console = Console()


def load_all_entries() -> list[dict]:
    entries = []
    for f in sorted(GENERATED_DIR.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def show_wildguard_stats(entries: list[dict]) -> None:
    scored = [e for e in entries if "wildguard" in e.get("metadata", {}).get("safety_scores", {})]
    if not scored:
        console.print("[yellow]WildGuard: no scored entries found.[/]")
        return

    ph = Counter(e["metadata"]["safety_scores"]["wildguard"].get("prompt_harmfulness") for e in scored)
    rh = Counter(e["metadata"]["safety_scores"]["wildguard"].get("response_harmfulness") for e in scored)
    rr = Counter(e["metadata"]["safety_scores"]["wildguard"].get("response_refusal") for e in scored)
    errs = sum(1 for e in scored if e["metadata"]["safety_scores"]["wildguard"].get("is_parsing_error"))

    table = Table(title=f"WildGuard — {len(scored):,} entries", show_header=True)
    table.add_column("Field")
    table.add_column("Value")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")

    for field, counter in [("prompt_harmfulness", ph), ("response_harmfulness", rh), ("response_refusal", rr)]:
        for val, count in sorted(counter.items(), key=lambda x: -x[1]):
            pct = f"{count / len(scored) * 100:.1f}%"
            table.add_row(field, str(val), f"{count:,}", pct)

    if errs:
        table.add_row("[red]parsing_error[/]", "True", f"{errs:,}", f"{errs / len(scored) * 100:.1f}%")

    console.print(table)


def show_qwen3guard_stats(entries: list[dict]) -> None:
    scored = [e for e in entries if "qwen3guard" in e.get("metadata", {}).get("safety_scores", {})]
    if not scored:
        console.print("[yellow]Qwen3Guard: no scored entries found.[/]")
        return

    ps = Counter(e["metadata"]["safety_scores"]["qwen3guard"].get("prompt_safety") for e in scored)
    rs = Counter(e["metadata"]["safety_scores"]["qwen3guard"].get("response_safety") for e in scored)
    rr = Counter(e["metadata"]["safety_scores"]["qwen3guard"].get("response_refusal") for e in scored)

    prompt_cats: Counter = Counter()
    resp_cats: Counter = Counter()
    for e in scored:
        q = e["metadata"]["safety_scores"]["qwen3guard"]
        for c in q.get("prompt_categories") or []:
            prompt_cats[c] += 1
        for c in q.get("response_categories") or []:
            resp_cats[c] += 1

    table = Table(title=f"Qwen3Guard — {len(scored):,} entries", show_header=True)
    table.add_column("Field")
    table.add_column("Value")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")

    for field, counter in [("prompt_safety", ps), ("response_safety", rs), ("response_refusal", rr)]:
        for val, count in sorted(counter.items(), key=lambda x: -x[1]):
            pct = f"{count / len(scored) * 100:.1f}%"
            table.add_row(field, str(val), f"{count:,}", pct)

    console.print(table)

    cat_table = Table(title="Qwen3Guard — Categories", show_header=True)
    cat_table.add_column("Category")
    cat_table.add_column("Prompt", justify="right")
    cat_table.add_column("Response", justify="right")

    all_cats = sorted(set(prompt_cats) | set(resp_cats))
    for cat in all_cats:
        cat_table.add_row(cat, str(prompt_cats.get(cat, "")), str(resp_cats.get(cat, "")))

    console.print(cat_table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--guard", choices=["wildguard", "qwen3guard", "both"], default="both")
    args = parser.parse_args()

    entries = load_all_entries()
    console.print(f"[bold]Total entries:[/] {len(entries):,}\n")

    if args.guard in ("wildguard", "both"):
        show_wildguard_stats(entries)
        console.print()
    if args.guard in ("qwen3guard", "both"):
        show_qwen3guard_stats(entries)


if __name__ == "__main__":
    main()
