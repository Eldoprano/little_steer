"""Generate a balanced IAA (Inter-Annotator Agreement) work order.

The work order picks a fixed number of entries per (model, dataset) bucket,
prioritises already LLM-labeled entries so labelers can compare against
existing judge runs, and interleaves them round-robin so any prefix remains
balanced.

Interactive mode:
    uv run generate_iaa_work_order.py

Non-interactive (all answers supplied via flags):
    uv run generate_iaa_work_order.py \\
        --pairs deepseek,ministral,gpt,qwen \\
        --datasets clear_harm,strong_reject,xs_test \\
        --per-bucket 8 \\
        --output work_order_iaa.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import click
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

HERE = Path(__file__).parent
DATASET_FILE = (HERE / "../../data/dataset.jsonl").resolve()
DEFAULT_OUTPUT = HERE / "work_order_iaa.json"

console = Console()

# ── Model pairs ───────────────────────────────────────────────────────────────

PAIRS: dict[str, tuple[str, str]] = {
    "deepseek": (
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "AISafety-Student/DeepSeek-R1-Distill-Llama-8B-heretic",
    ),
    "ministral": (
        "mistralai/Ministral-3-8B-Reasoning-2512",
        "ministral-3-8b-reasoning-2512-heretic_gguf",
    ),
    "gpt": (
        "gpt-oss-20b",
        "gpt-oss-20b-heretic-ara-v3-i1",
    ),
    "qwen": (
        "Qwen/Qwen3.5-9B",
        "trohrbaugh/Qwen3.5-9B-heretic-v2",
    ),
}

ALL_DATASETS = ["clear_harm", "strong_reject", "xs_test", "lima"]


# ── Entry record ──────────────────────────────────────────────────────────────

class BucketEntry(NamedTuple):
    id: str
    prompt_id: str
    is_labeled: bool  # has at least one label_run


# ── Data loading ──────────────────────────────────────────────────────────────

def load_buckets(
    pairs: dict[str, tuple[str, str]],
    datasets: list[str],
) -> dict[tuple[str, str], list[BucketEntry]]:
    all_models: set[str] = {m for m1, m2 in pairs.values() for m in (m1, m2)}
    ds_set = set(datasets)
    buckets: dict[tuple[str, str], list[BucketEntry]] = defaultdict(list)

    with open(DATASET_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            model = entry.get("model", "")
            ds = entry.get("metadata", {}).get("dataset_name", "")
            pid = entry.get("metadata", {}).get("prompt_id", "")
            eid = entry.get("id", "")
            if not eid or model not in all_models or ds not in ds_set:
                continue
            is_labeled = bool(entry.get("label_runs"))
            buckets[(model, ds)].append(BucketEntry(eid, pid, is_labeled))

    return dict(buckets)


# ── Size options ──────────────────────────────────────────────────────────────

def compute_size_options(
    buckets: dict[tuple[str, str], list[BucketEntry]],
    n_buckets: int,
) -> list[tuple[int, int, int]]:
    """Return (per_bucket, total, min_pct_labeled) rows for candidate sizes."""
    if not buckets:
        return []
    min_bucket_size = min(len(v) for v in buckets.values())
    labeled_per_bucket = {k: sum(1 for e in v if e.is_labeled) for k, v in buckets.items()}
    min_labeled = min(labeled_per_bucket.values())

    options = []
    for n in [3, 4, 6, 8, 10, 12, 16, 24]:
        if n > min_bucket_size:
            continue
        total = n * n_buckets
        pct = round(min(n, min_labeled) / n * 100)
        options.append((n, total, pct))
    return options


def print_size_table(
    options: list[tuple[int, int, int]],
    suggested_idx: int,
) -> None:
    table = Table(title="Size options", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Per bucket", justify="right")
    table.add_column("Total entries", justify="right")
    table.add_column("Min % LLM-labeled", justify="right")

    for i, (n, total, pct) in enumerate(options, 1):
        pct_str = f"[green]{pct}%[/green]" if pct == 100 else f"[yellow]{pct}%[/yellow]"
        marker = " ◀" if i == suggested_idx else ""
        table.add_row(str(i), str(n), str(total), pct_str + marker)

    console.print(table)


# ── Entry selection ───────────────────────────────────────────────────────────

def select_entries(
    buckets: dict[tuple[str, str], list[BucketEntry]],
    pairs: dict[str, tuple[str, str]],
    per_bucket: int,
    prefer_labeled: bool,
    seed: int,
) -> dict[tuple[str, str], list[str]]:
    """
    For each bucket pick `per_bucket` entries.

    Priority order (when prefer_labeled=True):
      1. Already LLM-labeled + prompt shared with partner model
      2. Already LLM-labeled
      3. Not labeled + prompt shared with partner model
      4. Not labeled

    Within each tier entries are shuffled randomly.
    """
    rng = random.Random(seed)

    # Build prompt_id → set of models that have it
    pid_to_models: dict[str, set[str]] = defaultdict(set)
    for (model, ds), entries in buckets.items():
        for e in entries:
            if e.prompt_id:
                pid_to_models[e.prompt_id].add(model)

    model_to_partner: dict[str, str] = {}
    for m1, m2 in pairs.values():
        model_to_partner[m1] = m2
        model_to_partner[m2] = m1

    selected: dict[tuple[str, str], list[str]] = {}

    for (model, ds), entries in buckets.items():
        partner = model_to_partner.get(model)

        def sort_key(e: BucketEntry) -> tuple:
            is_matched = partner is not None and partner in pid_to_models.get(e.prompt_id, set())
            return (
                -int(e.is_labeled),   # labeled first
                -int(is_matched),     # matched-pair prompt second
            )

        shuffled = list(entries)
        rng.shuffle(shuffled)
        if prefer_labeled:
            shuffled.sort(key=sort_key)

        selected[(model, ds)] = [e.id for e in shuffled[:per_bucket]]

    return selected


# ── Round-robin ordering ──────────────────────────────────────────────────────

def build_flat_order(
    selected: dict[tuple[str, str], list[str]],
    datasets: list[str],
    pairs: dict[str, tuple[str, str]],
    per_bucket: int,
) -> list[dict[str, str]]:
    """
    Interleave entries so any prefix is maximally balanced.

    Round structure: for each round index r in [0, per_bucket):
      iterate datasets → pairs → (censored, uncensored)

    Stopping after k complete rounds gives exactly k entries per bucket.
    Stopping mid-round adds at most 1 extra from some buckets.
    """
    # Fixed bucket order
    bucket_order: list[tuple[str, str]] = []
    for ds in datasets:
        for m1, m2 in pairs.values():
            for m in (m1, m2):
                if (m, ds) in selected:
                    bucket_order.append((m, ds))

    flat: list[dict[str, str]] = []
    for r in range(per_bucket):
        for key in bucket_order:
            entries = selected.get(key, [])
            if r < len(entries):
                flat.append({"file": "dataset.jsonl", "id": entries[r]})

    return flat


# ── Stats display ─────────────────────────────────────────────────────────────

def print_stats(
    selected: dict[tuple[str, str], list[str]],
    buckets: dict[tuple[str, str], list[BucketEntry]],
    pairs: dict[str, tuple[str, str]],
) -> None:
    id_to_entry: dict[str, BucketEntry] = {
        e.id: e for entries in buckets.values() for e in entries
    }

    pid_to_models: dict[str, set[str]] = defaultdict(set)
    for (model, ds), entries in buckets.items():
        for e in entries:
            if e.prompt_id:
                pid_to_models[e.prompt_id].add(model)

    model_to_partner: dict[str, str] = {}
    for m1, m2 in pairs.values():
        model_to_partner[m1] = m2
        model_to_partner[m2] = m1

    table = Table(title="Selection summary", show_header=True, header_style="bold")
    table.add_column("Model", style="dim", max_width=42)
    table.add_column("Dataset")
    table.add_column("N", justify="right")
    table.add_column("LLM-labeled", justify="right")
    table.add_column("Matched", justify="right")

    total = total_labeled = total_matched = 0

    for (model, ds) in sorted(selected.keys()):
        ids = selected[(model, ds)]
        n = len(ids)
        labeled = sum(1 for eid in ids if id_to_entry.get(eid, BucketEntry("", "", False)).is_labeled)
        partner = model_to_partner.get(model)
        matched = 0
        if partner:
            for eid in ids:
                e = id_to_entry.get(eid)
                if e and partner in pid_to_models.get(e.prompt_id, set()):
                    matched += 1

        short = model.split("/")[-1]
        labeled_style = "green" if labeled == n else "yellow"
        table.add_row(
            short, ds, str(n),
            f"[{labeled_style}]{labeled}/{n}[/{labeled_style}]",
            f"{matched}/{n}",
        )
        total += n
        total_labeled += labeled
        total_matched += matched

    console.print(table)
    console.print(
        f"\nTotal: [bold]{total}[/bold]  |  "
        f"LLM-labeled: [green]{total_labeled}[/green]/{total}  |  "
        f"Matched pairs: [cyan]{total_matched}[/cyan]/{total}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command(help=__doc__)
@click.option("--pairs", default=None, help="Comma-separated pair names (e.g. deepseek,gpt)")
@click.option("--datasets", default=None, help="Comma-separated dataset names")
@click.option("--per-bucket", "per_bucket", type=int, default=None, help="Entries per bucket")
@click.option("--output", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--prefer-labeled/--no-prefer-labeled", default=True,
              help="Prioritise already-labeled entries (default: yes)")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def main(
    pairs: str | None,
    datasets: str | None,
    per_bucket: int | None,
    output: Path,
    seed: int,
    prefer_labeled: bool,
    yes: bool,
) -> None:
    non_interactive = all(x is not None for x in [pairs, datasets, per_bucket])

    # ── 1. Pairs ──────────────────────────────────────────────────────────────
    selected_pairs: dict[str, tuple[str, str]]
    if pairs is not None:
        pair_names = [p.strip() for p in pairs.split(",")]
        bad = [p for p in pair_names if p not in PAIRS]
        if bad:
            console.print(f"[red]Unknown pairs: {bad}. Available: {list(PAIRS)}[/red]")
            sys.exit(1)
        selected_pairs = {k: PAIRS[k] for k in pair_names}
    else:
        console.print("\n[bold cyan]Available model pairs:[/bold cyan]")
        for name, (m1, m2) in PAIRS.items():
            console.print(f"  [bold]{name}[/bold]: {m1.split('/')[-1]} / {m2.split('/')[-1]}")
        raw = Prompt.ask(
            "\nWhich pairs? (comma-separated, or [bold]all[/bold])",
            default="all",
        )
        pair_names = list(PAIRS) if raw.strip().lower() == "all" else [p.strip() for p in raw.split(",")]
        bad = [p for p in pair_names if p not in PAIRS]
        if bad:
            console.print(f"[red]Unknown pairs: {bad}[/red]")
            sys.exit(1)
        selected_pairs = {k: PAIRS[k] for k in pair_names}

    # ── 2. Datasets ───────────────────────────────────────────────────────────
    selected_datasets: list[str]
    if datasets is not None:
        selected_datasets = [d.strip() for d in datasets.split(",")]
        bad = [d for d in selected_datasets if d not in ALL_DATASETS]
        if bad:
            console.print(f"[red]Unknown datasets: {bad}. Available: {ALL_DATASETS}[/red]")
            sys.exit(1)
    else:
        console.print(f"\n[bold cyan]Available datasets:[/bold cyan] {', '.join(ALL_DATASETS)}")
        raw = Prompt.ask(
            "Which datasets? (comma-separated, or [bold]all[/bold])",
            default="clear_harm,strong_reject,xs_test",
        )
        selected_datasets = ALL_DATASETS if raw.strip().lower() == "all" else [d.strip() for d in raw.split(",")]
        bad = [d for d in selected_datasets if d not in ALL_DATASETS]
        if bad:
            console.print(f"[red]Unknown datasets: {bad}[/red]")
            sys.exit(1)

    # ── 3. Load ───────────────────────────────────────────────────────────────
    console.print("\n[dim]Loading dataset...[/dim]")
    buckets = load_buckets(selected_pairs, selected_datasets)
    n_buckets = len(selected_pairs) * 2 * len(selected_datasets)

    if not buckets:
        console.print("[red]No entries found for the given pairs/datasets.[/red]")
        sys.exit(1)

    # ── 4. Per-bucket N ───────────────────────────────────────────────────────
    if per_bucket is not None:
        min_bucket = min(len(v) for v in buckets.values())
        if per_bucket > min_bucket:
            console.print(
                f"[red]--per-bucket {per_bucket} exceeds the smallest bucket ({min_bucket} entries).[/red]"
            )
            sys.exit(1)
    else:
        options = compute_size_options(buckets, n_buckets)
        if not options:
            console.print("[red]No valid size options — buckets are too small.[/red]")
            sys.exit(1)
        # Suggest the option closest to 8 per bucket
        suggested = min(range(len(options)), key=lambda i: abs(options[i][0] - 8)) + 1
        print_size_table(options, suggested)
        choice = IntPrompt.ask("Pick option number", default=suggested)
        if not 1 <= choice <= len(options):
            console.print("[red]Invalid choice.[/red]")
            sys.exit(1)
        per_bucket = options[choice - 1][0]

    # ── 5. Select and order ───────────────────────────────────────────────────
    selected = select_entries(buckets, selected_pairs, per_bucket, prefer_labeled, seed)
    flat_order = build_flat_order(selected, selected_datasets, selected_pairs, per_bucket)

    print_stats(selected, buckets, selected_pairs)

    # ── 6. Confirm and write ──────────────────────────────────────────────────
    if not yes and not non_interactive:
        if not Confirm.ask(f"\nWrite [bold]{len(flat_order)}[/bold] entries to [bold]{output}[/bold]?", default=True):
            console.print("[yellow]Aborted.[/yellow]")
            sys.exit(0)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "strategy": "iaa_balanced",
        "total_entries": len(flat_order),
        "n_buckets": n_buckets,
        "per_bucket": per_bucket,
        "pairs": {k: list(v) for k, v in selected_pairs.items()},
        "datasets": selected_datasets,
        "prefer_labeled": prefer_labeled,
        "dataset_file": str(DATASET_FILE),
        "file_order": ["dataset.jsonl"],
        "flat_order": flat_order,
        "per_file": {"dataset.jsonl": [item["id"] for item in flat_order]},
    }
    output.write_text(json.dumps(payload, indent=2))
    console.print(f"\n[green]✓ Wrote {len(flat_order)} entries to {output}[/green]")


if __name__ == "__main__":
    main()
