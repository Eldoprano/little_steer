#!/usr/bin/env python3
"""Interactive export and HuggingFace push tool for the canonical dataset."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

HF_REPO_ID = "Eldoprano/little-steer"

_ROOT = Path(__file__).resolve().parent
_DATASET_PATH = _ROOT / "dataset.jsonl"
_PARQUET_PATH = _ROOT / "dataset.parquet"
_SECRETS_PATH = _ROOT.parent / ".secrets.json"


def _load_hf_token() -> str:
    if not _SECRETS_PATH.exists():
        raise FileNotFoundError(f"Secrets file not found: {_SECRETS_PATH}")
    with open(_SECRETS_PATH, encoding="utf-8") as f:
        secrets = json.load(f)
    token = secrets.get("huggingface-api-key") or secrets.get("hf_token") or secrets.get("token")
    if not token:
        raise ValueError("No HuggingFace token found in .secrets.json (key: 'huggingface-api-key')")
    return token


def _mtime(path: Path) -> datetime | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def _fmt_time(dt: datetime | None) -> str:
    if dt is None:
        return "[red]not found[/red]"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _count_lines(path: Path) -> int:
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _dataset_row_count() -> int | None:
    if not _DATASET_PATH.exists():
        return None
    return _count_lines(_DATASET_PATH)


def show_status() -> None:
    dataset_mtime = _mtime(_DATASET_PATH)
    parquet_mtime = _mtime(_PARQUET_PATH)
    row_count = _dataset_row_count()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("dataset.jsonl", f"{_fmt_time(dataset_mtime)}" + (f"  ({row_count:,} entries)" if row_count else ""))
    table.add_row("dataset.parquet", _fmt_time(parquet_mtime))

    if dataset_mtime and parquet_mtime and parquet_mtime < dataset_mtime:
        table.add_row("", "[yellow]⚠ Parquet is older than the JSONL — may be stale[/yellow]")
    elif dataset_mtime and parquet_mtime and parquet_mtime >= dataset_mtime:
        table.add_row("", "[green]✓ Parquet is up to date[/green]")

    console.print(table)


def export_parquet() -> None:
    import pandas as pd

    console.print(f"[bold]Exporting[/bold] dataset.jsonl → dataset.parquet …")
    df = pd.read_json(_DATASET_PATH, lines=True)
    _PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PARQUET_PATH, index=False)
    console.print(f"[green]Done.[/green] {len(df):,} rows written to [cyan]{_PARQUET_PATH}[/cyan]")


def push_to_hub() -> None:
    from datasets import load_dataset

    token = _load_hf_token()
    console.print(f"[bold]Loading dataset …[/bold]")
    ds = load_dataset("json", data_files=str(_DATASET_PATH), split="train")
    console.print(f"  {len(ds):,} rows loaded")
    console.print(f"[bold]Pushing to[/bold] [cyan]{HF_REPO_ID}[/cyan] …")
    ds.push_to_hub(HF_REPO_ID, token=token, private=False)
    console.print(f"[green bold]Done.[/green bold] https://huggingface.co/datasets/{HF_REPO_ID}")


def main() -> None:
    if not _DATASET_PATH.exists():
        console.print(f"[red]dataset.jsonl not found at {_DATASET_PATH}[/red]")
        sys.exit(1)

    console.rule("[bold]little-steer dataset export[/bold]")
    show_status()
    console.print()

    choices = {
        "1": "Update Parquet from dataset.jsonl",
        "2": "Push dataset to HuggingFace",
        "3": "Update Parquet, then push to HuggingFace",
        "q": "Quit",
    }

    for key, label in choices.items():
        console.print(f"  [bold cyan]{key}[/bold cyan]  {label}")

    console.print()
    choice = Prompt.ask("What do you want to do?", choices=list(choices.keys()), default="q")

    if choice == "q":
        console.print("Bye!")
        return

    console.print()

    if choice in ("1", "3"):
        export_parquet()
        console.print()

    if choice in ("2", "3"):
        push_to_hub()


if __name__ == "__main__":
    main()
