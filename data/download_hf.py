#!/usr/bin/env python3
"""Download the little-steer dataset from HuggingFace and save as local JSONL."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()

HF_REPO_ID = "AISafety-Student/little-steer"

_ROOT = Path(__file__).resolve().parent
_DATASET_PATH = _ROOT / "dataset.jsonl"


def _fmt_size(path: Path) -> str:
    if not path.exists():
        return "[red]not found[/red]"
    count = sum(1 for line in open(path, encoding="utf-8") if line.strip())
    return f"[green]exists[/green] ({count:,} entries)"


def show_status() -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("HuggingFace repo", f"[cyan]{HF_REPO_ID}[/cyan]")
    table.add_row("local dataset.jsonl", _fmt_size(_DATASET_PATH))
    console.print(table)


def download(output_path: Path, split: str = "train") -> None:
    """Download HF dataset and write as JSONL of ConversationEntry objects."""
    from datasets import load_dataset as hf_load_dataset
    from thesis_schema import ConversationEntry

    console.print(f"[bold]Downloading[/bold] [cyan]{HF_REPO_ID}[/cyan] (split={split}) …")
    ds = hf_load_dataset(HF_REPO_ID, split=split)
    console.print(f"  {len(ds):,} rows fetched")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    written = 0

    console.print(f"[bold]Converting and writing[/bold] → [cyan]{output_path}[/cyan] …")
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            try:
                entry = ConversationEntry.model_validate(row)
                f.write(entry.model_dump_json() + "\n")
                written += 1
            except Exception as e:
                errors += 1
                console.print(f"  [yellow]⚠ skipped row ({e})[/yellow]")

    if errors:
        console.print(f"[yellow]Done with warnings.[/yellow] {written:,} written, {errors} skipped.")
    else:
        console.print(f"[green bold]Done.[/green bold] {written:,} entries written.")


def main() -> None:
    console.rule("[bold]little-steer dataset download[/bold]")
    show_status()
    console.print()

    default_out = str(_DATASET_PATH)
    choices = {
        "1": f"Download to {default_out}",
        "2": "Download to a custom path",
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

    if choice == "2":
        raw = Prompt.ask("Output path", default=default_out)
        output_path = Path(raw).expanduser().resolve()
    else:
        output_path = _DATASET_PATH

    if output_path.exists():
        if not Confirm.ask(f"[yellow]{output_path} already exists — overwrite?[/yellow]", default=False):
            console.print("Aborted.")
            return
        console.print()

    download(output_path)


if __name__ == "__main__":
    main()
