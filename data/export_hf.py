#!/usr/bin/env python3
"""Export the canonical JSONL dataset to Parquet and/or push to HuggingFace Hub.

Usage:
    uv run python export_hf.py                    # export to dataset.parquet only
    uv run python export_hf.py --push             # export + push to HuggingFace
    uv run python export_hf.py --push --dry-run   # show what would be pushed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

console = Console()

HF_REPO_ID = "Eldoprano/little-steer"

_SECRETS_PATH = Path(__file__).resolve().parent.parent / ".secrets.json"
_DEFAULT_INPUT = Path(__file__).resolve().parent / "dataset.jsonl"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "dataset.parquet"


def _load_hf_token() -> str:
    if not _SECRETS_PATH.exists():
        raise FileNotFoundError(f"Secrets file not found: {_SECRETS_PATH}")
    with open(_SECRETS_PATH, encoding="utf-8") as f:
        secrets = json.load(f)
    token = secrets.get("huggingface-api-key") or secrets.get("hf_token") or secrets.get("token")
    if not token:
        raise ValueError("No HuggingFace token found in .secrets.json (key: 'huggingface-api-key')")
    return token


def export_parquet(input_path: Path, output_path: Path) -> "pd.DataFrame":
    import pandas as pd

    df = pd.read_json(input_path, lines=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    console.print(f"[green]Exported[/green] {len(df):,} rows → [cyan]{output_path}[/cyan]")
    return df


def push_to_hub(input_path: Path, repo_id: str, token: str, dry_run: bool) -> None:
    from datasets import load_dataset

    console.print(f"[bold]Loading dataset from[/bold] {input_path}")
    ds = load_dataset("json", data_files=str(input_path), split="train")
    console.print(f"  {len(ds):,} rows, features: {list(ds.features.keys())}")

    if dry_run:
        console.print(f"[yellow]Dry run — would push to[/yellow] [bold]{repo_id}[/bold]")
        return

    console.print(f"[bold]Pushing to[/bold] [cyan]{repo_id}[/cyan] …")
    ds.push_to_hub(
        repo_id,
        token=token,
        private=False,
    )
    console.print(f"[green bold]Done.[/green bold] Dataset live at https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=str(_DEFAULT_INPUT), help="Path to dataset.jsonl")
    parser.add_argument("output", nargs="?", default=str(_DEFAULT_OUTPUT), help="Path to output Parquet file")
    parser.add_argument("--push", action="store_true", help="Push to HuggingFace Hub after export")
    parser.add_argument("--push-only", action="store_true", help="Skip Parquet export, push JSONL directly")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing/pushing")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        console.print(f"[red]Dataset not found:[/red] {input_path}")
        raise SystemExit(1)

    if not args.push_only:
        if args.dry_run:
            import pandas as pd
            df = pd.read_json(input_path, lines=True)
            console.print(f"[yellow]Dry run — would export[/yellow] {len(df):,} rows → [cyan]{output_path}[/cyan]")
        else:
            export_parquet(input_path, output_path)

    if args.push or args.push_only:
        token = _load_hf_token()
        push_to_hub(input_path, HF_REPO_ID, token, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
