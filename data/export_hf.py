#!/usr/bin/env python3
"""Export the canonical JSONL dataset to Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical dataset to Parquet.")
    parser.add_argument("input", nargs="?", default=str(Path(__file__).resolve().parent / "dataset.jsonl"))
    parser.add_argument("output", nargs="?", default=str(Path(__file__).resolve().parent / "dataset.parquet"))
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    df = pd.read_json(input_path, lines=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    console.print(f"[green]Exported[/green] {len(df)} rows to [cyan]{output_path}[/cyan]")


if __name__ == "__main__":
    main()
