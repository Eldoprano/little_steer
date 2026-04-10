#!/usr/bin/env python3
"""run.py — Top-level entry point for the sentence labeler.

Reads a YAML config file that may define one or many judges and runs the
labeling pipeline accordingly.

Single-judge usage (same as before):
    uv run run.py ../../data/1_generated/
    uv run run.py --config lmstudio.yaml ../../data/1_generated/

Multi-judge usage:
    uv run run.py --config compare_api.yaml ../../data/1_generated/

In multi-judge mode each judge writes its results to its own sub-directory
under the configured output dir (e.g. data/labeled/gpt-5-mini/).

Options that mirror compare_api_labelers.py behavior:
    --judge NAME      Only run this judge (repeatable)
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from sentence_labeler.labeler import (
    build_client,
    call_llm,
    format_prompt,
    load_secrets,
    parse_llm_output,
)
from sentence_labeler.pipeline import apply_labeling_output, run_pipeline
from sentence_labeler.schema import JudgeConfig, LabelerConfig, LabelingOutput

console = Console()

_HERE = Path(__file__).parent

# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_msg(entry: "ConversationEntry", role: str) -> str | None:
    for msg in entry.messages:
        if msg["role"] == role:
            return msg["content"]
    return None


def _crop_reasoning(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    match = None
    for m in re.finditer(r"[.!?](?:\s|$)", window):
        match = m
    if match:
        return window[: match.end()].rstrip()
    return window


def _resolve_files(
    files: tuple[str, ...],
    glob_patterns: tuple[str, ...],
) -> list[Path]:
    """Collect and deduplicate input .jsonl paths from explicit args and globs."""
    paths: set[Path] = set()

    for f in files:
        p = Path(f)
        if p.is_dir():
            paths.update(p.glob("*.jsonl"))
        elif p.suffix == ".jsonl" and p.exists():
            paths.add(p.resolve())
        else:
            expanded = list(Path(".").glob(f))
            if expanded:
                paths.update(x.resolve() for x in expanded if x.suffix == ".jsonl")
            else:
                console.print(f"[yellow]Warning: '{f}' not found or not a .jsonl file[/yellow]")

    for pattern in glob_patterns:
        expanded = list(Path(".").glob(pattern))
        paths.update(x.resolve() for x in expanded if x.suffix == ".jsonl")

    return sorted(paths)


# ── Pipeline runner (multi-judge) ──────────────────────────────────────────────

def run_multi_judge(
    input_files: list[Path],
    cfg: LabelerConfig,
    config_path: Path,
    active_judges: list[JudgeConfig],
    reset_checkpoints: bool,
    dry_run: bool,
) -> None:
    """Run the full labeling pipeline for each judge sequentially."""
    base_output_dir = cfg.resolve_path(config_path, cfg.output.dir)
    multi = len(active_judges) > 1

    all_results: list[tuple[str, list[tuple[str, int, int, int, int]]]] = []

    for judge_idx, judge in enumerate(active_judges):
        console.rule(
            f"[bold blue]Judge {judge_idx + 1}/{len(active_judges)}: "
            f"{judge.name}[/bold blue]"
        )

        # LMStudio pause
        if judge.lmstudio_prompt and not dry_run:
            console.print(
                f"\n[bold yellow]>>> LMStudio model required:[/bold yellow] "
                f"[cyan]{judge.name}[/cyan]\n"
                f"    Load/swap to this model in LMStudio then press Enter.",
                highlight=False,
            )
            input()

        # In multi-judge mode, put each judge in its own sub-dir
        out_dir_override = (base_output_dir / judge.name) if multi else None

        run_pipeline(
            input_files=input_files,
            config=cfg,
            config_path=config_path,
            console=console,
            reset_checkpoints=reset_checkpoints,
            dry_run=dry_run,
            judge_cfg=judge,
            output_dir_override=out_dir_override,
        )
        console.print()



# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("files", nargs=-1, required=False)
@click.option(
    "--config", "-c",
    default=None,
    help="Path to config YAML. Defaults to config.yaml next to this script.",
)
@click.option(
    "--glob", "-g", "glob_patterns",
    multiple=True,
    help="Glob pattern(s) to select input files. Repeatable.",
)
@click.option(
    "--judge", "-j", "judge_filter",
    multiple=True,
    help="Run only this judge (by name). Repeatable.",
)
@click.option(
    "--output-dir", "-o",
    default=None,
    help="Override output directory from config.",
)
@click.option(
    "--workers", "-w",
    default=None, type=int,
    help="Override max_workers from config.",
)
@click.option(
    "--retries",
    default=None, type=int,
    help="Override max_retries from config.",
)
@click.option(
    "--dry-run",
    is_flag=True, default=False,
    help="Show which files would be processed without calling the LLM.",
)
@click.option(
    "--reset-checkpoints",
    is_flag=True, default=False,
    help="Delete existing checkpoints and re-label all entries.",
)
def main(
    files: tuple[str, ...],
    config: str | None,
    glob_patterns: tuple[str, ...],
    judge_filter: tuple[str, ...],
    output_dir: str | None,
    workers: int | None,
    retries: int | None,
    dry_run: bool,
    reset_checkpoints: bool,
) -> None:
    """Label reasoning sentences in JSONL files using one or more LLM judges.

    \b
    FILES: explicit .jsonl paths or directories (expanded to *.jsonl).
           Combined with --glob patterns.

    \b
    Examples:
      # Single judge (config.yaml)
      uv run run.py ../../data/1_generated/

      # Different config
      uv run run.py --config lmstudio.yaml ../../data/1_generated/

      # Only run specific judges from a multi-judge config
      uv run run.py --config compare_api.yaml --judge gpt-5-mini ../../data/1_generated/
    """
    # Locate config file
    if config is None:
        config_path = _HERE / "config.yaml"
    else:
        config_path = Path(config).resolve()

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise SystemExit(1)

    cfg = LabelerConfig.from_yaml(config_path)

    # Apply CLI overrides
    if output_dir is not None:
        cfg.output.dir = output_dir
    if workers is not None:
        cfg.pipeline.max_workers = workers
    if retries is not None:
        cfg.pipeline.max_retries = retries

    # Filter judges
    if judge_filter:
        active_judges = [j for j in cfg.judges if j.name in judge_filter]
        if not active_judges:
            console.print(f"[red]None of the specified judges found: {list(judge_filter)}[/red]")
            console.print(f"Available: {[j.name for j in cfg.judges]}")
            raise SystemExit(1)
    else:
        active_judges = cfg.judges

    # Resolve input files
    input_files = _resolve_files(files, glob_patterns)

    multi = len(active_judges) > 1
    console.print(
        f"[bold]run.py[/bold] — "
        f"config: [dim]{config_path.name}[/dim] — "
        f"judges: [cyan]{', '.join(j.name for j in active_judges)}[/cyan] — "
        f"{len(input_files)} file(s)"
    )
    console.print()

    if not input_files and not dry_run:
        console.print(
            "[yellow]No input files found. "
            "Pass .jsonl paths as arguments or use --glob.[/yellow]"
        )
        raise SystemExit(0)

    # ── Standard pipeline mode ──────────────────────────────────────────────
    run_multi_judge(
        input_files=input_files,
        cfg=cfg,
        config_path=config_path,
        active_judges=active_judges,
        reset_checkpoints=reset_checkpoints,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
