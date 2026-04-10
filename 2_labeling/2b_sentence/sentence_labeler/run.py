"""run.py — CLI entry point."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from .pipeline import run_pipeline
from .schema import LabelerConfig

console = Console()


def _resolve_files(
    files: tuple[str, ...],
    glob_patterns: tuple[str, ...],
) -> list[Path]:
    """Collect and deduplicate input .jsonl paths from explicit args and glob patterns."""
    paths: set[Path] = set()

    for f in files:
        p = Path(f)
        if p.is_dir():
            paths.update(p.glob("*.jsonl"))
        elif p.suffix == ".jsonl" and p.exists():
            paths.add(p.resolve())
        else:
            # Try as glob relative to cwd
            expanded = list(Path(".").glob(f))
            if expanded:
                paths.update(x.resolve() for x in expanded if x.suffix == ".jsonl")
            else:
                console.print(f"[yellow]Warning: '{f}' not found or not a .jsonl file[/yellow]")

    for pattern in glob_patterns:
        expanded = list(Path(".").glob(pattern))
        paths.update(x.resolve() for x in expanded if x.suffix == ".jsonl")

    return sorted(paths)


@click.command()
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
    "--output-dir", "-o",
    default=None,
    help="Override output directory from config.",
)
@click.option(
    "--in-place",
    is_flag=True, default=False,
    help="Overwrite input files atomically (ignores --output-dir).",
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
def cli(
    files: tuple[str, ...],
    config: str | None,
    glob_patterns: tuple[str, ...],
    output_dir: str | None,
    in_place: bool,
    workers: int | None,
    retries: int | None,
    dry_run: bool,
    reset_checkpoints: bool,
) -> None:
    """Label reasoning sentences in JSONL files using an LLM judge.

    \b
    FILES: explicit .jsonl paths or directories (expanded to *.jsonl).
           Combined with --glob patterns.

    \b
    Examples:
      sentence-labeler data/run1.jsonl data/run2.jsonl
      sentence-labeler --glob "data/deepseek*.jsonl" --workers 8
      sentence-labeler --config prod.yaml --output-dir labeled/ data/
      sentence-labeler --dry-run --glob "data/*.jsonl"
    """
    # Locate config file
    if config is None:
        # Default: config.yaml next to this package's parent directory
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config).resolve()

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        raise SystemExit(1)

    cfg = LabelerConfig.from_yaml(config_path)

    # Apply CLI overrides
    if output_dir is not None:
        cfg.output.dir = output_dir
    if in_place:
        cfg.output.in_place = True
    if workers is not None:
        cfg.pipeline.max_workers = workers
    if retries is not None:
        cfg.pipeline.max_retries = retries

    # Resolve input files
    input_files = _resolve_files(files, glob_patterns)

    if not input_files and not dry_run:
        console.print(
            "[yellow]No input files found. "
            "Pass .jsonl paths as arguments or use --glob.[/yellow]"
        )
        raise SystemExit(0)

    console.print(
        f"[bold]sentence-labeler[/bold] — "
        f"judge: [cyan]{cfg.judge.name}[/cyan] ({cfg.judge.backend}) — "
        f"{len(input_files)} file(s)"
    )

    run_pipeline(
        input_files=input_files,
        config=cfg,
        config_path=config_path,
        console=console,
        reset_checkpoints=reset_checkpoints,
        dry_run=dry_run,
    )
