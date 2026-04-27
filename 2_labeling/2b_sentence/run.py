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
    build_messages,
    call_llm,
    load_secrets,
    parse_llm_output,
)
from sentence_labeler.pipeline import apply_labeling_output, run_pipeline
from sentence_labeler.schema import JudgeConfig, LabelerConfig, LabelerRegistry, LabelingOutput
from sentence_labeler.taxonomy_loader import inject_taxonomy

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


def _prepare_backfill_work_order(
    labeler_cfg: LabelerConfig,
    config_path: Path,
    judge_name: str,
) -> Path | None:
    """Filter the active work order to entries missing logprobs for this judge.

    Mutates labeler_cfg.pipeline.work_order_file and .overwrite_existing in-place.
    Returns the path to the written temp work order file, or None when there is
    nothing to backfill (judge not logprob-capable, or all entries already done).
    """
    judge_cfg = labeler_cfg.judges[0]
    if not judge_cfg.logprobs:
        console.print(f"[dim]  Backfill: {judge_name} has logprobs=false — skipping[/dim]")
        return None

    if not labeler_cfg.pipeline.work_order_file:
        console.print(f"[yellow]  Backfill: no work_order_file configured for {judge_name}[/yellow]")
        return None

    wo_path = labeler_cfg.resolve_path(config_path, labeler_cfg.pipeline.work_order_file)
    if not wo_path.exists():
        console.print(f"[yellow]  Backfill: work order not found: {wo_path}[/yellow]")
        return None

    with open(wo_path) as f:
        wo = json.load(f)

    flat_order: list[dict] = wo.get("flat_order", [])
    dataset_file: str = wo.get("dataset_file", "")

    if not flat_order:
        console.print(f"[yellow]  Backfill: work order has no flat_order entries[/yellow]")
        return None

    dataset_path = Path(dataset_file) if dataset_file else None
    if not dataset_path or not dataset_path.exists():
        console.print(f"[yellow]  Backfill: dataset file not found: {dataset_file}[/yellow]")
        return None

    # Scan dataset for entries that have this judge's run but are missing logprobs
    target_ids = {item["id"] for item in flat_order}
    needs_backfill: set[str] = set()
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                eid = data.get("id")
                if eid not in target_ids:
                    continue
                for run in data.get("label_runs", []):
                    if run.get("judge_name") == judge_name:
                        anns = run.get("sentence_annotations", [])
                        if not any(ann.get("label_logprobs") for ann in anns):
                            needs_backfill.add(eid)
                        break
            except Exception:
                pass

    console.print(
        f"[dim]  Backfill: {judge_name}: "
        f"[bold]{len(needs_backfill)}[/bold] / {len(target_ids)} entries need logprob backfill[/dim]"
    )
    if not needs_backfill:
        return None

    filtered_order = [item for item in flat_order if item["id"] in needs_backfill]

    artifacts_dir = _HERE / "_artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    tmp_path = artifacts_dir / f"_backfill_wo_{judge_name}.json"
    tmp_path.write_text(json.dumps({
        "generated_at": wo.get("generated_at", ""),
        "strategy": "backfill_logprobs",
        "total_entries": len(filtered_order),
        "dataset_file": dataset_file,
        "file_order": wo.get("file_order", ["dataset.jsonl"]),
        "flat_order": filtered_order,
        "per_file": {"dataset.jsonl": [item["id"] for item in filtered_order]},
    }, indent=2))

    labeler_cfg.pipeline.work_order_file = str(tmp_path)
    labeler_cfg.pipeline.overwrite_existing = True
    return tmp_path


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


# ── Prompt preview ────────────────────────────────────────────────────────────

def _show_prompt(
    input_files: list[Path],
    cfg: LabelerConfig,
    config_path: Path,
) -> None:
    """Format and print the prompt for the first valid entry without calling the LLM."""
    from thesis_schema import ConversationEntry

    prompt_path = cfg.resolve_path(config_path, cfg.prompt_file)
    prompt_template = inject_taxonomy(prompt_path.read_text(encoding="utf-8"))

    system_prompt_text: str | None = None
    if cfg.system_prompt_file:
        sp_path = cfg.resolve_path(config_path, cfg.system_prompt_file)
        if sp_path.exists():
            system_prompt_text = inject_taxonomy(sp_path.read_text(encoding="utf-8"))
            system_prompt_text = system_prompt_text.replace(
                "{response_sentences}", str(cfg.pipeline.response_sentences)
            )

    # Collect all valid entries (with reasoning) across all files, then pick one at random
    candidates: list[tuple[ConversationEntry, Path]] = []
    for fpath in input_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = ConversationEntry.model_validate_json(line)
                    if any(m["role"] == "reasoning" for m in candidate.messages):
                        candidates.append((candidate, fpath))
                except Exception:
                    continue

    if not candidates:
        console.print("[red]No entry with reasoning found in the provided files.[/red]")
        return

    entry, source_file = random.choice(candidates)

    console.print(f"[dim]Showing prompt for entry [bold]{entry.id}[/bold] from {source_file.name}[/dim]")

    reasoning_text = next(m["content"] for m in entry.messages if m["role"] == "reasoning")
    original_len = len(reasoning_text)
    reasoning_truncated = False
    if cfg.pipeline.max_reasoning_chars is not None:
        reasoning_text = _crop_reasoning(reasoning_text, cfg.pipeline.max_reasoning_chars)
        reasoning_truncated = len(reasoning_text) < original_len

    user_prompt = next((m["content"] for m in entry.messages if m["role"] == "user"), "")
    response_text = next((m["content"] for m in entry.messages if m["role"] == "assistant"), "")

    is_unsafe = entry.metadata.get("prompt_safety") == "unsafe"
    user_message, effective_system = build_messages(
        prompt_template, user_prompt, reasoning_text, response_text,
        cfg.pipeline.response_sentences, reasoning_truncated=reasoning_truncated,
        system_prompt=system_prompt_text, is_unsafe=is_unsafe,
    )

    if effective_system:
        console.rule("[bold cyan]SYSTEM PROMPT[/bold cyan]")
        console.print(effective_system)
    console.rule("[bold cyan]USER PROMPT[/bold cyan]")
    console.print(user_message)
    console.rule()
    if reasoning_truncated:
        console.print(f"[yellow]Reasoning was truncated from {original_len} → {len(reasoning_text)} chars[/yellow]")


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

        run_pipeline(
            input_files=input_files,
            config=cfg,
            config_path=config_path,
            console=console,
            reset_checkpoints=reset_checkpoints,
            dry_run=dry_run,
            judge_cfg=judge,
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
@click.option(
    "--token-budget",
    default=None, type=int,
    help="Override daily token limit from config (e.g. 2500000). "
         "Run stops when the limit is reached; resume tomorrow to continue.",
)
@click.option(
    "--max-entries",
    default=None, type=int,
    help="Stop after labeling this many entries per judge (across all files). "
         "Useful for test runs (e.g. --max-entries 1).",
)
@click.option(
    "--show-prompt",
    is_flag=True, default=False,
    help="Print the formatted prompt for the first valid entry and exit (no LLM call).",
)
@click.option(
    "--work-order", "work_order",
    default=None,
    help="Override pipeline.work_order_file for all judges in this run.",
)
@click.option(
    "--pass", "pass_number",
    default=1, type=int,
    help="Labeling pass number. Pass >1 appends '_pass<N>' to all judge names so "
         "labels are stored separately, enabling multiple independent passes over the same work order.",
)
@click.option(
    "--backfill-logprobs",
    is_flag=True, default=False,
    help="Re-label only entries that already have a run for this judge but are missing logprobs.",
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
    token_budget: int | None,
    max_entries: int | None,
    show_prompt: bool,
    work_order: str | None,
    pass_number: int,
    backfill_logprobs: bool,
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

    # Resolve input files
    input_files = _resolve_files(files, glob_patterns)

    if not input_files and not dry_run and not show_prompt:
        console.print(
            "[yellow]No input files found. "
            "Pass .jsonl paths as arguments or use --glob.[/yellow]"
        )
        raise SystemExit(0)

    # --show-prompt: preview the formatted prompt for the first entry and exit
    if show_prompt:
        cfg = LabelerConfig.from_yaml(config_path)
        _show_prompt(input_files, cfg, config_path)
        raise SystemExit(0)

    # ── Registry file (labelers.yaml) ────────────────────────────────────────
    if LabelerRegistry.is_registry_file(config_path):
        registry = LabelerRegistry.from_yaml(config_path)

        # When --judge is given, look up even disabled entries so the user can
        # explicitly run any labeler regardless of its enabled flag.
        if judge_filter:
            try:
                labeler_cfgs = [(name, registry.get_config(name)) for name in judge_filter]
            except KeyError as e:
                all_names = [e.get("judge", {}).get("name") for e in registry._entries]
                console.print(f"[red]{e}[/red]")
                console.print(f"Available: {[n for n in all_names if n]}")
                raise SystemExit(1)
        else:
            labeler_cfgs = registry.all_configs(enabled_only=True)

        if not labeler_cfgs:
            console.print("[yellow]No enabled labelers found in registry.[/yellow]")
            raise SystemExit(0)

        console.print(
            f"[bold]run.py[/bold] — "
            f"config: [dim]{config_path.name}[/dim] (registry) — "
            f"judges: [cyan]{', '.join(n for n, _ in labeler_cfgs)}[/cyan] — "
            f"{len(input_files)} file(s)"
        )
        console.print()

        for judge_name, labeler_cfg in labeler_cfgs:
            # Apply CLI overrides to each labeler's config independently
            if output_dir is not None:
                labeler_cfg.output.dir = output_dir
            if workers is not None:
                labeler_cfg.pipeline.max_workers = workers
            if retries is not None:
                labeler_cfg.pipeline.max_retries = retries
            if token_budget is not None:
                labeler_cfg.pipeline.token_budget = token_budget
            if max_entries is not None:
                labeler_cfg.pipeline.max_entries = max_entries
            if work_order is not None:
                labeler_cfg.pipeline.work_order_file = work_order
            if pass_number > 1:
                for j in labeler_cfg.judges:
                    j.name = f"{j.name}_pass{pass_number}"
                # Ensure checkpoint files are also pass-specific
                if not labeler_cfg.output.suffix.endswith(f"_pass{pass_number}"):
                    labeler_cfg.output.suffix += f"_pass{pass_number}"

            judge = labeler_cfg.judges[0]
            console.rule(f"[bold blue]Judge: {judge.name}[/bold blue]")

            if judge.lmstudio_prompt and not dry_run:
                console.print(
                    f"\n[bold yellow]>>> LMStudio model required:[/bold yellow] "
                    f"[cyan]{judge.name}[/cyan]\n"
                    f"    Load/swap to this model in LMStudio then press Enter.",
                    highlight=False,
                )
                input()

            tmp_wo_path: Path | None = None
            try:
                if backfill_logprobs and not dry_run:
                    tmp_wo_path = _prepare_backfill_work_order(labeler_cfg, config_path, judge.name)
                    if tmp_wo_path is None:
                        console.print()
                        continue

                run_pipeline(
                    input_files=input_files,
                    config=labeler_cfg,
                    config_path=config_path,
                    console=console,
                    reset_checkpoints=reset_checkpoints,
                    dry_run=dry_run,
                    judge_cfg=judge,
                )
                console.print()
            finally:
                if tmp_wo_path is not None and tmp_wo_path.exists():
                    tmp_wo_path.unlink()
        return

    # ── Standard single/multi-judge config ───────────────────────────────────
    cfg = LabelerConfig.from_yaml(config_path)

    # Apply CLI overrides
    if output_dir is not None:
        cfg.output.dir = output_dir
    if workers is not None:
        cfg.pipeline.max_workers = workers
    if retries is not None:
        cfg.pipeline.max_retries = retries
    if token_budget is not None:
        cfg.pipeline.token_budget = token_budget
    if max_entries is not None:
        cfg.pipeline.max_entries = max_entries
    if work_order is not None:
        cfg.pipeline.work_order_file = work_order
    if pass_number > 1:
        for j in cfg.judges:
            j.name = f"{j.name}_pass{pass_number}"
        # Ensure checkpoint files are also pass-specific
        if not cfg.output.suffix.endswith(f"_pass{pass_number}"):
            cfg.output.suffix += f"_pass{pass_number}"

    # Filter judges
    if judge_filter:
        active_judges = [j for j in cfg.judges if j.name in judge_filter]
        if not active_judges:
            console.print(f"[red]None of the specified judges found: {list(judge_filter)}[/red]")
            console.print(f"Available: {[j.name for j in cfg.judges]}")
            raise SystemExit(1)
    else:
        active_judges = cfg.judges

    multi = len(active_judges) > 1
    console.print(
        f"[bold]run.py[/bold] — "
        f"config: [dim]{config_path.name}[/dim] — "
        f"judges: [cyan]{', '.join(j.name for j in active_judges)}[/cyan] — "
        f"{len(input_files)} file(s)"
    )
    console.print()

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
