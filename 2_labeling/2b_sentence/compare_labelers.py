#!/usr/bin/env python3
"""compare_labelers.py — Compare multiple judge LLMs on a random sample of clear_harm entries.

For each normal-model clear_harm file, picks one random entry, then runs all
configured judges on it sequentially. Saves results to JSON for side-by-side comparison.

LMStudio judges require you to manually swap the loaded model between runs.
The script will pause and prompt you before each LMStudio judge.

Usage:
    uv run compare_labelers.py [--seed N] [--output-dir results/]
"""

from __future__ import annotations

import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from sentence_labeler.labeler import build_client, call_llm, format_prompt, load_secrets, parse_llm_output
from sentence_labeler.schema import JudgeConfig, LabelingOutput
from thesis_schema import ConversationEntry

console = Console()

# ── Paths ──────────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
DATA_DIR = _HERE.parent.parent.parent / "1_generating" / "data"
SECRETS_PATH = _HERE.parent.parent / ".secrets.json"
PROMPT_PATH = _HERE / "prompt.md"

MAX_REASONING_CHARS = 8000  # crop long reasonings (same as lmstudio.yaml)
RESPONSE_SENTENCES = 3

# ── Judge definitions ──────────────────────────────────────────────────────────

_LMSTUDIO_BASE = "http://localhost:1234/v1"

JUDGES: list[JudgeConfig] = [
    JudgeConfig(
        name="gpt-oss-20b",
        model_id="gpt-oss-20b",
        backend="custom",
        base_url=_LMSTUDIO_BASE,
        api_key_source="lm-studio-key",
        temperature=0.4,
        max_tokens=20000,
        timeout=300,
    ),
    JudgeConfig(
        name="google/gemma-4-26b-a4b",
        model_id="google/gemma-4-26b-a4b",
        backend="custom",
        base_url=_LMSTUDIO_BASE,
        api_key_source="lm-studio-key",
        temperature=0.4,
        max_tokens=20000,
        timeout=300,
    ),
    JudgeConfig(
        name="qwen3.5-27b-uncensored-hauhaucs-aggressive",
        model_id="qwen3.5-27b-uncensored-hauhaucs-aggressive",
        backend="custom",
        base_url=_LMSTUDIO_BASE,
        api_key_source="lm-studio-key",
        temperature=0.4,
        max_tokens=20000,
        timeout=300,
    ),
    JudgeConfig(
        name="gpt-5.4-mini-2026-03-17",
        model_id="gpt-5.4-mini-2026-03-17",
        backend="openai",
        base_url=None,
        api_key_source="openai-api-key",
        temperature=1.0,
        max_completion_tokens=8192,
        timeout=120,
    ),
]

_LMSTUDIO_JUDGES = {j.name for j in JUDGES if j.base_url == _LMSTUDIO_BASE}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_msg(entry: ConversationEntry, role: str) -> str | None:
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


def load_random_entries(seed: int) -> list[dict]:
    """Pick one random entry per normal-model clear_harm file."""
    rng = random.Random(seed)
    entries_meta = []

    EXCLUDED_MODELS = {"qwen3.5-4B"}

    files = sorted(DATA_DIR.glob("*_clear_harm.jsonl"))
    normal_files = [
        f for f in files
        if "heretic" not in f.name
        and not any(f.name.startswith(ex) for ex in EXCLUDED_MODELS)
    ]

    if not normal_files:
        console.print(f"[red]No clear_harm files found in {DATA_DIR}[/red]")
        sys.exit(1)

    for path in normal_files:
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            console.print(f"[yellow]Skipping empty file: {path.name}[/yellow]")
            continue

        # Filter to entries that have a reasoning message
        valid_lines = []
        for line in lines:
            try:
                e = ConversationEntry.model_validate_json(line)
                if any(m["role"] == "reasoning" for m in e.messages):
                    valid_lines.append((line, e))
            except Exception:
                pass

        if not valid_lines:
            console.print(f"[yellow]No valid reasoning entries in: {path.name}[/yellow]")
            continue

        raw_line, entry = rng.choice(valid_lines)
        entries_meta.append({
            "source_file": path.name,
            "entry": entry,
        })

    return entries_meta


def label_with_retry(
    client,
    cfg: JudgeConfig,
    prompt_template: str,
    entry: ConversationEntry,
    max_retries: int = 3,
    retry_delay: float = 3.0,
) -> LabelingOutput | None:
    """Call the judge with retries. Returns None on total failure."""
    import time

    user_prompt = _find_msg(entry, "user") or ""
    model_response = _find_msg(entry, "assistant") or ""
    reasoning_raw = _find_msg(entry, "reasoning") or ""
    reasoning = _crop_reasoning(reasoning_raw, MAX_REASONING_CHARS)

    prompt = format_prompt(
        template=prompt_template,
        user_prompt=user_prompt,
        model_reasoning=reasoning,
        model_response=model_response,
        response_sentences=RESPONSE_SENTENCES,
    )

    last_exc = None
    for attempt in range(max_retries):
        try:
            raw, metadata = call_llm(client, cfg, prompt)
            output = parse_llm_output(raw)
            output.metadata = metadata
            return output
        except Exception as e:
            last_exc = e
            wait = retry_delay * (2 ** attempt)
            console.print(
                f"  [yellow]Attempt {attempt + 1}/{max_retries} failed: {e}. "
                f"Retrying in {wait:.0f}s...[/yellow]"
            )
            time.sleep(wait)

    console.print(f"  [red]All {max_retries} attempts failed. Last error: {last_exc}[/red]")
    return None


def output_as_dict(output: LabelingOutput | None) -> dict:
    if output is None:
        return {"error": "labeling_failed"}
    return {
        "assessment": output.assessment.model_dump(),
        "sentences": [
            {
                "text": s.text[:120],  # truncate for readability in output
                "labels": s.labels,
                "safety_score": s.safety_score,
            }
            for s in output.sentences
        ],
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--seed", default=42, show_default=True, help="Random seed for entry selection.")
@click.option(
    "--output-dir", "-o",
    default="data/labeler_comparison",
    show_default=True,
    help="Directory to save JSON results (relative to this script).",
)
@click.option(
    "--skip-lmstudio-prompt",
    is_flag=True,
    default=False,
    help="Don't pause between LMStudio models (if you know what you're doing).",
)
@click.option(
    "--judge", "-j", "judge_filter",
    multiple=True,
    help="Name of judge to run. Can be used multiple times. If not provided, runs all configured judges.",
)
def main(seed: int, output_dir: str, skip_lmstudio_prompt: bool, judge_filter: tuple[str, ...]) -> None:
    """Compare judge LLMs on one random clear_harm entry per model file."""

    output_path = (_HERE / output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Load prompt template
    if not PROMPT_PATH.exists():
        console.print(f"[red]Prompt file not found: {PROMPT_PATH}[/red]")
        sys.exit(1)
    prompt_template = PROMPT_PATH.read_text()

    # Load secrets
    try:
        secrets = load_secrets(SECRETS_PATH)
    except FileNotFoundError:
        console.print(f"[yellow]Secrets file not found at {SECRETS_PATH}. Will rely on environment variables.[/yellow]")
        secrets = {}

    # Filter judges
    if judge_filter:
        active_judges = [j for j in JUDGES if j.name in judge_filter]
        if not active_judges:
            console.print(f"[red]None of the specified judges found: {judge_filter}[/red]")
            console.print(f"Available judges: {[j.name for j in JUDGES]}")
            sys.exit(1)
    else:
        active_judges = JUDGES

    # Pick entries
    console.print(f"[bold]Loading random entries (seed={seed})...[/bold]")
    entries_meta = load_random_entries(seed)
    console.print(f"  Selected [green]{len(entries_meta)}[/green] entries from {len(entries_meta)} files\n")

    for em in entries_meta:
        console.print(f"  [cyan]{em['source_file']}[/cyan] → entry [dim]{em['entry'].id[:16]}[/dim]")
    console.print()

    # Build results structure
    results = {
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": [
            {
                "source_file": em["source_file"],
                "entry_id": em["entry"].id,
                "user_prompt": (_find_msg(em["entry"], "user") or "")[:300],
                "labels_by_judge": {},
            }
            for em in entries_meta
        ],
    }

    # Run each judge sequentially
    prev_was_lmstudio = False

    for judge_idx, cfg in enumerate(active_judges):
        is_lmstudio = cfg.name in _LMSTUDIO_JUDGES

        console.rule(f"[bold blue]Judge {judge_idx + 1}/{len(active_judges)}: {cfg.name}[/bold blue]")

        # Pause between LMStudio judges so user can swap the model
        if is_lmstudio and not skip_lmstudio_prompt:
            if judge_idx == 0 or prev_was_lmstudio:
                console.print(
                    f"\n[bold yellow]>>> LMStudio model required:[/bold yellow] "
                    f"[cyan]{cfg.name}[/cyan]\n"
                    f"    Load this model in LMStudio (default port 1234) then press Enter.",
                    highlight=False,
                )
                input()
            else:
                console.print(
                    f"\n[bold yellow]>>> LMStudio model required:[/bold yellow] "
                    f"[cyan]{cfg.name}[/cyan]\n"
                    f"    Swap to this model in LMStudio then press Enter.",
                    highlight=False,
                )
                input()

        try:
            client = build_client(cfg, secrets)
        except ValueError as e:
            console.print(f"[red]Could not build client for {cfg.name}: {e}[/red]")
            for res in results["entries"]:
                res["labels_by_judge"][cfg.name] = {"error": str(e)}
            prev_was_lmstudio = is_lmstudio
            continue

        for entry_idx, em in enumerate(entries_meta):
            entry = em["entry"]
            console.print(
                f"  [{entry_idx + 1}/{len(entries_meta)}] "
                f"[dim]{em['source_file']}[/dim] → entry [dim]{entry.id[:16]}[/dim] ... ",
                end="",
            )

            output = label_with_retry(client, cfg, prompt_template, entry)
            results["entries"][entry_idx]["labels_by_judge"][cfg.name] = output_as_dict(output)

            if output:
                trajectory = output.assessment.trajectory
                alignment = output.assessment.alignment
                n_sentences = len(output.sentences)
                console.print(
                    f"[green]OK[/green] "
                    f"[dim]{trajectory} / {alignment} / {n_sentences} sentences[/dim]"
                )
            else:
                console.print("[red]FAILED[/red]")

        prev_was_lmstudio = is_lmstudio
        console.print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_path / f"comparison_seed{seed}_{timestamp}.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    console.print(f"[bold green]Results saved to:[/bold green] {out_file}\n")

    # Summary table
    _print_summary(results, active_judges)


def _print_summary(results: dict, active_judges: list[JudgeConfig]) -> None:
    """Print a concise summary table: entry × judge → trajectory/alignment."""
    judge_names = [cfg.name for cfg in active_judges]

    table = Table(title="Labeler Comparison Summary", show_lines=True)
    table.add_column("Source model", style="cyan", no_wrap=True)
    table.add_column("Entry ID", style="dim", no_wrap=True)
    for name in judge_names:
        table.add_column(name, overflow="fold")

    for entry_res in results["entries"]:
        row = [
            entry_res["source_file"].replace("_clear_harm.jsonl", ""),
            entry_res["entry_id"][:12],
        ]
        for name in judge_names:
            labels = entry_res["labels_by_judge"].get(name, {})
            if "error" in labels:
                row.append(f"[red]{labels['error']}[/red]")
            else:
                assessment = labels.get("assessment", {})
                traj = assessment.get("trajectory", "?")
                align = assessment.get("alignment", "?")
                n = len(labels.get("sentences", []))
                row.append(f"{traj}\n{align}\n{n} sents")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()
