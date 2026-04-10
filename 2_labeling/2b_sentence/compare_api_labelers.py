#!/usr/bin/env python3
"""compare_api_labelers.py — Compare GPT and Gemini judges on a random sample of clear_harm entries.

For each normal-model clear_harm file, picks one random entry, then runs all
configured judges on it. Saves results to JSON for side-by-side comparison
in the web viewer (view_comparison.py).

Usage:
    uv run compare_api_labelers.py [--seed N] [--output-dir data/labeler_comparison]
    uv run compare_api_labelers.py --judge gpt-4.1 --judge gemini-2.5-flash
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

MAX_REASONING_CHARS = 6000
RESPONSE_SENTENCES = 3

# ── Judge definitions ──────────────────────────────────────────────────────────
# Edit model_id here to change which specific model is used.

JUDGES: list[JudgeConfig] = [
    JudgeConfig(
        name="gpt-5-mini",
        model_id="gpt-5-mini-2025-08-07",          
        backend="openai",
        base_url=None,
        api_key_source="openai-api-key",
        temperature=1.0,
        max_completion_tokens=8192,
        timeout=120,
    ),
    JudgeConfig(
        name="gpt-5.4-mini",
        model_id="gpt-5.4-mini-2026-03-17",          
        backend="openai",
        base_url=None,
        api_key_source="openai-api-key",
        temperature=1.0,
        max_completion_tokens=8192,
        timeout=120,
    ),
    JudgeConfig(
        name="gpt-5.4-nano",
        model_id="gpt-5.4-nano-2026-03-17",          
        backend="openai",
        base_url=None,
        api_key_source="openai-api-key",
        temperature=1.0,
        max_completion_tokens=8192,
        timeout=120,
    ),
    JudgeConfig(
        name="gpt-5",
        model_id="gpt-5-2025-08-07",          
        backend="openai",
        base_url=None,
        api_key_source="openai-api-key",
        temperature=1.0,
        max_completion_tokens=8192,
        timeout=120,
    ),
    JudgeConfig(
        name="gemini-3-flash-preview",
        model_id="gemini-3-flash-preview", 
        backend="gemini",
        base_url=None,
        api_key_source="gemini-api-key",
        temperature=0.4,
        max_tokens=8192,
        timeout=300,
        service_tier="flex",
        reasoning_effort="low",
    ),
    JudgeConfig(
        name="gemini-3.1-flash-lite-preview",
        model_id="gemini-3.1-flash-lite-preview",
        backend="gemini",
        base_url=None,
        api_key_source="gemini-api-key",
        temperature=0.4,
        max_tokens=8192,
        timeout=300,
        service_tier="flex",
        reasoning_effort="low",
    ),
    JudgeConfig(
        name="gemini-3.1-pro-preview",
        model_id="gemini-3.1-pro-preview",
        backend="gemini",
        base_url=None,
        api_key_source="gemini-api-key",
        temperature=0.4,
        max_tokens=8192,
        timeout=300,
        service_tier="flex",
        reasoning_effort="low",
    ),
]


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

    entries_meta = []
    for path in normal_files:
        lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            console.print(f"[yellow]Skipping empty file: {path.name}[/yellow]")
            continue

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
                "text": s.text,  # full text for viewer
                "labels": s.labels,
                "safety_score": s.safety_score,
            }
            for s in output.sentences
        ],
        "metadata": output.metadata,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--seed", default=12345, show_default=True, help="Random seed for entry selection.")
@click.option(
    "--output-dir", "-o",
    default="data/labeler_comparison",
    show_default=True,  
    help="Directory to save JSON results (relative to this script).",
)
@click.option(
    "--judge", "-j", "judge_filter",
    multiple=True,
    help="Name of judge to run. Can be repeated. Defaults to all judges.",
)
def main(seed: int, output_dir: str, judge_filter: tuple[str, ...]) -> None:
    """Compare API judge LLMs (GPT, Gemini) on one random clear_harm entry per model file."""

    output_path = (_HERE / output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    if not PROMPT_PATH.exists():
        console.print(f"[red]Prompt file not found: {PROMPT_PATH}[/red]")
        sys.exit(1)
    prompt_template = PROMPT_PATH.read_text()

    try:
        secrets = load_secrets(SECRETS_PATH)
    except FileNotFoundError:
        console.print(f"[yellow]Secrets file not found at {SECRETS_PATH}. Will rely on env vars.[/yellow]")
        secrets = {}

    if judge_filter:
        active_judges = [j for j in JUDGES if j.name in judge_filter]
        if not active_judges:
            console.print(f"[red]None of the specified judges found: {judge_filter}[/red]")
            console.print(f"Available: {[j.name for j in JUDGES]}")
            sys.exit(1)
    else:
        active_judges = JUDGES

    console.print(f"[bold]Loading random entries (seed={seed})...[/bold]")
    entries_meta = load_random_entries(seed)
    console.print(f"  Selected [green]{len(entries_meta)}[/green] entries\n")

    for em in entries_meta:
        console.print(f"  [cyan]{em['source_file']}[/cyan] → [dim]{em['entry'].id[:16]}[/dim]")
    console.print()

    results = {
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "entries": [
            {
                "source_file": em["source_file"],
                "entry_id": em["entry"].id,
                "user_prompt": _find_msg(em["entry"], "user") or "",
                "reasoning": _crop_reasoning(_find_msg(em["entry"], "reasoning") or "", MAX_REASONING_CHARS),
                "model_response": _find_msg(em["entry"], "assistant") or "",
                "labels_by_judge": {},
            }
            for em in entries_meta
        ],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_path / f"comparison_seed{seed}_{timestamp}.json"

    def save_results():
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    try:
        for judge_idx, cfg in enumerate(active_judges):
            console.rule(f"[bold blue]Judge {judge_idx + 1}/{len(active_judges)}: {cfg.name}[/bold blue]")

            try:
                client = build_client(cfg, secrets)
            except ValueError as e:
                console.print(f"[red]Could not build client for {cfg.name}: {e}[/red]")
                for res in results["entries"]:
                    res["labels_by_judge"][cfg.name] = {"error": str(e)}
                save_results()
                continue

            for entry_idx, em in enumerate(entries_meta):
                entry = em["entry"]
                console.print(
                    f"  [{entry_idx + 1}/{len(entries_meta)}] "
                    f"[dim]{em['source_file']}[/dim] → [dim]{entry.id[:16]}[/dim] ... ",
                    end="",
                )

                output = label_with_retry(client, cfg, prompt_template, entry)
                results["entries"][entry_idx]["labels_by_judge"][cfg.name] = output_as_dict(output)
                save_results()  # Save after every single response

                if output:
                    trajectory = output.assessment.trajectory
                    alignment = output.assessment.alignment
                    n_sentences = len(output.sentences)
                    usage = output.metadata.get("usage", {})
                    total_tokens = usage.get("total_tokens", "?")
                    reason = output.metadata.get("finish_reason", "unknown")
                    console.print(
                        f"[green]OK[/green] "
                        f"[dim]{trajectory} / {alignment} / {n_sentences} sents / {total_tokens} tokens ({reason})[/dim]"
                    )
                else:
                    console.print("[red]FAILED[/red]")

            console.print()

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted! Partial results saved to disk.[/yellow]")
    finally:
        console.print(f"[bold green]Results saved to:[/bold green] {out_file}\n")
        _print_summary(results, active_judges)
        console.print(f"\nOpen in viewer: [bold]uv run view_comparison.py {out_file}[/bold]")


def _print_summary(results: dict, active_judges: list[JudgeConfig]) -> None:
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
