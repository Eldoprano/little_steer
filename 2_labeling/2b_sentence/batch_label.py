#!/usr/bin/env python3
"""batch_label.py — Prepare and submit batch labeling jobs (50% cost discount).

Supports Anthropic Message Batches, OpenAI Batch API, and Google Gemini Batch API.
Results are not returned synchronously — use batch_poll.py to check status and
download completed results into dataset.jsonl.

Usage:
    uv run batch_label.py                        # submit all enabled labelers
    uv run batch_label.py --judge claude-haiku-4-5
    uv run batch_label.py --dry-run              # cost estimate only, no submission
    uv run batch_label.py --test                 # submit exactly 1 entry as a test
    uv run batch_label.py -n 50                  # override n_samples from config
    uv run batch_label.py --config other.yaml    # use a different config file
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from sentence_labeler.labeler import format_prompt, load_secrets
from sentence_labeler.taxonomy_loader import get_taxonomy_version, inject_taxonomy
from thesis_schema import ConversationEntry

DATASET_FILE = (HERE / "../../data/dataset.jsonl").resolve()
WORK_ORDER_FILE = HERE / "work_order.json"
BATCHES_DIR = HERE / "_artifacts" / "batches"
BATCHES_DIR.mkdir(parents=True, exist_ok=True)

MAX_REASONING_CHARS = 8000
RESPONSE_SENTENCES = 10


# ── Config loading ─────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        elif v is not None:
            result[k] = v
    return result


def load_batch_configs(
    config_path: Path, judge_filter: list[str] | None = None
) -> list[dict]:
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    defaults = raw.get("defaults", {})
    configs = []
    for entry in raw.get("labelers", []):
        if not entry.get("enabled", True):
            continue
        merged = _deep_merge(defaults, entry)
        name = (merged.get("judge") or {}).get("name", "unknown")
        if judge_filter and name not in judge_filter:
            continue
        configs.append(merged)
    return configs


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_labeled_ids_for_judge(judge_name: str) -> set[str]:
    """Return IDs already labeled by this judge in dataset.jsonl."""
    labeled: set[str] = set()
    if not DATASET_FILE.exists():
        return labeled
    with open(DATASET_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                for r in data.get("label_runs") or []:
                    if r.get("judge_name") == judge_name:
                        labeled.add(data["id"])
                        break
            except Exception:
                pass
    return labeled


def load_entries_by_id(ids: list[str]) -> dict[str, ConversationEntry]:
    entries: dict[str, ConversationEntry] = {}
    if not DATASET_FILE.exists():
        return entries
    needed = set(ids)
    with open(DATASET_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                eid = data.get("id")
                if eid in needed:
                    entries[eid] = ConversationEntry.model_validate(data)
            except Exception:
                pass
    return entries


def pick_entry_ids(judge_name: str, n_samples: int) -> list[str]:
    """Return up to n_samples IDs from work_order.json not yet labeled by this judge."""
    if not WORK_ORDER_FILE.exists():
        raise FileNotFoundError(
            f"work_order.json not found. Run 'uv run run_all.py' once to generate it."
        )
    with open(WORK_ORDER_FILE) as f:
        wo = json.load(f)
    all_ids = [item["id"] for item in wo.get("flat_order", [])]
    labeled = load_labeled_ids_for_judge(judge_name)
    pending = [eid for eid in all_ids if eid not in labeled]
    return pending[:n_samples]


# ── Prompt building ────────────────────────────────────────────────────────────

def _find_reasoning_msg_idx(entry: ConversationEntry) -> int | None:
    for i, msg in enumerate(entry.messages):
        if msg["role"] == "reasoning":
            return i
    return None


def _find_user_msg(entry: ConversationEntry) -> str:
    for msg in entry.messages:
        if msg["role"] == "user":
            return msg["content"]
    return ""


def _find_assistant_msg(entry: ConversationEntry) -> str:
    for msg in entry.messages:
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


def build_prompts(
    entries: list[ConversationEntry],
    prompt_template: str,
    system_prompt_text: str,
) -> list[tuple[str, str, bool] | None]:
    """Return one (user_prompt, system_prompt, reasoning_truncated) per entry, or None if skipped."""
    results: list[tuple[str, str, bool] | None] = []
    for entry in entries:
        ridx = _find_reasoning_msg_idx(entry)
        if ridx is None:
            results.append(None)
            continue
        reasoning = entry.messages[ridx]["content"]
        truncated = len(reasoning) > MAX_REASONING_CHARS
        if truncated:
            reasoning = reasoning[:MAX_REASONING_CHARS]
        user_prompt = _find_user_msg(entry)
        response = _find_assistant_msg(entry)
        formatted = format_prompt(
            prompt_template,
            user_prompt,
            reasoning,
            response,
            response_sentences=RESPONSE_SENTENCES,
            reasoning_truncated=truncated,
        )
        results.append((formatted, system_prompt_text, truncated))
    return results


# ── Cost estimation ────────────────────────────────────────────────────────────

def estimate_cost(
    prompts: list[tuple | None],
    price_per_1m_input: float,
    price_per_1m_output: float,
    batch_discount: float,
    output_ratio: float = 1.6,
) -> dict[str, Any]:
    valid = [p for p in prompts if p is not None]
    total_input_tokens = sum((len(p[0]) + len(p[1])) // 4 for p in valid)
    total_output_tokens = int(total_input_tokens * output_ratio)
    standard_cost = (
        total_input_tokens / 1_000_000 * price_per_1m_input
        + total_output_tokens / 1_000_000 * price_per_1m_output
    )
    batch_cost = standard_cost * (1.0 - batch_discount)
    return {
        "n_entries": len(valid),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "standard_cost_usd": standard_cost,
        "batch_cost_usd": batch_cost,
        "savings_usd": standard_cost - batch_cost,
    }


# ── API key resolution ─────────────────────────────────────────────────────────

def _get_api_key(source: str, secrets: dict) -> str:
    key = os.environ.get(source, "") or secrets.get(source, "")
    if not key:
        raise ValueError(
            f"API key '{source}' not found in environment variables or secrets file. "
            f"Available keys: {list(secrets.keys())}"
        )
    return key


# ── Provider submission ────────────────────────────────────────────────────────

def submit_anthropic(
    cfg: dict,
    entries: list[ConversationEntry],
    prompts: list[tuple | None],
    secrets: dict,
) -> dict:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: uv add anthropic")

    judge = cfg["judge"]
    api_key = _get_api_key(judge.get("api_key_source", "anthropic-api-key"), secrets)
    client = anthropic.Anthropic(api_key=api_key)

    requests = []
    entry_ids = []
    for entry, prompt_data in zip(entries, prompts):
        if prompt_data is None:
            continue
        user_msg, system_msg, _ = prompt_data
        requests.append({
            "custom_id": entry.id,
            "params": {
                "model": judge["model_id"],
                "max_tokens": judge.get("max_tokens", 8192),
                "temperature": judge.get("temperature", 0.2),
                "system": system_msg,
                "messages": [{"role": "user", "content": user_msg}],
            },
        })
        entry_ids.append(entry.id)

    batch = client.messages.batches.create(requests=requests)
    return {
        "batch_id": batch.id,
        "provider": "anthropic",
        "status": "in_progress",
        "entry_ids": entry_ids,
        "provider_meta": {"processing_status": batch.processing_status},
    }


def submit_openai(
    cfg: dict,
    entries: list[ConversationEntry],
    prompts: list[tuple | None],
    secrets: dict,
) -> dict:
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required: uv add openai")

    judge = cfg["judge"]
    api_key = _get_api_key(judge.get("api_key_source", "openai-api-key"), secrets)
    client = openai.OpenAI(api_key=api_key)

    lines = []
    entry_ids = []
    for entry, prompt_data in zip(entries, prompts):
        if prompt_data is None:
            continue
        user_msg, system_msg, _ = prompt_data
        lines.append(json.dumps({
            "custom_id": entry.id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": judge["model_id"],
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": judge.get("max_tokens", 8192),
                "temperature": judge.get("temperature", 0.2),
            },
        }))
        entry_ids.append(entry.id)

    jsonl_bytes = "\n".join(lines).encode()
    file_obj = client.files.create(
        file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    return {
        "batch_id": batch.id,
        "provider": "openai",
        "status": "in_progress",
        "entry_ids": entry_ids,
        "provider_meta": {"input_file_id": file_obj.id},
    }


def submit_gemini(
    cfg: dict,
    entries: list[ConversationEntry],
    prompts: list[tuple | None],
    secrets: dict,
) -> dict:
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai package required: uv add google-genai")

    judge = cfg["judge"]
    api_key = _get_api_key(judge.get("api_key_source", "gemini-api-key"), secrets)
    client = genai.Client(api_key=api_key)

    requests_list = []
    entry_ids = []
    for entry, prompt_data in zip(entries, prompts):
        if prompt_data is None:
            continue
        user_msg, system_msg, _ = prompt_data
        requests_list.append({
            "contents": [{"parts": [{"text": user_msg}], "role": "user"}],
            "system_instruction": {"parts": [{"text": system_msg}]},
            "generation_config": {
                "max_output_tokens": judge.get("max_tokens", 8192),
                "temperature": judge.get("temperature", 0.2),
            },
        })
        entry_ids.append(entry.id)

    # Gemini inline batch preserves order; entry_ids in tracking file maps position → id
    batch_job = client.batches.create(model=judge["model_id"], src=requests_list)
    return {
        "batch_id": batch_job.name,
        "provider": "gemini",
        "status": "in_progress",
        "entry_ids": entry_ids,
        "provider_meta": {"state": str(batch_job.state)},
    }


SUBMIT_FNS = {
    "anthropic": submit_anthropic,
    "openai": submit_openai,
    "gemini": submit_gemini,
}


# ── Tracking file ──────────────────────────────────────────────────────────────

def write_tracking_file(
    batch_result: dict,
    cfg: dict,
    taxonomy_version: str,
) -> Path:
    judge = cfg["judge"]
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    provider = batch_result["provider"]
    # Sanitize batch_id for use in filename (Gemini uses slash-separated paths)
    batch_id_short = batch_result["batch_id"].replace("/", "_")[-30:]
    fname = f"{provider}_{batch_id_short}_{ts}.json"

    tracking = {
        "batch_id": batch_result["batch_id"],
        "provider": provider,
        "judge_name": judge["name"],
        "judge_model_id": judge["model_id"],
        "taxonomy_version": taxonomy_version,
        "submitted_at": now.isoformat(),
        "status": batch_result["status"],
        "n_entries": len(batch_result["entry_ids"]),
        "entry_ids": batch_result["entry_ids"],
        "config_snapshot": {
            "judge": judge,
            "batch": cfg.get("batch", {}),
        },
        "results_file": None,
        "provider_meta": batch_result.get("provider_meta", {}),
    }
    path = BATCHES_DIR / fname
    path.write_text(json.dumps(tracking, indent=2))
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit batch labeling jobs for 50% cost discount."
    )
    parser.add_argument(
        "--judge", action="append", dest="judges", metavar="NAME",
        help="Run only this judge (repeatable)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show cost estimate only; don't submit",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Submit exactly 1 entry (to verify the pipeline works)",
    )
    parser.add_argument(
        "-n", "--n", type=int, metavar="N",
        help="Override n_samples from config",
    )
    parser.add_argument(
        "--config", default="batch_labelers.yaml",
        help="Config file (default: batch_labelers.yaml)",
    )
    args = parser.parse_args()

    console = Console()
    config_path = HERE / args.config

    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        sys.exit(1)

    configs = load_batch_configs(config_path, judge_filter=args.judges)
    if not configs:
        console.print("[red]No enabled labelers found.[/red]")
        sys.exit(1)

    for cfg in configs:
        judge = cfg["judge"]
        batch_cfg = cfg.get("batch", {})
        out_cfg = cfg.get("output", {})
        name = judge["name"]
        provider = judge.get("provider", "anthropic")
        taxonomy_version = out_cfg.get("taxonomy_version") or get_taxonomy_version()

        n_samples = args.n or batch_cfg.get("n_samples", 200)
        if args.test:
            n_samples = 1

        console.rule(f"[cyan]{name}[/cyan]  ([dim]{provider}[/dim])")

        # Load secrets
        secrets_path = Path(cfg.get("secrets_file", "../../.secrets.json"))
        if not secrets_path.is_absolute():
            secrets_path = (HERE / secrets_path).resolve()
        try:
            secrets = load_secrets(secrets_path)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            continue

        # Load prompts
        prompt_file = HERE / cfg.get("prompt_file", "prompt.md")
        system_file = HERE / cfg.get("system_prompt_file", "system_prompt.md")
        if not prompt_file.exists():
            console.print(f"[red]Prompt file not found: {prompt_file}[/red]")
            continue
        prompt_template = inject_taxonomy(prompt_file.read_text())
        system_prompt_text = system_file.read_text() if system_file.exists() else ""

        # Pick entries
        try:
            entry_ids = pick_entry_ids(name, n_samples)
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            continue

        if not entry_ids:
            console.print(f"[yellow]No pending entries for {name} — already fully labeled.[/yellow]")
            continue

        console.print(
            f"Selected [green]{len(entry_ids)}[/green] entries "
            f"not yet labeled by [cyan]{name}[/cyan]"
        )

        entries_map = load_entries_by_id(entry_ids)
        entries = [entries_map[eid] for eid in entry_ids if eid in entries_map]
        if not entries:
            console.print("[yellow]Could not load entries from dataset.jsonl.[/yellow]")
            continue

        prompts = build_prompts(entries, prompt_template, system_prompt_text)
        skipped = sum(1 for p in prompts if p is None)
        if skipped:
            console.print(f"[dim]Skipping {skipped} entries with no reasoning message.[/dim]")

        # Cost estimate
        est = estimate_cost(
            prompts,
            price_per_1m_input=batch_cfg.get("price_per_1m_input_tokens", 1.0),
            price_per_1m_output=batch_cfg.get("price_per_1m_output_tokens", 5.0),
            batch_discount=batch_cfg.get("batch_discount", 0.5),
            output_ratio=batch_cfg.get("output_ratio", 1.6),
        )

        discount_pct = int(batch_cfg.get("batch_discount", 0.5) * 100)
        table = Table(
            title=f"Batch estimate — {name} ({est['n_entries']} entries)",
            border_style="blue",
            header_style="bold white",
        )
        table.add_column("", style="dim")
        table.add_column("", justify="right")
        table.add_row("Est. input tokens", f"{est['total_input_tokens']:,}")
        table.add_row("Est. output tokens (1.6× input)", f"{est['total_output_tokens']:,}")
        table.add_section()
        table.add_row(
            f"[bold white]You pay[/bold white]  [dim](batch API, {discount_pct}% off)[/dim]",
            f"[bold green]${est['batch_cost_usd']:.4f}[/bold green]",
        )
        table.add_row(
            "[dim]vs. standard API[/dim]",
            f"[dim]${est['standard_cost_usd']:.4f}[/dim]",
        )
        console.print(table)

        if args.dry_run:
            console.print("[dim]Dry run — skipping submission.[/dim]\n")
            continue

        if not args.test:
            try:
                answer = input("Submit this batch? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer != "y":
                console.print("[dim]Skipped.[/dim]\n")
                continue

        # Submit
        submit_fn = SUBMIT_FNS.get(provider)
        if submit_fn is None:
            console.print(f"[red]Unknown provider '{provider}'. Supported: {list(SUBMIT_FNS)}[/red]")
            continue

        n_valid = est["n_entries"]
        console.print(f"Submitting [green]{n_valid}[/green] requests to [cyan]{provider}[/cyan]...")
        try:
            batch_result = submit_fn(cfg, entries, prompts, secrets)
        except Exception as e:
            console.print(f"[red]Submission failed: {e}[/red]")
            continue

        tracking_path = write_tracking_file(batch_result, cfg, taxonomy_version)
        console.print(f"[green]✓ Submitted[/green]  batch_id: [dim]{batch_result['batch_id']}[/dim]")
        console.print(f"Tracking file: {tracking_path.name}")
        console.print(f"Run [bold]uv run batch_poll.py[/bold] to check status.\n")


if __name__ == "__main__":
    main()
