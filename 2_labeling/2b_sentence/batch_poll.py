#!/usr/bin/env python3
"""batch_poll.py — Poll batch jobs, download results, and integrate into dataset.jsonl.

Usage:
    uv run batch_poll.py               # show status of all tracked batches
    uv run batch_poll.py --download    # download + integrate any completed batches
    uv run batch_poll.py --watch 60    # auto-refresh every N seconds
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from sentence_labeler.labeler import load_secrets, parse_llm_output
from sentence_labeler.pipeline import _atomic_merge_label_runs, apply_labeling_output
from sentence_labeler.schema import MapperConfig
from thesis_schema import ConversationEntry

DATASET_FILE = (HERE / "../../data/dataset.jsonl").resolve()
BATCHES_DIR = HERE / "_artifacts" / "batches"
MAX_REASONING_CHARS = 8000

DEFAULT_MAPPER_CFG = MapperConfig()
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


# ── Tracking file I/O ─────────────────────────────────────────────────────────

def load_tracking_files() -> list[tuple[Path, dict]]:
    if not BATCHES_DIR.exists():
        return []
    result = []
    for p in sorted(BATCHES_DIR.glob("*.json")):
        if "_results" in p.stem:
            continue
        try:
            data = json.loads(p.read_text())
            result.append((p, data))
        except Exception:
            pass
    return result


def save_tracking(path: Path, tracking: dict) -> None:
    path.write_text(json.dumps(tracking, indent=2))


# ── Secrets ────────────────────────────────────────────────────────────────────

def _load_secrets() -> dict:
    secrets_path = HERE / "../../.secrets.json"
    try:
        return load_secrets(secrets_path)
    except Exception:
        return {}


# ── Status polling ─────────────────────────────────────────────────────────────

def poll_anthropic(tracking: dict, secrets: dict) -> str:
    try:
        import anthropic
    except ImportError:
        return tracking["status"]

    api_key = os.environ.get("anthropic-api-key", "") or secrets.get("anthropic-api-key", "")
    if not api_key:
        return tracking["status"]
    client = anthropic.Anthropic(api_key=api_key)
    batch = client.messages.batches.retrieve(tracking["batch_id"])
    # processing_status: "in_progress" | "ended" | "canceling" | "expired"
    return "completed" if batch.processing_status == "ended" else batch.processing_status


def poll_openai(tracking: dict, secrets: dict) -> str:
    try:
        import openai
    except ImportError:
        return tracking["status"]

    api_key = (
        os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("openai-api-key", "")
        or secrets.get("openai-api-key", "")
    )
    if not api_key:
        return tracking["status"]
    client = openai.OpenAI(api_key=api_key)
    batch = client.batches.retrieve(tracking["batch_id"])
    return batch.status  # "validating"|"in_progress"|"completed"|"failed"|"expired"|...


def poll_gemini(tracking: dict, secrets: dict) -> str:
    try:
        from google import genai
    except ImportError:
        return tracking["status"]

    api_key = (
        os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("gemini-api-key", "")
        or secrets.get("gemini-api-key", "")
    )
    if not api_key:
        return tracking["status"]
    client = genai.Client(api_key=api_key)
    job = client.batches.get(name=tracking["batch_id"])
    state_map = {
        "JOB_STATE_SUCCEEDED": "completed",
        "JOB_STATE_FAILED": "failed",
        "JOB_STATE_CANCELLED": "cancelled",
        "JOB_STATE_CANCELLING": "cancelling",
        "JOB_STATE_RUNNING": "in_progress",
        "JOB_STATE_PENDING": "in_progress",
        "JOB_STATE_QUEUED": "in_progress",
    }
    return state_map.get(str(job.state), str(job.state))


POLL_FNS = {
    "anthropic": poll_anthropic,
    "openai": poll_openai,
    "gemini": poll_gemini,
}


def refresh_statuses(
    items: list[tuple[Path, dict]], secrets: dict
) -> list[tuple[Path, dict]]:
    updated = []
    for path, tracking in items:
        if tracking.get("status") in TERMINAL_STATUSES:
            updated.append((path, tracking))
            continue
        poll_fn = POLL_FNS.get(tracking.get("provider", ""))
        if poll_fn:
            try:
                new_status = poll_fn(tracking, secrets)
                if new_status != tracking.get("status"):
                    tracking = dict(tracking)
                    tracking["status"] = new_status
                    save_tracking(path, tracking)
            except Exception:
                pass
        updated.append((path, tracking))
    return updated


# ── Rich table ─────────────────────────────────────────────────────────────────

def _status_text(status: str) -> Text:
    if status == "completed":
        return Text("✓ completed", style="bold green")
    if status in {"failed", "expired", "cancelled"}:
        return Text(f"✗ {status}", style="bold red")
    if status == "cancelling":
        return Text("⏸ cancelling", style="yellow")
    return Text(f"● {status}", style="yellow")


def build_status_table(items: list[tuple[Path, dict]]) -> Table:
    table = Table(
        title=f"Batch Jobs — {datetime.now().strftime('%H:%M:%S')}",
        header_style="bold white",
        border_style="blue",
        expand=True,
    )
    table.add_column("Provider", style="cyan", no_wrap=True, min_width=10)
    table.add_column("Judge", style="cyan", no_wrap=True, min_width=18)
    table.add_column("Submitted", no_wrap=True, min_width=16)
    table.add_column("Status", no_wrap=True, min_width=14)
    table.add_column("Entries", justify="right", min_width=7)
    table.add_column("Integrated?", no_wrap=True, min_width=12)

    for _, tracking in items:
        submitted = tracking.get("submitted_at", "")[:16].replace("T", " ")
        results_file = tracking.get("results_file")
        status = tracking.get("status", "?")

        if results_file:
            integrated = Text("✓ yes", style="bold green")
        elif status == "completed":
            integrated = Text("⚠ run --download", style="bold yellow")
        else:
            integrated = Text("—", style="dim")

        table.add_row(
            tracking.get("provider", "?"),
            tracking.get("judge_name", "?"),
            submitted,
            _status_text(status),
            str(tracking.get("n_entries", "?")),
            integrated,
        )

    return table


# ── Entry loading ─────────────────────────────────────────────────────────────

def _load_entry(entry_id: str) -> ConversationEntry | None:
    if not DATASET_FILE.exists():
        return None
    with open(DATASET_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("id") == entry_id:
                    return ConversationEntry.model_validate(data)
            except Exception:
                pass
    return None


def _find_reasoning_msg_idx(entry: ConversationEntry) -> int | None:
    for i, msg in enumerate(entry.messages):
        if msg["role"] == "reasoning":
            return i
    return None


# ── Result integration ─────────────────────────────────────────────────────────

RawResult = tuple[str, str, dict]  # (entry_id, raw_text, metadata)


def _integrate_results(
    raw_results: list[RawResult],
    tracking: dict,
    console: Console,
) -> dict:
    """Parse results and merge into dataset.jsonl. Returns a stats dict."""
    judge_name = tracking["judge_name"]
    judge_model_id = tracking["judge_model_id"]
    taxonomy_version = tracking["taxonomy_version"]

    stats = {
        "downloaded": len(raw_results),
        "integrated": 0,
        "parse_errors": 0,
        "missing_in_dataset": 0,
        "other_errors": 0,
        "mapper_warnings": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    new_entries: dict[str, ConversationEntry] = {}

    for entry_id, raw_text, metadata in raw_results:
        # Sum up token usage
        usage = metadata.get("usage", {})
        stats["prompt_tokens"] += usage.get("prompt_tokens", 0)
        stats["completion_tokens"] += usage.get("completion_tokens", 0)

        try:
            output = parse_llm_output(raw_text)
        except Exception as e:
            console.print(f"  [yellow]  Parse error [{entry_id[:8]}]: {e}[/yellow]")
            stats["parse_errors"] += 1
            continue

        output.metadata = metadata

        entry = _load_entry(entry_id)
        if entry is None:
            console.print(f"  [yellow]  Entry {entry_id[:8]} not found in dataset.jsonl[/yellow]")
            stats["missing_in_dataset"] += 1
            continue

        ridx = _find_reasoning_msg_idx(entry)
        if ridx is None:
            console.print(f"  [yellow]  No reasoning message in {entry_id[:8]}[/yellow]")
            stats["other_errors"] += 1
            continue

        # Reconstruct reasoning_truncated from the entry (same logic as batch_label.py)
        reasoning_truncated = len(entry.messages[ridx]["content"]) > MAX_REASONING_CHARS

        # Collect mapper warnings (sentences that couldn't be aligned to spans)
        mapper_warns: list[str] = []
        def warn_fn(msg: str, entry_id: str = entry_id) -> None:
            mapper_warns.append(msg)

        try:
            labeled_entry = apply_labeling_output(
                entry=entry,
                output=output,
                reasoning_msg_idx=ridx,
                judge_name=judge_name,
                judge_model_id=judge_model_id,
                mapper_cfg=DEFAULT_MAPPER_CFG,
                warn_fn=warn_fn,
                taxonomy_version=taxonomy_version,
                reasoning_truncated=reasoning_truncated,
            )
            new_entries[entry_id] = labeled_entry
            stats["mapper_warnings"] += len(mapper_warns)
        except Exception as e:
            console.print(f"  [yellow]  Integration error [{entry_id[:8]}]: {e}[/yellow]")
            stats["other_errors"] += 1

    if new_entries:
        _atomic_merge_label_runs(DATASET_FILE, new_entries)

    stats["integrated"] = len(new_entries)
    return stats


def _print_stats(stats: dict, tracking: dict, console: Console) -> None:
    table = Table(border_style="blue", header_style="bold white", show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    ok = stats["integrated"] == stats["downloaded"]
    table.add_row(
        "Results from API",
        str(stats["downloaded"]),
    )
    table.add_row(
        "Successfully integrated",
        f"[{'green' if ok else 'yellow'}]{stats['integrated']}[/{'green' if ok else 'yellow'}]",
    )
    if stats["parse_errors"]:
        table.add_row("JSON parse errors", f"[red]{stats['parse_errors']}[/red]")
    if stats["missing_in_dataset"]:
        table.add_row("Not found in dataset", f"[red]{stats['missing_in_dataset']}[/red]")
    if stats["other_errors"]:
        table.add_row("Other errors", f"[red]{stats['other_errors']}[/red]")
    if stats["mapper_warnings"]:
        table.add_row(
            "Sentences without span match",
            f"[yellow]{stats['mapper_warnings']}[/yellow]",
        )

    # Cost and Token Ratio Analysis
    prompt_tokens = stats.get("prompt_tokens", 0)
    completion_tokens = stats.get("completion_tokens", 0)

    if prompt_tokens > 0:
        batch_cfg = tracking.get("config_snapshot", {}).get("batch", {})
        price_in = batch_cfg.get("price_per_1m_input_tokens", 1.0)
        price_out = batch_cfg.get("price_per_1m_output_tokens", 5.0)
        discount = batch_cfg.get("batch_discount", 0.5)

        actual_cost = (
            (prompt_tokens / 1_000_000 * price_in) +
            (completion_tokens / 1_000_000 * price_out)
        ) * (1.0 - discount)

        ratio = completion_tokens / prompt_tokens

        table.add_section()
        table.add_row("Total input tokens", f"{prompt_tokens:,}")
        table.add_row("Total output tokens", f"{completion_tokens:,}")
        table.add_row("Actual cost (USD)", f"[bold green]${actual_cost:.4f}[/bold green]")
        table.add_row("Actual I/O ratio", f"[bold cyan]{ratio:.2f}x[/bold cyan]")

        # Suggest update for batch_label.py
        table.add_row(
            "[dim]Suggestion[/dim]",
            f"[dim]Update {tracking['judge_name']} output_ratio to {ratio:.2f}[/dim]"
        )

    console.print(table)


# ── Provider-specific download ─────────────────────────────────────────────────

def download_anthropic(tracking: dict, secrets: dict) -> list[RawResult]:
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required: uv add anthropic")

    api_key = os.environ.get("anthropic-api-key", "") or secrets.get("anthropic-api-key", "")
    if not api_key:
        raise ValueError("anthropic-api-key not found in secrets or environment.")
    client = anthropic.Anthropic(api_key=api_key)

    results: list[RawResult] = []
    for result in client.messages.batches.results(tracking["batch_id"]):
        if result.result.type != "succeeded":
            continue
        entry_id = result.custom_id
        raw_text = result.result.message.content[0].text
        msg = result.result.message
        metadata: dict[str, Any] = {
            "usage": {
                "prompt_tokens": msg.usage.input_tokens if msg.usage else 0,
                "completion_tokens": msg.usage.output_tokens if msg.usage else 0,
            },
            "finish_reason": msg.stop_reason or "",
        }
        results.append((entry_id, raw_text, metadata))
    return results


def download_openai(tracking: dict, secrets: dict) -> list[RawResult]:
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required: uv add openai")

    api_key = (
        os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("openai-api-key", "")
        or secrets.get("openai-api-key", "")
    )
    if not api_key:
        raise ValueError("openai-api-key not found in secrets or environment.")
    client = openai.OpenAI(api_key=api_key)

    batch = client.batches.retrieve(tracking["batch_id"])
    if not batch.output_file_id:
        raise ValueError("Batch has no output_file_id — may have failed or not completed yet.")
    content = client.files.content(batch.output_file_id).text

    results: list[RawResult] = []
    for line in content.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            entry_id = obj["custom_id"]
            body = obj["response"]["body"]
            raw_text = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            metadata: dict[str, Any] = {
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
                "finish_reason": body["choices"][0].get("finish_reason", ""),
            }
            results.append((entry_id, raw_text, metadata))
        except Exception:
            pass
    return results


def download_gemini(tracking: dict, secrets: dict) -> list[RawResult]:
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai package required: uv add google-genai")

    api_key = (
        os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("gemini-api-key", "")
        or secrets.get("gemini-api-key", "")
    )
    if not api_key:
        raise ValueError("gemini-api-key not found in secrets or environment.")
    client = genai.Client(api_key=api_key)

    job = client.batches.get(name=tracking["batch_id"])
    entry_ids = tracking["entry_ids"]

    results: list[RawResult] = []

    # Gemini Developer API: inline batch results are accessible via job.responses (positional)
    responses = getattr(job, "responses", None) or getattr(job, "inline_responses", None)
    if responses is not None:
        for i, response in enumerate(responses):
            if i >= len(entry_ids):
                break
            try:
                raw_text = response.candidates[0].content.parts[0].text
                usage = getattr(response, "usage_metadata", None)
                metadata = {}
                if usage:
                    metadata["usage"] = {
                        "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                        "completion_tokens": getattr(usage, "candidates_token_count", 0),
                    }
                results.append((entry_ids[i], raw_text, metadata))
            except Exception:
                pass
        return results

    # Fallback: results stored in a destination file (Vertex AI or alternative format)
    dest = getattr(job, "dest", None)
    if dest:
        try:
            file_content = client.files.download(name=dest)
            for i, line in enumerate(file_content.decode().splitlines()):
                if not line.strip() or i >= len(entry_ids):
                    continue
                try:
                    obj = json.loads(line)
                    raw_text = obj.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    usage = obj.get("usage_metadata", {})
                    metadata = {}
                    if usage:
                        metadata["usage"] = {
                            "prompt_tokens": usage.get("prompt_token_count", 0),
                            "completion_tokens": usage.get("candidates_token_count", 0),
                        }
                    if raw_text:
                        results.append((entry_ids[i], raw_text, metadata))
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(
                f"Could not retrieve Gemini batch results from dest={dest}: {e}\n"
                "The Gemini batch result format may have changed — check google-genai docs."
            )
        return results

    raise RuntimeError(
        "Could not retrieve Gemini batch results: job has neither 'responses' nor 'dest' attribute.\n"
        "The google-genai SDK version may not support this retrieval method yet."
    )



DOWNLOAD_FNS = {
    "anthropic": download_anthropic,
    "openai": download_openai,
    "gemini": download_gemini,
}


def process_downloads(items: list[tuple[Path, dict]], console: Console) -> None:
    any_downloaded = False
    for path, tracking in items:
        if tracking.get("status") != "completed":
            continue
        if tracking.get("results_file") is not None:
            continue  # already downloaded

        provider = tracking.get("provider", "")
        judge_name = tracking.get("judge_name", "?")
        console.print(f"\n[bold]Downloading:[/bold] [cyan]{judge_name}[/cyan] ({provider})")

        secrets = _load_secrets()
        download_fn = DOWNLOAD_FNS.get(provider)
        if download_fn is None:
            console.print(f"  [red]Unknown provider '{provider}'. Supported: {list(DOWNLOAD_FNS)}[/red]")
            continue

        try:
            raw_results = download_fn(tracking, secrets)
        except Exception as e:
            console.print(f"  [red]Download failed: {e}[/red]")
            continue

        console.print(f"  Downloaded {len(raw_results)} results from API")

        # Save raw results for debugging / re-processing
        batch_id_short = tracking["batch_id"].replace("/", "_")[-30:]
        results_path = BATCHES_DIR / f"{batch_id_short}_results.jsonl"
        with open(results_path, "w") as f:
            for entry_id, raw_text, meta in raw_results:
                f.write(json.dumps({"id": entry_id, "raw": raw_text, "meta": meta}) + "\n")
        console.print(f"  Raw results saved → {results_path.name}")

        console.print(f"  Integrating into dataset.jsonl...")
        stats = _integrate_results(raw_results, tracking, console)
        _print_stats(stats, tracking, console)

        # Update tracking file
        tracking = dict(tracking)
        tracking["results_file"] = str(results_path)
        save_tracking(path, tracking)
        any_downloaded = True

    if not any_downloaded:
        pending_complete = [
            t for _, t in items
            if t.get("status") == "completed" and t.get("results_file") is not None
        ]
        if pending_complete:
            console.print("\n[dim]All completed batches already downloaded.[/dim]")
        else:
            complete = [t for _, t in items if t.get("status") == "completed"]
            if not complete:
                console.print("\n[dim]No completed batches available for download yet.[/dim]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll batch labeling jobs and integrate completed results."
    )
    parser.add_argument(
        "--watch", type=int, metavar="SECONDS", nargs="?", const=5,
        help="Poll every N seconds (default 5). Ctrl+C to stop.",
    )
    args = parser.parse_args()

    # Default behaviour: poll continuously every 5s (same as --watch 5)
    interval = args.watch if args.watch is not None else 5

    console = Console()

    prev_statuses: dict[str, str] = {}

    def prompt_download(newly_done: list[tuple[Path, dict]]) -> None:
        for path, tracking in newly_done:
            judge_name = tracking.get("judge_name", "?")
            n = tracking.get("n_entries", "?")
            console.print(
                f"\n[bold green]✓ Batch finished:[/bold green] "
                f"[cyan]{judge_name}[/cyan] ({n} entries)"
            )
            try:
                answer = input("  Download and integrate into dataset.jsonl now? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer in ("", "y", "yes"):
                secrets = _load_secrets()
                download_fn = DOWNLOAD_FNS.get(tracking.get("provider", ""))
                if download_fn is None:
                    console.print("  [red]Unknown provider — cannot download.[/red]")
                    continue
                try:
                    raw_results = download_fn(tracking, secrets)
                except Exception as e:
                    console.print(f"  [red]Download failed: {e}[/red]")
                    continue
                batch_id_short = tracking["batch_id"].replace("/", "_")[-30:]
                results_path = BATCHES_DIR / f"{batch_id_short}_results.jsonl"
                with open(results_path, "w") as f:
                    for entry_id, raw_text, meta in raw_results:
                        f.write(json.dumps({"id": entry_id, "raw": raw_text, "meta": meta}) + "\n")
                stats = _integrate_results(raw_results, tracking, console)
                _print_stats(stats, tracking, console)
                tracking = dict(tracking)
                tracking["results_file"] = str(results_path)
                save_tracking(path, tracking)
            else:
                console.print("  [dim]Skipped — results will stay on the provider's servers until you re-run.[/dim]")

    try:
        while True:
            console.clear()
            items = load_tracking_files()
            if not items:
                console.print("[dim]No batch tracking files found in _artifacts/batches/[/dim]")
                break
            secrets = _load_secrets()
            items = refresh_statuses(items, secrets)
            console.print(build_status_table(items))

            # Detect newly completed batches and prompt to download
            newly_done = [
                (path, tracking)
                for path, tracking in items
                if tracking.get("status") == "completed"
                and not tracking.get("results_file")
                and prev_statuses.get(tracking["batch_id"]) != "completed"
            ]
            if newly_done:
                prompt_download(newly_done)
                items = load_tracking_files()  # reload after potential download

            prev_statuses.update({t["batch_id"]: t.get("status", "") for _, t in items})

            all_done = all(t.get("status") in TERMINAL_STATUSES for _, t in items)
            if all_done:
                all_integrated = all(
                    t.get("results_file") or t.get("status") != "completed"
                    for _, t in items
                )
                if all_integrated:
                    console.print("\n[bold green]All done — results integrated into dataset.jsonl.[/bold green]")
                else:
                    console.print("\n[bold green]All batches finished.[/bold green]")
                break

            console.print(f"\n[dim]Refreshing in {interval}s... (Ctrl+C to stop)[/dim]")
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")


if __name__ == "__main__":
    main()
