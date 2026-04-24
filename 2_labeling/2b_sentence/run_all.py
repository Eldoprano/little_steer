#!/usr/bin/env python3
"""run_all.py — Run all judges in parallel with a live Rich dashboard.

Usage:
    uv run run_all.py                        # label everything in data/1_generated/
    uv run run_all.py path/to/file.jsonl     # label a specific file
    uv run run_all.py status                 # show status of a running or finished batch

Each judge's full output is saved to run_all_<name>.log.
Checkpoints prevent re-labeling — re-run tomorrow to resume after a budget reset.
"""

from __future__ import annotations

import json
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rich.console import Console
from sentence_labeler.taxonomy_loader import get_taxonomy_version
from sentence_labeler.schema import LabelerRegistry
from rich.live import Live
from rich.table import Table
from rich.text import Text

HERE = Path(__file__).parent
ARTIFACTS_DIR = HERE / "_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = (HERE / "../../data/2b_labeled").resolve()
DATASET_FILE = (HERE / "../../data/dataset.jsonl").resolve()

LABELERS_FILE = "labelers.yaml"  # single source of truth; edit to add/enable/disable labelers

# Entries from these datasets are excluded from the work order entirely.
EXCLUDE_DATASETS = ["lima"]

# Only process entries whose model field contains at least one of these substrings.
# Set to None (or empty list) to process all entries.
INCLUDE_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "simmessa/DeepSeek-R1-Distill-Llama-8B-heretic",
    "AISafety-Student/DeepSeek-R1-Distill-Llama-8B-heretic",
    "google/gemma-4-26b-a4b",
    "nohurry/gemma-4-26B-A4B-it-heretic-GGUF",
    "gpt-oss-20b",
    "gpt-oss-20b-heretic-ara-v3-i1",
    "mistralai/Ministral-3-8B-Reasoning-2512",
    "ministral-3-8b-reasoning-2512-heretic_gguf",
    "unsloth/Phi-4-reasoning-GGUF",
    "Qwen/Qwen3.5-9B",
    "trohrbaugh/Qwen3.5-9B-heretic-v2",
]

# Matches ANSI escape codes and Unicode spinner/box-drawing characters
_NOISE_RE = re.compile(r"\x1b\[[0-9;]*[mKABCDEFGHJKSTf]|[━─│┌┐└┘├┤┬┴┼╴╵╶╷]")
_SPINNER_CHARS = set("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

# Lines matching these patterns are "signal" lines worth showing
_SIGNAL_RE = re.compile(
    r"(failed|error|billing quota|daily limit|budget|labeled|summary|judge|traceback|starting)",
    re.IGNORECASE,
)


# ── Config loading ─────────────────────────────────────────────────────────────

@dataclass
class JudgeMeta:
    config_file: str
    name: str
    suffix: str
    budget_type: str       # "tokens" or "requests"
    budget_limit: int
    budget_state_file: Path
    logfile: Path
    taxonomy_version: str = ""  # e.g. "v5"; output is scoped to {OUTPUT_DIR}/{taxonomy_version}/


def load_judge_meta(name: str, entry: dict) -> JudgeMeta:
    """Build JudgeMeta from a merged labeler entry dict (as returned by LabelerRegistry)."""
    judge = entry.get("judge") or {}
    output = entry.get("output") or {}
    suffix = output.get("suffix", f"_{name}")
    taxonomy_version = output.get("taxonomy_version", "") or get_taxonomy_version()
    pipeline = entry.get("pipeline") or {}

    if pipeline.get("token_budget") is not None:
        budget_type = "tokens"
        budget_limit = int(pipeline["token_budget"])
        state_file = pipeline.get("budget_state_file", "token_budget_state.json")
    else:
        budget_type = "requests"
        budget_limit = int(judge.get("rpd") or 9_999_999)
        state_file = pipeline.get("request_state_file", "request_budget_state.json")

    return JudgeMeta(
        config_file=LABELERS_FILE,
        name=name,
        suffix=suffix,
        budget_type=budget_type,
        budget_limit=budget_limit,
        budget_state_file=HERE / state_file,
        logfile=ARTIFACTS_DIR / f"run_all_{name}.log",
        taxonomy_version=taxonomy_version,
    )


# ── Data readers ───────────────────────────────────────────────────────────────

def read_budget_used(meta: JudgeMeta) -> int:
    if not meta.budget_state_file.exists():
        return 0
    try:
        data = json.loads(meta.budget_state_file.read_text())
        today = datetime.now(timezone.utc).date().isoformat()
        return int(data.get(today, 0))
    except Exception:
        return 0


# Cache: (mtime, labeled_ids_by_judge)
_labeled_cache: tuple[float, dict[str, set[str]]] | None = None


def _scan_dataset_labeled_ids() -> dict[str, set[str]]:
    """Scan dataset.jsonl once and return a set of labeled entry IDs per judge_name."""
    labeled: dict[str, set[str]] = {}
    if not DATASET_FILE.exists():
        return labeled
    try:
        with open(DATASET_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    eid = data.get("id")
                    runs = data.get("label_runs") or []
                    for r in runs:
                        jn = r.get("judge_name")
                        if jn and eid:
                            labeled.setdefault(jn, set()).add(eid)
                except Exception:
                    pass
    except Exception:
        pass
    return labeled


def _scan_checkpoint_labeled_ids(suffix: str) -> set[str]:
    """Scan the per-judge checkpoint file for currently labeled IDs."""
    # Checkpoint filename logic matches pipeline.py: dataset{suffix}.checkpoint.json
    cp_path = DATASET_FILE.parent / f"dataset{suffix}.checkpoint.json"
    if not cp_path.exists():
        return set()
    try:
        with open(cp_path) as f:
            data = json.load(f)
            return set(data.get("labeled_ids", []))
    except Exception:
        return set()


def count_labeled(judge_name: str, suffix: str = "", work_order_ids: set[str] | None = None) -> int:
    """Count entries in dataset.jsonl + checkpoint that have a label_run from this judge.

    If work_order_ids is provided, only counts entries that are also in that set.
    Result is cached until dataset.jsonl is modified.
    """
    global _labeled_cache
    if not DATASET_FILE.exists() or not judge_name:
        return 0
    try:
        mtime = DATASET_FILE.stat().st_mtime
    except OSError:
        return 0

    if _labeled_cache is None or _labeled_cache[0] != mtime:
        _labeled_cache = (mtime, _scan_dataset_labeled_ids())

    # Get baseline from dataset.jsonl
    labeled_set = set(_labeled_cache[1].get(judge_name, set()))
    
    # ADDITION: Also check the checkpoint file for labels not yet merged into the main file
    checkpoint_ids = _scan_checkpoint_labeled_ids(suffix)
    labeled_set.update(checkpoint_ids)

    if work_order_ids is not None:
        return len(labeled_set.intersection(work_order_ids))
    return len(labeled_set)


def _clean(line: str) -> str:
    line = _NOISE_RE.sub("", line)
    return "".join(c for c in line if c not in _SPINNER_CHARS).strip()


def parse_log(logfile: Path) -> dict:
    """Extract useful info from a judge log file."""
    result = {
        "error_count": 0,
        "budget_exhausted": False,
        "done": False,
        "last_signal": "",
    }
    if not logfile.exists():
        return result

    try:
        lines = logfile.read_text(errors="replace").splitlines()
    except Exception:
        return result

    for line in lines:
        clean = _clean(line)
        if not clean or len(clean) < 8:
            continue
        lower = clean.lower()

        if "daily limit reached" in lower or "budget already exhausted" in lower:
            result["budget_exhausted"] = True
        if "labeling summary" in lower:
            result["done"] = True
        if _SIGNAL_RE.search(clean):
            # Count failures (but not "budget" lines — those aren't errors)
            if "failed" in lower and "budget" not in lower:
                result["error_count"] += 1
            result["last_signal"] = clean[:100]

    return result


# ── Process state ──────────────────────────────────────────────────────────────

@dataclass
class JudgeState:
    meta: JudgeMeta
    proc: subprocess.Popen | None = None
    exit_code: int | None = None
    _log_fh: object = None  # kept open so the subprocess can write

    def poll(self) -> None:
        """Update exit_code if the process has finished."""
        if self.proc is not None and self.exit_code is None:
            ret = self.proc.poll()
            if ret is not None:
                self.exit_code = ret

    @property
    def running(self) -> bool:
        return self.proc is not None and self.exit_code is None


# ── Rich rendering ─────────────────────────────────────────────────────────────

def _status_text(state: JudgeState, log: dict) -> Text:
    if state.proc is None:
        return Text("WAITING", style="dim")
    if state.running:
        return Text("● RUNNING", style="bold green")
    if log["budget_exhausted"]:
        return Text("⏸ BUDGET", style="yellow")
    if state.exit_code == 0 or log["done"]:
        return Text("✓ DONE", style="green")
    return Text(f"✗ EXIT {state.exit_code}", style="bold red")


def _progress_text(labeled: int, total: int) -> Text:
    if total <= 0:
        return Text("—", style="dim")
    pct = labeled / total
    style = "green" if pct >= 1.0 else "yellow" if pct >= 0.5 else "white"
    return Text(f"{labeled}/{total} ({pct:.0%})", style=style)


def _budget_text(used: int, limit: int, budget_type: str) -> Text:
    pct = used / limit if limit > 0 else 0.0
    style = "red" if pct >= 1.0 else "yellow" if pct >= 0.75 else "green"

    if budget_type == "tokens":
        def _fmt(n: int) -> str:
            return f"{n/1_000_000:.1f}M" if n >= 1_000_000 else f"{n//1_000}k" if n >= 1_000 else str(n)
        label = f"{_fmt(used)} / {_fmt(limit)}"
    else:
        label = f"{used} / {limit} req"

    return Text(label, style=style)


def build_dashboard(
    states: list[JudgeState],
    start_time: float,
    total_entries: int = 0,
    work_order_ids: set[str] | None = None,
    pass_number: int = 1,
) -> Table:
    elapsed = int(time.monotonic() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60

    table = Table(
        title=(
            f"Labeling Dashboard — {datetime.now().strftime('%H:%M:%S')} "
            f"— elapsed {h:02d}:{m:02d}:{s:02d}"
        ),
        header_style="bold white",
        border_style="blue",
        expand=True,
    )
    table.add_column("Judge", style="cyan", no_wrap=True, min_width=22)
    table.add_column("Status", no_wrap=True, min_width=12)
    table.add_column("Progress", no_wrap=True, min_width=16)
    table.add_column("Budget used", no_wrap=True, min_width=16)
    table.add_column("Errors", justify="right", min_width=6)
    table.add_column("Last activity", overflow="fold")

    for state in states:
        log = parse_log(state.meta.logfile)
        used = read_budget_used(state.meta)

        # Count labels for this specific judge and pass
        effective_name = state.meta.name
        if pass_number > 1:
            effective_name = f"{effective_name}_pass{pass_number}"
        labeled = count_labeled(
            effective_name, 
            suffix=state.meta.suffix, 
            work_order_ids=work_order_ids
        )

        err_text = Text(str(log["error_count"]))
        if log["error_count"] > 0:
            err_text.stylize("red")

        table.add_row(
            state.meta.name,
            _status_text(state, log),
            _progress_text(labeled, total_entries),
            _budget_text(used, state.meta.budget_limit, state.meta.budget_type),
            err_text,
            log["last_signal"],
        )

    return table


# ── Status subcommand ──────────────────────────────────────────────────────────

def print_status(
    console: Console,
    metas: list[JudgeMeta],
    total_entries: int = 0,
    work_order_ids: set[str] | None = None,
    pass_number: int = 1,
) -> None:
    """Print a one-shot status table (like the old 'bash run_all.sh status')."""
    table = Table(title="Judge Status", header_style="bold white", border_style="blue")
    table.add_column("Judge", style="cyan", no_wrap=True)
    table.add_column("Process")
    table.add_column("Progress", no_wrap=True)
    table.add_column("Budget used", no_wrap=True)
    table.add_column("Errors", justify="right")
    table.add_column("Last activity", overflow="fold")

    import subprocess as sp
    for meta in metas:
        try:
            result = sp.run(
                ["pgrep", "-f", f"run.py --config {meta.config_file} --judge {meta.name}"],
                capture_output=True, text=True,
            )
            pid = result.stdout.strip().split("\n")[0] if result.returncode == 0 else None
        except Exception:
            pid = None

        proc_text = Text(f"running (pid {pid})", style="green") if pid else Text("stopped", style="dim")
        log = parse_log(meta.logfile)
        used = read_budget_used(meta)

        effective_name = meta.name
        if pass_number > 1:
            effective_name = f"{effective_name}_pass{pass_number}"
        labeled = count_labeled(
            effective_name, 
            suffix=meta.suffix, 
            work_order_ids=work_order_ids
        )

        err_text = Text(str(log["error_count"]))
        if log["error_count"] > 0:
            err_text.stylize("red")

        table.add_row(
            meta.name,
            proc_text,
            _progress_text(labeled, total_entries),
            _budget_text(used, meta.budget_limit, meta.budget_type),
            err_text,
            log["last_signal"],
        )

    console.print(table)


# ── Work order ─────────────────────────────────────────────────────────────────

WORK_ORDER_FILE = HERE / "work_order.json"
WORK_ORDER_SEED = 42


def _collect_labeled_ids() -> set[str]:
    """Scan all existing judge output files and return the union of labeled entry IDs."""
    labeled: set[str] = set()
    for version_dir in OUTPUT_DIR.iterdir() if OUTPUT_DIR.exists() else []:
        search_dir = version_dir if version_dir.is_dir() else OUTPUT_DIR
        for fpath in search_dir.glob("*.jsonl"):
            try:
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            eid = json.loads(line).get("id")
                            if eid:
                                labeled.add(eid)
                        except Exception:
                            pass
            except Exception:
                pass
        if not version_dir.is_dir():
            break  # OUTPUT_DIR has no subdirs, already searched it directly
    return labeled


def generate_work_order(dataset_file: Path, output_path: Path, seed: int) -> None:
    """Generate a shared work order from the unified dataset.jsonl file.

    Strategy: entries that have already been labeled by at least one judge go
    first (so we can quickly get multi-judge coverage for comparison), then
    untouched entries shuffled with a fixed seed.
    """
    import random as _rng_mod
    rng = _rng_mod.Random(seed)

    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    # IDs that appear in any existing judge output file
    already_labeled_ids = _collect_labeled_ids()

    started_ids: list[str] = []
    untouched_ids: list[str] = []

    with open(dataset_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                eid = data.get("id")
                if not eid:
                    continue
                dataset_name = (data.get("metadata") or {}).get("dataset_name", "")
                if EXCLUDE_DATASETS and dataset_name in EXCLUDE_DATASETS:
                    continue
                model = data.get("model", "")
                if INCLUDE_MODELS and not any(pat in model for pat in INCLUDE_MODELS):
                    continue
                if eid in already_labeled_ids or data.get("label_runs"):
                    started_ids.append(eid)
                else:
                    untouched_ids.append(eid)
            except Exception:
                pass

    rng.shuffle(untouched_ids)
    ordered_ids = started_ids + untouched_ids

    fname = dataset_file.name  # "dataset.jsonl"
    flat_order = [{"file": fname, "id": eid} for eid in ordered_ids]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "strategy": "dataset_unified",
        "total_entries": len(flat_order),
        "started_entries": len(started_ids),
        "untouched_entries": len(untouched_ids),
        "dataset_file": str(dataset_file),
        "file_order": [fname],
        "flat_order": flat_order,
        "per_file": {fname: ordered_ids},
    }
    output_path.write_text(json.dumps(payload, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    is_tty = sys.stdout.isatty()
    console = Console() if is_tty else Console(force_terminal=True)

    # Parse CLI flags before the positional input argument
    import argparse as _ap
    _parser = _ap.ArgumentParser(add_help=False)
    _parser.add_argument("--work-order", default=None, dest="work_order")
    _parser.add_argument("--pass", default=1, type=int, dest="pass_number")
    _parser.add_argument("input", nargs="?", default=None)
    _args, _ = _parser.parse_known_args()
    work_order_override: str | None = _args.work_order
    pass_number: int = _args.pass_number

    # Load config metadata from the registry
    registry = LabelerRegistry.from_yaml(HERE / LABELERS_FILE)
    metas: list[JudgeMeta] = []
    for entry in registry.all_entries(enabled_only=True):
        name = (entry.get("judge") or {}).get("name", "unknown")
        try:
            meta = load_judge_meta(name, entry)
            if pass_number > 1:
                if not meta.suffix.endswith(f"_pass{pass_number}"):
                    meta.suffix += f"_pass{pass_number}"
            metas.append(meta)
        except Exception as e:
            console.print(f"[yellow]Warning: skipping {name}: {e}[/yellow]")

    if not metas:
        console.print("[red]No valid configs found.[/red]")
        sys.exit(1)

    # Resolve work order and capture IDs/count for progress tracking
    effective_wo_file: Path = HERE / work_order_override if work_order_override else WORK_ORDER_FILE
    total_entries = 0
    work_order_ids: set[str] | None = None

    # Status subcommand (positional "status" still works)
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        if effective_wo_file.exists():
            try:
                with open(effective_wo_file) as f:
                    _wo = json.load(f)
                    total_entries = _wo.get("total_entries", 0)
                    if "flat_order" in _wo:
                        work_order_ids = {item["id"] for item in _wo["flat_order"]}
            except Exception:
                pass
        print_status(console, metas, total_entries, work_order_ids, pass_number)
        return

    input_arg = _args.input or str(DATASET_FILE)
    input_path = Path(input_arg) if Path(input_arg).is_absolute() else (HERE / input_arg).resolve()

    # Detect stale default work order (only relevant when not overriding)
    if not work_order_override and WORK_ORDER_FILE.exists():
        with open(WORK_ORDER_FILE) as f:
            _wo_check = json.load(f)
        _first = (_wo_check.get("flat_order") or [{}])[0]
        if _first.get("file") != "dataset.jsonl":
            console.print(
                "[bold yellow]Work order references old per-file structure — "
                "deleting and regenerating from dataset.jsonl...[/bold yellow]"
            )
            WORK_ORDER_FILE.unlink()

    # Ensure default work order exists — generate it if not (skipped when using override)
    if not work_order_override and not WORK_ORDER_FILE.exists():
        console.print(
            f"[bold yellow]Work order not found — generating work_order.json "
            f"(all entries, seed={WORK_ORDER_SEED})...[/bold yellow]"
        )
        try:
            generate_work_order(input_path, WORK_ORDER_FILE, WORK_ORDER_SEED)
        except Exception as e:
            console.print(f"[red]Failed to generate work order: {e}[/red]")
            sys.exit(1)

    # Load effective work order for dashboard
    if effective_wo_file.exists():
        with open(effective_wo_file) as f:
            wo = json.load(f)
        total_entries = wo.get("total_entries", 0)
        if "flat_order" in wo:
            work_order_ids = {item["id"] for item in wo["flat_order"]}
        
        pass_label = f"  [dim](pass {pass_number})[/dim]" if pass_number > 1 else ""
        console.print(
            f"[dim]Work order:[/dim] {effective_wo_file.name} — "
            f"{total_entries} entries — "
            f"generated {wo['generated_at'][:10]}{pass_label}\n"
        )

    console.print(f"[bold]Starting {len(metas)} judges on:[/bold] {input_path}")
    console.print(f"Logs: run_all_<judge>.log   |   Ctrl+C to stop\n")

    # Launch all judge subprocesses
    states: list[JudgeState] = []
    log_handles = []
    for meta in metas:
        console.print(f"  Starting [cyan]{meta.name}[/cyan] → {meta.logfile.name}")
        log_fh = open(meta.logfile, "w")
        log_handles.append(log_fh)
        cmd = ["uv", "run", "run.py", "--config", meta.config_file, "--judge", meta.name]
        if work_order_override:
            cmd += ["--work-order", work_order_override]
        if pass_number > 1:
            cmd += ["--pass", str(pass_number)]
        cmd.append(str(input_path))
        proc = subprocess.Popen(
            cmd,
            cwd=str(HERE),
            stdout=log_fh,
            stderr=log_fh,
        )
        states.append(JudgeState(meta=meta, proc=proc, _log_fh=log_fh))

    console.print()
    start_time = time.monotonic()

    try:
        if is_tty:
            with Live(console=console, refresh_per_second=0.5, screen=False) as live:
                while True:
                    for s in states:
                        s.poll()
                    live.update(build_dashboard(states, start_time, total_entries, work_order_ids, pass_number))

                    if all(not s.running for s in states if s.proc is not None):
                        time.sleep(1)
                        for s in states:
                            s.poll()
                        live.update(build_dashboard(states, start_time, total_entries, work_order_ids, pass_number))
                        break

                    time.sleep(2)
        else:
            while True:
                for s in states:
                    s.poll()
                console.print(build_dashboard(states, start_time, total_entries, work_order_ids, pass_number))

                if all(not s.running for s in states if s.proc is not None):
                    time.sleep(1)
                    for s in states:
                        s.poll()
                    console.print(build_dashboard(states, start_time, total_entries, work_order_ids, pass_number))
                    break

                time.sleep(2)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — sending SIGINT for graceful shutdown...[/yellow]")
        for state in states:
            if state.running:
                state.proc.send_signal(signal.SIGINT)
        console.print("[dim]Waiting for workers to save... (Ctrl+C again to force exit)[/dim]")
        # Give them a few seconds, then we exit run_all.py itself
        time.sleep(3)
    finally:
        for fh in log_handles:
            try:
                fh.close()
            except Exception:
                pass

    console.print()
    total = sum(
        count_labeled(
            s.meta.name if pass_number == 1 else f"{s.meta.name}_pass{pass_number}", 
            suffix=s.meta.suffix,
            work_order_ids=work_order_ids
        ) for s in states
    )
    console.print(f"[bold green]=== All done ===[/bold green]  total labeled: [green]{total}[/green]")


if __name__ == "__main__":
    main()
