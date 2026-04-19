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

LABELERS_FILE = "labelers.yaml"  # single source of truth; edit to add/enable/disable labelers

# Only process input files whose names contain at least one of these substrings.
# Set to None (or empty list) to process all files.
INCLUDE_FILES = [
    "deepseek-r1-distill-llama-8b_",           # base model
    "deepseek-r1-distill-llama-8b-self-heretic_",  # self-heretic variant
    "gemma-4-26b-a4b_",
    "gemma-4-26b-a4b-heretic_",
    "gpt-oss-20b_",
    "gpt-oss-20b-heretic-ara-v3-i1_",
    "ministral-3-8B-Reasoning-2512_",
    "ministral-3-8B-Reasoning-2512-self-heretic_",
    "phi-4-reasoning-14B_",
    "qwen3.5-9B_",
    "qwen3.5-9B-heretic-v2_",
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


def count_labeled(suffix: str, taxonomy_version: str = "") -> int:
    base = OUTPUT_DIR / taxonomy_version if taxonomy_version else OUTPUT_DIR
    if not base.exists():
        return 0
    total = 0
    for f in base.glob(f"*{suffix}.jsonl"):
        try:
            total += sum(1 for _ in open(f))
        except Exception:
            pass
    return total


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


def build_dashboard(states: list[JudgeState], start_time: float) -> Table:
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
    table.add_column("Budget used", no_wrap=True, min_width=16)
    table.add_column("Labeled", justify="right", min_width=7)
    table.add_column("Errors", justify="right", min_width=6)
    table.add_column("Last activity", overflow="fold")

    for state in states:
        log = parse_log(state.meta.logfile)
        used = read_budget_used(state.meta)
        labeled = count_labeled(state.meta.suffix, state.meta.taxonomy_version)

        err_text = Text(str(log["error_count"]))
        if log["error_count"] > 0:
            err_text.stylize("red")

        table.add_row(
            state.meta.name,
            _status_text(state, log),
            _budget_text(used, state.meta.budget_limit, state.meta.budget_type),
            str(labeled),
            err_text,
            log["last_signal"],
        )

    return table


# ── Status subcommand ──────────────────────────────────────────────────────────

def print_status(console: Console, metas: list[JudgeMeta]) -> None:
    """Print a one-shot status table (like the old 'bash run_all.sh status')."""
    table = Table(title="Judge Status", header_style="bold white", border_style="blue")
    table.add_column("Judge", style="cyan", no_wrap=True)
    table.add_column("Process")
    table.add_column("Budget used", no_wrap=True)
    table.add_column("Labeled", justify="right")
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
        labeled = count_labeled(meta.suffix, meta.taxonomy_version)

        err_text = Text(str(log["error_count"]))
        if log["error_count"] > 0:
            err_text.stylize("red")

        table.add_row(
            meta.name,
            proc_text,
            _budget_text(used, meta.budget_limit, meta.budget_type),
            str(labeled),
            err_text,
            log["last_signal"],
        )

    console.print(table)


# ── Work order ─────────────────────────────────────────────────────────────────

WORK_ORDER_FILE = HERE / "work_order.json"
WORK_ORDER_SEED = 42


def _count_labeled_for_file(stem: str) -> int:
    """Count how many judges have any labeled output for the given input file stem."""
    if not OUTPUT_DIR.exists():
        return 0
    return sum(1 for f in OUTPUT_DIR.glob(f"{stem}_*.jsonl") if f.stat().st_size > 0)


def generate_work_order(input_dir: Path, output_path: Path, seed: int) -> None:
    """Generate a shared breadth-first work order that all judges will follow.

    Strategy: round-robin interleaving across all files so that with any daily
    budget, every model/dataset gets roughly equal coverage.

    File ordering within each round:
      1. Files where at least one judge has already labeled some entries
         — sorted by how many judges are ahead, descending (catch-up first)
      2. Files not yet touched by any judge — shuffled randomly

    Within each file, entries are shuffled with a fixed seed.
    flat_order is the canonical processing sequence: one entry from file 1,
    one from file 2, ..., one from file N, then repeat.  This ensures that
    any budget-limited run labels a few entries from EVERY file before going
    deeper into any single file.
    """
    import random as _rng_mod
    rng = _rng_mod.Random(seed)

    input_files = sorted(input_dir.glob("*.jsonl"))
    if INCLUDE_FILES:
        input_files = [f for f in input_files if any(pat in f.name for pat in INCLUDE_FILES)]
    if not input_files:
        raise FileNotFoundError(f"No .jsonl files found in {input_dir}")

    # Split into already-started vs untouched, then sort/shuffle each group
    started = sorted(
        [f for f in input_files if _count_labeled_for_file(f.stem) > 0],
        key=lambda f: _count_labeled_for_file(f.stem),
        reverse=True,
    )
    untouched = [f for f in input_files if _count_labeled_for_file(f.stem) == 0]
    rng.shuffle(untouched)
    file_order = started + untouched

    # Load and shuffle IDs per file
    per_file: dict[str, list[str]] = {}
    for fpath in file_order:
        ids: list[str] = []
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "id" in data:
                        ids.append(data["id"])
                except Exception:
                    pass
        rng.shuffle(ids)
        per_file[fpath.name] = ids

    # Build flat round-robin order across all files
    # Round 1: entry[0] from file1, entry[0] from file2, ..., entry[0] from fileN
    # Round 2: entry[1] from file1, entry[1] from file2, ..., entry[1] from fileN
    # etc.
    file_names = [f.name for f in file_order]
    queues: dict[str, list[str]] = {fname: list(ids) for fname, ids in per_file.items()}
    flat_order: list[dict[str, str]] = []
    while any(queues[fname] for fname in file_names):
        for fname in file_names:
            if queues[fname]:
                flat_order.append({"file": fname, "id": queues[fname].pop(0)})

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "strategy": "breadth_first_round_robin",
        "total_entries": len(flat_order),
        "started_files": len(started),
        "untouched_files": len(untouched),
        "file_order": file_names,
        "flat_order": flat_order,
        "per_file": per_file,  # kept for checkpoint compatibility
    }
    output_path.write_text(json.dumps(payload, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    console = Console()

    # Load config metadata from the registry
    registry = LabelerRegistry.from_yaml(HERE / LABELERS_FILE)
    metas: list[JudgeMeta] = []
    for entry in registry.all_entries(enabled_only=True):
        name = (entry.get("judge") or {}).get("name", "unknown")
        try:
            metas.append(load_judge_meta(name, entry))
        except Exception as e:
            console.print(f"[yellow]Warning: skipping {name}: {e}[/yellow]")

    if not metas:
        console.print("[red]No valid configs found.[/red]")
        sys.exit(1)

    # Status subcommand
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        print_status(console, metas)
        return

    input_arg = sys.argv[1] if len(sys.argv) > 1 else "../../data/1_generated/"
    input_dir = (HERE / input_arg).resolve() if not Path(input_arg).is_absolute() else Path(input_arg)

    # Ensure work order exists — generate it if not
    if not WORK_ORDER_FILE.exists():
        console.print(
            f"[bold yellow]Work order not found — generating work_order.json "
            f"(all entries, seed={WORK_ORDER_SEED})...[/bold yellow]"
        )
        try:
            generate_work_order(input_dir, WORK_ORDER_FILE, WORK_ORDER_SEED)
            with open(WORK_ORDER_FILE) as f:
                wo = json.load(f)
            console.print(
                f"[green]Generated:[/green] {wo['total_entries']} entries across "
                f"{len(wo['file_order'])} files "
                f"({wo['started_files']} already started, {wo['untouched_files']} untouched)\n"
            )
        except Exception as e:
            console.print(f"[red]Failed to generate work order: {e}[/red]")
            sys.exit(1)
    else:
        with open(WORK_ORDER_FILE) as f:
            wo = json.load(f)
        console.print(
            f"[dim]Work order:[/dim] {wo['total_entries']} entries across "
            f"{len(wo['file_order'])} files — "
            f"generated {wo['generated_at'][:10]}\n"
        )

    console.print(f"[bold]Starting {len(metas)} judges on:[/bold] {input_arg}")
    console.print(f"Logs: run_all_<judge>.log   |   Ctrl+C to stop\n")

    # Launch all judge subprocesses
    states: list[JudgeState] = []
    log_handles = []
    for meta in metas:
        console.print(f"  Starting [cyan]{meta.name}[/cyan] → {meta.logfile.name}")
        log_fh = open(meta.logfile, "w")
        log_handles.append(log_fh)
        proc = subprocess.Popen(
            ["uv", "run", "run.py", "--config", meta.config_file, "--judge", meta.name, input_arg],
            cwd=str(HERE),
            stdout=log_fh,
            stderr=log_fh,
        )
        states.append(JudgeState(meta=meta, proc=proc, _log_fh=log_fh))

    console.print()
    start_time = time.monotonic()

    try:
        with Live(console=console, refresh_per_second=0.5, screen=False) as live:
            while True:
                for s in states:
                    s.poll()
                live.update(build_dashboard(states, start_time))

                if all(not s.running for s in states if s.proc is not None):
                    time.sleep(1)
                    for s in states:
                        s.poll()
                    live.update(build_dashboard(states, start_time))
                    break

                time.sleep(2)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — sending SIGTERM to all judges...[/yellow]")
        for state in states:
            if state.running:
                state.proc.terminate()
        console.print("[dim]Checkpoints are saved. Re-run tomorrow to resume.[/dim]")
    finally:
        for fh in log_handles:
            try:
                fh.close()
            except Exception:
                pass

    console.print()
    total = sum(count_labeled(s.meta.suffix, s.meta.taxonomy_version) for s in states)
    console.print(f"[bold green]=== All done ===[/bold green]  total labeled: [green]{total}[/green]")


if __name__ == "__main__":
    main()
