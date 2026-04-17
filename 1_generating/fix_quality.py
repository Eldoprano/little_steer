#!/usr/bin/env python3
"""
fix_quality.py — Audit and repair quality issues in generated JSONL files.

Three problems are detected and handled:
  1. think_artifact  — assistant content starts with a bare [/THINK] or </think>
                       tag (LMStudio artifact); fixed in-place without regeneration.
  2. max_length      — response was cut off when it hit the token limit.
  3. repetition      — model entered an n-gram repetition loop.

The script also manages the ``approved`` flag in metadata and can sync bad
entries out of 2b_labeled output files and their checkpoints.

Usage:
    uv run python fix_quality.py                        # audit only (no writes)
    uv run python fix_quality.py --tag                  # add approved field to all entries
    uv run python fix_quality.py --fix                  # fix think_artifacts in-place
    uv run python fix_quality.py --sync-labeled         # remove bad entries from 2b_labeled
    uv run python fix_quality.py --remove               # delete bad entries from 1_generated
    uv run python fix_quality.py --tag --fix --sync-labeled --remove  # do everything

Filters:
    --data-dir PATH     Path to 1_generated dir (default: ../data/1_generated)
    --labeled-dir PATH  Path to 2b_labeled dir  (default: ../data/2b_labeled)
    --model NAME        Process only files whose name contains NAME
    --dry-run           Print what would change without writing anything
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from thesis_schema import ConversationEntry  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Detection (mirrors generate_responses.py helpers)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_repetition(text: str, n: int = 8, threshold: int = 3) -> tuple[bool, int]:
    words = text.split()
    if len(words) < n * 2:
        return False, -1
    seen: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        if gram in seen:
            prev = seen[gram]
            if i - prev <= n * threshold:
                return True, prev
        seen[gram] = i
    return False, -1


def _has_think_artifact(text: str) -> bool:
    stripped = text.lstrip()
    return stripped.startswith("[/THINK]") or stripped.startswith("</think>")


def _entry_quality_issues(entry: ConversationEntry) -> set[str]:
    issues: set[str] = set()
    reasoning_content = next(
        (m.get("content", "") for m in entry.messages if m["role"] == "reasoning"), None
    )
    assistant_content = next(
        (m.get("content", "") for m in entry.messages if m["role"] == "assistant"), None
    )

    if assistant_content and _has_think_artifact(assistant_content):
        issues.add("think_artifact")

    finish_reason = entry.metadata.get("finish_reason", "")
    if finish_reason == "max_length":
        issues.add("max_length")
    elif finish_reason == "failed":
        issues.add("failed")

    for content in (reasoning_content, assistant_content):
        if content:
            has_rep, _ = _detect_repetition(content)
            if has_rep:
                issues.add("repetition")
                break

    if reasoning_content is None:
        issues.add("missing_reasoning")
    if not assistant_content:
        issues.add("empty_response")

    return issues


def _is_approved(entry: ConversationEntry) -> bool:
    blocking = {"think_artifact", "max_length", "repetition", "failed", "empty_response"}
    return not _entry_quality_issues(entry).intersection(blocking)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_entries(path: Path) -> list[ConversationEntry]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(ConversationEntry.model_validate_json(line))
                except Exception:
                    pass
    return entries


def _rewrite(path: Path, entries: list[ConversationEntry], dry_run: bool) -> None:
    if dry_run:
        return
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.model_dump_json() + "\n")
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Per-file operations
# ─────────────────────────────────────────────────────────────────────────────

def audit_file(path: Path) -> dict[str, int]:
    """Return issue counts for a single JSONL file."""
    counts: dict[str, int] = Counter()
    for entry in _load_entries(path):
        issues = _entry_quality_issues(entry)
        for issue in issues:
            counts[issue] += 1
        if not issues:
            counts["ok"] += 1
    return counts


def tag_file(path: Path, dry_run: bool) -> int:
    """Add/update the ``approved`` field on every entry. Returns number of changes."""
    entries = _load_entries(path)
    changed = 0
    for entry in entries:
        new_val = _is_approved(entry)
        if entry.metadata.get("approved") != new_val:
            entry.metadata["approved"] = new_val
            changed += 1
    if changed:
        _rewrite(path, entries, dry_run)
    return changed


def fix_artifacts_file(path: Path, dry_run: bool) -> int:
    """Strip leading [/THINK] / </think> from assistant content. Returns fixes made."""
    entries = _load_entries(path)
    fixed = 0
    for entry in entries:
        assistant_msg = next(
            (m for m in entry.messages if m["role"] == "assistant"), None
        )
        if assistant_msg and _has_think_artifact(assistant_msg.get("content", "")):
            content = assistant_msg["content"].lstrip()
            for tag in ("[/THINK]", "</think>"):
                if content.startswith(tag):
                    content = content[len(tag):].lstrip()
                    break
            assistant_msg["content"] = content
            entry.metadata["approved"] = _is_approved(entry)
            fixed += 1
    if fixed:
        _rewrite(path, entries, dry_run)
    return fixed


def remove_bad_file(path: Path, dry_run: bool) -> tuple[int, set[str]]:
    """Remove entries with blocking quality issues. Returns (removed_count, removed_ids)."""
    BLOCKING = {"max_length", "repetition", "failed"}
    entries = _load_entries(path)
    bad_ids: set[str] = set()
    kept = []
    for entry in entries:
        issues = _entry_quality_issues(entry)
        if issues.intersection(BLOCKING):
            bad_ids.add(entry.id)
        else:
            kept.append(entry)
    if bad_ids:
        _rewrite(path, kept, dry_run)
    return len(bad_ids), bad_ids


# ─────────────────────────────────────────────────────────────────────────────
# Labeled-data sync
# ─────────────────────────────────────────────────────────────────────────────

def _find_labeled_files(labeled_dir: Path) -> list[Path]:
    """Recursively find all JSONL files under labeled_dir."""
    return sorted(labeled_dir.rglob("*.jsonl"))


def _remove_ids_from_labeled(labeled_path: Path, bad_ids: set[str], dry_run: bool) -> int:
    """Remove entries with given IDs from a labeled JSONL. Returns count removed."""
    if not labeled_path.exists() or not bad_ids:
        return 0
    entries = []
    removed = 0
    with open(labeled_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("id") in bad_ids:
                    removed += 1
                else:
                    entries.append(line)
            except json.JSONDecodeError:
                entries.append(line)
    if removed and not dry_run:
        tmp = labeled_path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for line in entries:
                f.write(line + "\n")
        tmp.replace(labeled_path)
    return removed


def _remove_ids_from_checkpoint(checkpoint_path: Path, bad_ids: set[str], dry_run: bool) -> int:
    """Remove IDs from checkpoint's labeled_ids and failed_ids. Returns count removed."""
    if not checkpoint_path.exists() or not bad_ids:
        return 0
    with open(checkpoint_path) as f:
        data = json.load(f)
    labeled = data.get("labeled_ids", [])
    failed = data.get("failed_ids", {})
    removed = 0
    new_labeled = [eid for eid in labeled if eid not in bad_ids]
    new_failed = {k: v for k, v in failed.items() if k not in bad_ids}
    removed = (len(labeled) - len(new_labeled)) + (len(failed) - len(new_failed))
    if removed and not dry_run:
        data["labeled_ids"] = new_labeled
        data["failed_ids"] = new_failed
        tmp = checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.rename(checkpoint_path)
    return removed


def _reasoning_broken(entry: ConversationEntry) -> bool:
    """True when the *reasoning* content is bad or missing.

    Used by sync_labeled: since the labeling pipeline labels reasoning sentences,
    only entries with broken reasoning should be purged.

    Rules:
    - Repetition in the reasoning block → bad reasoning data.
    - ``finish_reason == "failed"`` → entry is completely broken.
    - ``finish_reason == "max_length"`` AND the response is empty/tiny → the
      token limit was hit *inside* the reasoning block (it never finished).
    - No reasoning content at all → nothing to label.

    Intentionally NOT removed:
    - ``max_length`` where a non-trivial response exists → reasoning completed,
      only the response was cut off; reasoning sentences are still valid.
    - ``empty_response`` alone → reasoning is present and usable.
    - ``think_artifact`` → the reasoning field is intact; only response
      formatting is broken.
    - Repetition only in the response (not in reasoning) → reasoning is fine.
    """
    reasoning_content = next(
        (m.get("content", "") for m in entry.messages if m["role"] == "reasoning"), None
    )
    assistant_content = next(
        (m.get("content", "") for m in entry.messages if m["role"] == "assistant"), None
    )
    finish_reason = entry.metadata.get("finish_reason", "")

    # No reasoning at all → nothing to label.
    if not reasoning_content:
        return True

    # Failed generation.
    if finish_reason == "failed":
        return True

    # Repetition in the reasoning block.
    if _detect_repetition(reasoning_content)[0]:
        return True

    # max_length where reasoning itself was cut (response is absent or tiny).
    if finish_reason == "max_length":
        if not assistant_content or len(assistant_content.strip()) < 50:
            return True

    return False


def sync_labeled(
    generated_dir: Path,
    labeled_dir: Path,
    model_filter: str | None,
    dry_run: bool,
    console: Console,
) -> None:
    """Remove entries from 2b_labeled whose *reasoning* is broken in 1_generated.

    Since the labeling pipeline labels reasoning sentences, only entries with
    bad or missing reasoning are purged.  Entries where the response is broken
    but the reasoning is complete are kept — their reasoning labels are valid.

    See ``_reasoning_broken`` for exact criteria.
    """
    if not labeled_dir.exists():
        console.print(f"  [yellow]Labeled dir not found:[/yellow] {labeled_dir}")
        return

    # Collect IDs where the reasoning is broken.
    bad_ids_global: set[str] = set()
    for gen_path in sorted(generated_dir.glob("*.jsonl")):
        if model_filter and model_filter not in gen_path.name:
            continue
        for entry in _load_entries(gen_path):
            if _reasoning_broken(entry):
                bad_ids_global.add(entry.id)

    if not bad_ids_global:
        console.print("  [green]No entries with broken responses found — nothing to remove.[/green]")
        return

    # ── Preview: count what would be removed ─────────────────────────────────
    preview: list[tuple[Path, int]] = []  # (path, to_remove)
    total_would_remove = 0
    total_entries_all = 0
    for labeled_path in _find_labeled_files(labeled_dir):
        if labeled_path.suffix != ".jsonl":
            continue
        file_total = 0
        file_remove = 0
        for line in open(labeled_path):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                file_total += 1
                if data.get("id") in bad_ids_global:
                    file_remove += 1
            except json.JSONDecodeError:
                pass
        total_entries_all += file_total
        if file_remove:
            preview.append((labeled_path, file_remove))
            total_would_remove += file_remove

    if total_would_remove == 0:
        console.print("  [green]No matching entries found in labeled files.[/green]")
        return

    # ── Show preview table ────────────────────────────────────────────────────
    preview_table = Table(title="Entries to remove from 2b_labeled", box=box.SIMPLE_HEAVY)
    preview_table.add_column("Labeled file", style="cyan")
    preview_table.add_column("To remove", justify="right", style="red")
    for lp, cnt in preview:
        try:
            rel = str(lp.relative_to(labeled_dir))
        except ValueError:
            rel = lp.name
        preview_table.add_row(rel, str(cnt))
    preview_table.add_section()
    preview_table.add_row("[bold]TOTAL[/bold]", f"[bold red]{total_would_remove}[/bold red]")
    console.print(preview_table)

    console.print(
        f"\n  [yellow]This will permanently delete {total_would_remove} of {total_entries_all} "
        f"labeled entries[/yellow] from {len(preview)} file(s)."
    )
    console.print("  Entries removed: broken reasoning (repetition / failed / missing / max_length mid-think).")
    console.print("  Entries kept:    good reasoning even if response was cut off or empty.")

    if dry_run:
        console.print("  [yellow]DRY RUN[/yellow] — no files written.\n")
        return

    # ── Confirmation ──────────────────────────────────────────────────────────
    try:
        answer = input("\n  Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    if answer not in ("y", "yes"):
        console.print("  [yellow]Aborted.[/yellow]\n")
        return

    # ── Execute ───────────────────────────────────────────────────────────────
    total_removed = 0
    for labeled_path in _find_labeled_files(labeled_dir):
        if labeled_path.suffix != ".jsonl":
            continue
        removed = _remove_ids_from_labeled(labeled_path, bad_ids_global, dry_run=False)
        if removed:
            try:
                rel = str(labeled_path.relative_to(labeled_dir))
            except ValueError:
                rel = labeled_path.name
            console.print(f"    [red]{rel}[/red]: removed {removed} entries")
            total_removed += removed
            cp = labeled_path.with_name(labeled_path.stem + ".checkpoint.json")
            cp_removed = _remove_ids_from_checkpoint(cp, bad_ids_global, dry_run=False)
            if cp_removed:
                console.print(f"    [dim]{cp.name}[/dim]: removed {cp_removed} checkpoint entries")

    console.print(f"  Total removed from labeled data: [red]{total_removed}[/red]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ISSUE_KEYS = ["ok", "think_artifact", "max_length", "repetition",
              "missing_reasoning", "empty_response", "failed"]

# Short column headers to keep the table narrow enough for long filenames.
ISSUE_HEADERS = {
    "ok":                "ok",
    "think_artifact":    "artifact",
    "max_length":        "max_len",
    "repetition":        "repeat",
    "missing_reasoning": "no_reason",
    "empty_response":    "empty",
    "failed":            "failed",
}

ISSUE_STYLES = {
    "ok":                "green",
    "think_artifact":    "yellow",
    "max_length":        "red",
    "repetition":        "red",
    "missing_reasoning": "magenta",
    "empty_response":    "magenta",
    "failed":            "bold red",
}


def _build_audit_table(file_reports: list[tuple[str, dict]], total_counts: Counter) -> Table:
    table = Table(title="Quality Audit", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("File", style="cyan", no_wrap=True)
    for key in ISSUE_KEYS:
        table.add_column(ISSUE_HEADERS[key], justify="right", style=ISSUE_STYLES.get(key, ""))

    for name, counts in file_reports:
        # Strip .jsonl suffix to save width.
        display_name = name.removesuffix(".jsonl")
        row_vals = [display_name] + [
            str(counts.get(k, 0)) if counts.get(k, 0) else "[dim]0[/dim]"
            for k in ISSUE_KEYS
        ]
        table.add_row(*row_vals)

    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        *[f"[bold]{total_counts.get(k, 0)}[/bold]" for k in ISSUE_KEYS],
    )
    return table


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(
        description="Audit and fix quality issues in generated JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", default=str(here / "../data/1_generated"),
        help="Path to 1_generated dir (default: ../data/1_generated)"
    )
    parser.add_argument(
        "--labeled-dir", default=str(here / "../data/2b_labeled"),
        help="Path to 2b_labeled dir (default: ../data/2b_labeled)"
    )
    parser.add_argument("--model", default=None, help="Filter to files containing MODEL in name")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--tag", action="store_true", help="Add/update approved field on all entries")
    parser.add_argument("--fix", action="store_true", help="Fix think_artifacts in-place")
    parser.add_argument("--sync-labeled", action="store_true",
                        help="Remove bad entries (approved=False) from 2b_labeled files")
    parser.add_argument("--remove", action="store_true",
                        help="Delete bad entries from 1_generated (triggers re-generation)")
    args = parser.parse_args()

    console = Console(highlight=False)
    data_dir = Path(args.data_dir).resolve()
    labeled_dir = Path(args.labeled_dir).resolve()
    dry = args.dry_run

    if dry:
        console.print("[bold yellow]DRY RUN[/bold yellow] — no files will be written.\n")

    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        console.print(f"[red]No JSONL files found in {data_dir}[/red]")
        return
    if args.model:
        files = [f for f in files if args.model in f.name]
        if not files:
            console.print(f"[red]No files match model filter '{args.model}'[/red]")
            return

    # ── Audit ─────────────────────────────────────────────────────────────────
    console.print(f"Auditing [cyan]{len(files)}[/cyan] file(s) in [dim]{data_dir}[/dim]...\n")
    total_counts: Counter = Counter()
    file_reports: list[tuple[str, dict]] = []
    for path in files:
        counts = audit_file(path)
        file_reports.append((path.name, counts))
        for k, v in counts.items():
            total_counts[k] += v

    console.print(_build_audit_table(file_reports, total_counts))

    any_action = args.tag or args.fix or args.sync_labeled or args.remove

    # ── Tag ───────────────────────────────────────────────────────────────────
    if args.tag:
        console.rule("[bold]Tagging entries[/bold]")
        total_changed = 0
        for path in files:
            changed = tag_file(path, dry)
            if changed:
                tag = "[yellow]DRY[/yellow] " if dry else ""
                console.print(f"  {tag}[cyan]{path.name}[/cyan]: {changed} entries updated")
            total_changed += changed
        console.print(f"  Total tagged: [green]{total_changed}[/green]\n")

    # ── Fix artifacts ─────────────────────────────────────────────────────────
    if args.fix:
        console.rule("[bold]Fixing \\[/THINK] artifacts[/bold]")
        total_fixed = 0
        for path in files:
            fixed = fix_artifacts_file(path, dry)
            if fixed:
                tag = "[yellow]DRY[/yellow] " if dry else ""
                console.print(f"  {tag}[cyan]{path.name}[/cyan]: {fixed} artifact(s) fixed")
            total_fixed += fixed
        console.print(f"  Total fixed: [green]{total_fixed}[/green]\n")

    # ── Remove bad entries ────────────────────────────────────────────────────
    if args.remove:
        console.rule("[bold]Removing bad entries from 1_generated[/bold]")
        total_removed = 0
        for path in files:
            removed, _ = remove_bad_file(path, dry)
            if removed:
                tag = "[yellow]DRY[/yellow] " if dry else ""
                console.print(f"  {tag}[cyan]{path.name}[/cyan]: [red]{removed}[/red] entries removed")
            total_removed += removed
        console.print(f"  Total removed: [red]{total_removed}[/red]")
        if not dry and total_removed:
            console.print("  [dim]Re-run generate_responses.py to regenerate them.[/dim]\n")
        else:
            console.print()

    # ── Sync labeled ─────────────────────────────────────────────────────────
    if args.sync_labeled:
        console.rule("[bold]Syncing broken-response entries out of 2b_labeled[/bold]")
        sync_labeled(data_dir, labeled_dir, args.model, dry, console)
        console.print()

    if not any_action:
        console.print("[dim](Audit only — pass --tag, --fix, --remove, or --sync-labeled to make changes.)[/dim]")


if __name__ == "__main__":
    main()
