#!/usr/bin/env python3
"""
fix_quality.py — Audit and repair quality issues in the canonical dataset.

The canonical development store is ``data/dataset.jsonl``. This script:
  1. audits entries for known generation-quality issues
  2. stores findings in ``metadata.quality``
  3. mirrors the blocking decision to ``metadata.approved`` for legacy readers
  4. can fix assistant-side think-tag artifacts in place
  5. can remove blocking entries entirely so generation can recreate them

Usage:
    uv run python fix_quality.py
    uv run python fix_quality.py --tag
    uv run python fix_quality.py --fix
    uv run python fix_quality.py --remove
    uv run python fix_quality.py --tag --fix --remove

Filters:
    --dataset PATH   Path to canonical dataset (default: ../data/dataset.jsonl)
    --model NAME     Process only entries whose ``model`` contains NAME
    --dry-run        Print what would change without writing anything
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from thesis_schema import ConversationEntry  # noqa: E402


DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "data" / "dataset.jsonl"


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


# TODO: Foreign-script generation may reflect intentional model behaviour (e.g.
# reasoning in the training language) rather than a simple corruption artefact.
# Before treating this as blocking, investigate whether it correlates with
# specific models, datasets, or system prompts, and consider whether it should
# be flagged for review rather than auto-removed.
_FOREIGN_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF"   # Cyrillic
    r"\u0600-\u06FF"    # Arabic
    r"\u0900-\u097F"    # Devanagari
    r"\u3040-\u30FF"    # Hiragana / Katakana
    r"\u4E00-\u9FFF"    # CJK Unified Ideographs
    r"\uAC00-\uD7A3]"   # Korean Hangul
)
_FOREIGN_SCRIPT_THRESHOLD = 5


def _has_foreign_script(text: str) -> bool:
    return len(_FOREIGN_SCRIPT_RE.findall(text)) >= _FOREIGN_SCRIPT_THRESHOLD


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

    generation_meta = entry.metadata.get("generation") or {}
    finish_reason = entry.metadata.get("finish_reason") or generation_meta.get("finish_reason", "")
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

    for content in (reasoning_content, assistant_content):
        if content and _has_foreign_script(content):
            issues.add("foreign_script")
            break

    return issues


def _is_approved(entry: ConversationEntry) -> bool:
    blocking = {"think_artifact", "max_length", "repetition", "failed", "empty_response", "foreign_script"}
    return not _entry_quality_issues(entry).intersection(blocking)


def _quality_payload(entry: ConversationEntry) -> dict:
    issues = sorted(_entry_quality_issues(entry))
    return {
        "approved": not {
            "think_artifact",
            "max_length",
            "repetition",
            "failed",
            "empty_response",
            "foreign_script",
        }.intersection(issues),
        "issues": issues,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


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
    """Add/update canonical quality metadata on every entry. Returns number of changes."""
    entries = _load_entries(path)
    changed = 0
    for entry in entries:
        quality = _quality_payload(entry)
        if entry.metadata.get("quality") != quality or entry.metadata.get("approved") != quality["approved"]:
            entry.metadata["quality"] = quality
            entry.metadata["approved"] = quality["approved"]
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
            quality = _quality_payload(entry)
            entry.metadata["quality"] = quality
            entry.metadata["approved"] = quality["approved"]
            fixed += 1
    if fixed:
        _rewrite(path, entries, dry_run)
    return fixed


def remove_bad_file(path: Path, dry_run: bool) -> tuple[int, set[str]]:
    """Remove entries with blocking quality issues. Returns (removed_count, removed_ids)."""
    BLOCKING = {"max_length", "repetition", "failed", "missing_reasoning", "foreign_script"}
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

ISSUE_KEYS = ["ok", "think_artifact", "max_length", "repetition",
              "missing_reasoning", "empty_response", "failed", "foreign_script"]

# Short column headers to keep the table narrow enough for long filenames.
ISSUE_HEADERS = {
    "ok":                "ok",
    "think_artifact":    "artifact",
    "max_length":        "max_len",
    "repetition":        "repeat",
    "missing_reasoning": "no_reason",
    "empty_response":    "empty",
    "failed":            "failed",
    "foreign_script":    "foreign",
}

ISSUE_STYLES = {
    "ok":                "green",
    "think_artifact":    "yellow",
    "max_length":        "red",
    "repetition":        "red",
    "missing_reasoning": "magenta",
    "empty_response":    "magenta",
    "failed":            "bold red",
    "foreign_script":    "yellow",
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
    parser = argparse.ArgumentParser(
        description="Audit and fix quality issues in the canonical dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET),
        help="Path to canonical dataset.jsonl"
    )
    parser.add_argument("--model", default=None, help="Filter to entries whose model contains MODEL")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    parser.add_argument("--tag", action="store_true", help="Add/update metadata.quality on all entries")
    parser.add_argument("--fix", action="store_true", help="Fix think_artifacts in-place")
    parser.add_argument("--remove", action="store_true",
                        help="Delete bad canonical entries so generation can recreate them")
    args = parser.parse_args()

    console = Console(highlight=False)
    dataset_path = Path(args.dataset).resolve()
    dry = args.dry_run

    if dry:
        console.print("[bold yellow]DRY RUN[/bold yellow] — no files will be written.\n")

    if not dataset_path.exists():
        console.print(f"[red]Dataset not found:[/red] {dataset_path}")
        return

    files = [dataset_path]

    # ── Audit ─────────────────────────────────────────────────────────────────
    console.print(f"Auditing canonical dataset [dim]{dataset_path}[/dim]...\n")
    total_counts: Counter = Counter()
    file_reports: list[tuple[str, dict]] = []
    for path in files:
        counts = Counter()
        for entry in _load_entries(path):
            if args.model and args.model not in entry.model:
                continue
            issues = _entry_quality_issues(entry)
            for issue in issues:
                counts[issue] += 1
            if not issues:
                counts["ok"] += 1
        file_reports.append((path.name, counts))
        for k, v in counts.items():
            total_counts[k] += v

    console.print(_build_audit_table(file_reports, total_counts))

    any_action = args.tag or args.fix or args.remove

    # ── Tag ───────────────────────────────────────────────────────────────────
    if args.tag:
        console.rule("[bold]Tagging entries[/bold]")
        total_changed = 0
        for path in files:
            if args.model:
                entries = _load_entries(path)
                changed = 0
                for entry in entries:
                    if args.model not in entry.model:
                        continue
                    quality = _quality_payload(entry)
                    if entry.metadata.get("quality") != quality or entry.metadata.get("approved") != quality["approved"]:
                        entry.metadata["quality"] = quality
                        entry.metadata["approved"] = quality["approved"]
                        changed += 1
                if changed:
                    _rewrite(path, entries, dry)
            else:
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
            if args.model:
                entries = _load_entries(path)
                fixed = 0
                for entry in entries:
                    if args.model not in entry.model:
                        continue
                    assistant_msg = next((m for m in entry.messages if m["role"] == "assistant"), None)
                    if assistant_msg and _has_think_artifact(assistant_msg.get("content", "")):
                        content = assistant_msg["content"].lstrip()
                        for tag_text in ("[/THINK]", "</think>"):
                            if content.startswith(tag_text):
                                content = content[len(tag_text):].lstrip()
                                break
                        assistant_msg["content"] = content
                        quality = _quality_payload(entry)
                        entry.metadata["quality"] = quality
                        entry.metadata["approved"] = quality["approved"]
                        fixed += 1
                if fixed:
                    _rewrite(path, entries, dry)
            else:
                fixed = fix_artifacts_file(path, dry)
            if fixed:
                tag = "[yellow]DRY[/yellow] " if dry else ""
                console.print(f"  {tag}[cyan]{path.name}[/cyan]: {fixed} artifact(s) fixed")
            total_fixed += fixed
        console.print(f"  Total fixed: [green]{total_fixed}[/green]\n")

    # ── Remove bad entries ────────────────────────────────────────────────────
    if args.remove:
        console.rule("[bold]Removing bad entries from canonical dataset[/bold]")
        total_removed = 0
        for path in files:
            if args.model:
                entries = _load_entries(path)
                blocking = {"max_length", "repetition", "failed", "missing_reasoning", "foreign_script"}
                kept: list[ConversationEntry] = []
                removed = 0
                for entry in entries:
                    if args.model in entry.model and _entry_quality_issues(entry).intersection(blocking):
                        removed += 1
                    else:
                        kept.append(entry)
                if removed and not dry:
                    _rewrite(path, kept, dry)
            else:
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

    if not any_action:
        console.print("[dim](Audit only — pass --tag, --fix, or --remove to make changes.)[/dim]")


if __name__ == "__main__":
    main()
