#!/usr/bin/env python3
"""
reparse_existing.py — Re-split reasoning/response in already-generated files.

The original generation run stored the full output (thinking block + final
answer) in the 'assistant' message because _parse_response only matched
<think>...</think> pairs. But Qwen3 with enable_thinking puts <think> in the
input prompt, so decoded output only contains </think> — no opening tag.

This script re-splits all existing JSONL files in-place without re-generating
anything. Safe to run multiple times (idempotent).

Usage:
    uv run python reparse_existing.py              # re-parse data/
    uv run python reparse_existing.py --dry-run    # preview only, no writes
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from thesis_schema import ConversationEntry


def _split(text: str) -> tuple[str | None, str]:
    """Same logic as generate_responses._parse_response."""
    m = re.search(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.search(r"(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, text.strip()


def reparse_file(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Re-parse one JSONL file. Returns (total, fixed) counts."""
    entries = []
    total = fixed = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            entry = ConversationEntry.model_validate_json(line)

            # Only touch entries where reasoning is missing but </think> exists
            # in the assistant message (i.e. the split was never done)
            has_reasoning = any(m["role"] == "reasoning" for m in entry.messages)
            assistant_idx = next(
                (i for i, m in enumerate(entry.messages) if m["role"] == "assistant"),
                None,
            )
            if has_reasoning or assistant_idx is None:
                entries.append(entry)
                continue

            assistant_text = entry.messages[assistant_idx]["content"]
            if "</think>" not in assistant_text:
                entries.append(entry)
                continue

            # Re-split
            reasoning, response = _split(assistant_text)
            new_messages = [m for m in entry.messages if m["role"] != "assistant"]
            # Insert reasoning before assistant, preserving message order
            insert_at = assistant_idx
            if reasoning:
                new_messages.insert(insert_at, {"role": "reasoning", "content": reasoning})
                insert_at += 1
            new_messages.insert(insert_at, {"role": "assistant", "content": response})

            entry = ConversationEntry(
                id=entry.id,
                messages=new_messages,
                annotations=entry.annotations,
                model=entry.model,
                judge=entry.judge,
                metadata={**entry.metadata, "has_reasoning": reasoning is not None},
            )
            fixed += 1
            entries.append(entry)

    if not dry_run and fixed > 0:
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for e in entries:
                f.write(e.model_dump_json() + "\n")
        tmp.replace(path)

    return total, fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-split reasoning/response in existing JSONL files.")
    parser.add_argument("--data-dir", default="data", help="Directory containing JSONL files")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        print(f"No JSONL files found in {data_dir}")
        return

    total_all = fixed_all = 0
    for f in files:
        total, fixed = reparse_file(f, dry_run=args.dry_run)
        status = "DRY RUN — " if args.dry_run else ""
        print(f"  {status}{f.name}: {fixed}/{total} entries re-split")
        total_all += total
        fixed_all += fixed

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Total: {fixed_all}/{total_all} entries updated")
    if args.dry_run and fixed_all > 0:
        print("Run without --dry-run to apply changes.")


if __name__ == "__main__":
    main()
