"""
little_steer.data.converter

Convert the existing JSON format (analysis_results_detailed_*.json) to
the new ConversationEntry format.

Old format structure:
  {
    "<model>_<type>_<prompt>": {
      "metadata": { "judge_model": ..., "model": ..., "type": ..., ... },
      "prompt": "...",
      "response_text": "<think>...\n</think>\n\n...",
      "judge_response": {
        "safety_decision_category": ...,
        "reasoning_usage_score": ...,
        "annotations": [{"categories": [...], "sentence_safety_score": ..., "sentence_text": ...}]
      },
      "results": {
        "extracted_segments": [
          {"text": ..., "start": ..., "end": ..., "categories": [...], "score": ...}
        ],
        "think_text": "...",   ← segments reference positions in THIS string
        "final_response": "...",
        "correctly_extracted": true
      }
    }
  }

New format: one ConversationEntry per key, where:
  - messages = [{"role": "user", "content": prompt},
                {"role": "reasoning", "content": think_text},   (if present)
                {"role": "assistant", "content": final_response}]
  - annotations = list of AnnotatedSpan with:
      message_idx = index of the "reasoning" message (1 if reasoning exists, else 0)
      char_start / char_end = positions from extracted_segments["start"/"end"]
                              (these are already relative to think_text)
"""

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Iterator

from tqdm.auto import tqdm

from thesis_schema import AnnotatedSpan, ConversationEntry


def convert_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    skip_incorrectly_extracted: bool = True,
    verbose: bool = True,
) -> list[ConversationEntry]:
    """Convert a JSON file from old format to a list of ConversationEntry.

    Args:
        input_path: Path to the source JSON file.
        output_path: If given, save the converted entries as JSONL.
        skip_incorrectly_extracted: Skip entries where correctly_extracted=False.
        verbose: Print progress and stats.

    Returns:
        List of ConversationEntry objects.
    """
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    entries: list[ConversationEntry] = []
    skipped = 0
    errors = 0

    for key, entry_data in tqdm(raw_data.items(), desc="Converting entries", unit="entry", disable=not verbose):
        try:
            result = entry_data.get("results", {})

            # Skip entries that weren't correctly extracted
            if skip_incorrectly_extracted and not result.get("correctly_extracted", True):
                skipped += 1
                continue

            entry = _convert_entry(key, entry_data)
            if entry is not None:
                entries.append(entry)
            else:
                skipped += 1

        except Exception as e:
            if verbose:
                tqdm.write(f"  ⚠️  Error converting '{key[:60]}': {e}")
            errors += 1

    if verbose:
        print(f"✅ Converted {len(entries)} entries "
              f"({skipped} skipped, {errors} errors) from {input_path.name}")

    if output_path is not None:
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(entry.model_dump_json() + "\n")
        if verbose:
            print(f"💾 Saved to {output_path}")

    return entries


def _convert_entry(key: str, data: dict) -> ConversationEntry | None:
    """Convert a single old-format entry to ConversationEntry."""
    meta = data.get("metadata", {})
    results = data.get("results", {})

    prompt = data.get("prompt", "")
    think_text = results.get("think_text", "")
    final_response = results.get("final_response", "")
    extracted_segments = results.get("extracted_segments", [])

    if not prompt:
        return None

    # Build messages list
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]

    reasoning_message_idx: int | None = None
    if think_text:
        reasoning_message_idx = len(messages)
        messages.append({"role": "reasoning", "content": think_text})

    if final_response:
        messages.append({"role": "assistant", "content": final_response})

    # Build annotations from extracted_segments
    # Segment start/end are relative to think_text (or full response if no think)
    annotations: list[AnnotatedSpan] = []

    for seg in extracted_segments:
        text = seg.get("text", "")
        start = seg.get("start", 0)
        end = seg.get("end", len(text))
        categories = seg.get("categories", [])
        score = seg.get("score", 0.0)

        if not categories or not text:
            continue

        # Determine which message this annotation belongs to
        if reasoning_message_idx is not None:
            msg_idx = reasoning_message_idx
        else:
            # No reasoning: annotations point to... user message? This is unusual.
            # Fallback to message 0.
            msg_idx = 0

        annotations.append(
            AnnotatedSpan(
                text=text,
                message_idx=msg_idx,
                char_start=start,
                char_end=end,
                labels=categories,
                score=float(score),
                meta={
                    "start_phrase": seg.get("start_phrase", ""),
                    "end_phrase": seg.get("end_phrase", ""),
                },
            )
        )

    # Build metadata
    entry_metadata = {
        "dataset_source": _infer_source(key),
        "type": meta.get("type", ""),
        "template_type": meta.get("template_type", ""),
        "processed_at": meta.get("processed_at", ""),
        "safety_decision_category": data.get("judge_response", {}).get(
            "safety_decision_category", ""
        ),
        "reasoning_usage_score": data.get("judge_response", {}).get(
            "reasoning_usage_score", None
        ),
        "original_key": key,
    }

    # Generate a stable ID from the key
    entry_id = hashlib.md5(key.encode()).hexdigest()[:12]

    return ConversationEntry(
        id=entry_id,
        messages=messages,
        annotations=annotations,
        model=meta.get("model", "unknown"),
        judge=meta.get("judge_model", "unknown"),
        metadata=entry_metadata,
    )


def _infer_source(key: str) -> str:
    """Infer dataset source from the entry key."""
    key_lower = key.lower()
    if "harmbench" in key_lower:
        return "harmbench"
    if "strong_reject" in key_lower or "docent" in key_lower:
        return "strong_reject"
    return "unknown"


def load_dataset(path: str | Path) -> list[ConversationEntry]:
    """Load a JSONL file of ConversationEntry objects."""
    path = Path(path)
    entries: list[ConversationEntry] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Loading dataset", unit="entry"):
        line = line.strip()
        if line:
            entries.append(ConversationEntry.model_validate_json(line))
    return entries


def save_dataset(entries: list[ConversationEntry], path: str | Path) -> None:
    """Save a list of ConversationEntry objects to JSONL."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.model_dump_json() + "\n")


def iter_dataset(path: str | Path) -> Iterator[ConversationEntry]:
    """Lazily iterate over a JSONL dataset file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ConversationEntry.model_validate_json(line)
