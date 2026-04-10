"""
Data loading: JSONL discovery, parsing, and sub-text extraction.

Supports both eager loading (load_sub_texts) for backwards compat and
lazy loading (build_index + load_record_paragraphs) for fast startup.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from model_profiles import ModelProfile, get_profile

logger = logging.getLogger(__name__)

import os
_DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "1_generated"
DATA_DIR = Path(os.environ["LABEL_DATA_DIR"]) if "LABEL_DATA_DIR" in os.environ else _DEFAULT_DATA_DIR


@dataclass
class SubText:
    composite_id: str   # f"{model_stem}::{sample_id}" — unique across all models
    model_stem: str     # filename stem, e.g. "deepseek-r1-distill-llama-8b_xs_test"
    sample_id: str      # original JSONL record id
    sub_text_idx: int   # 0-based index within this sample
    text: str
    user_prompt: str    # the original user message that triggered the reasoning


# ── Index entry ───────────────────────────────────────────────────────────────

@dataclass
class RecordRef:
    composite_id: str
    stem: str
    sample_id: str
    line_num: int   # 0-based line number in the JSONL file


# ── Discovery helpers ─────────────────────────────────────────────────────────

def discover_models(data_dir: Path = DATA_DIR) -> list[str]:
    """Return sorted list of filename stems for all JSONL files in data_dir."""
    return sorted(p.stem for p in data_dir.glob("*.jsonl"))


def match_models(requested: list[str], available: list[str]) -> list[str]:
    """
    Return all available stems that contain at least one requested substring (case-insensitive).
    If requested is empty, return all available.
    """
    if not requested:
        return available
    result = []
    for stem in available:
        stem_lower = stem.lower()
        if any(r.lower() in stem_lower for r in requested):
            result.append(stem)
    return result


# ── Record extraction helpers ─────────────────────────────────────────────────

def extract_reasoning_text(record: dict) -> str:
    """
    Extract reasoning text from a JSONL record.
    Returns the content of role:'reasoning' message, or '' if not found.
    """
    for msg in record.get("messages", []):
        if msg.get("role") == "reasoning":
            content = msg.get("content", "")
            return content.strip()
    return ""


def _record_to_sub_texts(
    record: dict,
    composite_id: str,
    stem: str,
    sample_id: str,
    min_chars: int = 250,
) -> list[SubText]:
    """Parse a loaded record dict into a list of SubText paragraphs.

    Args:
        record: The raw JSONL record dict.
        composite_id: Unique composite ID for this record.
        stem: Model filename stem.
        sample_id: Original JSONL record ID.
        min_chars: Minimum number of characters a paragraph must have to be
            included. Paragraphs shorter than this are skipped (e.g. lone
            list items). Defaults to 250.
    """
    user_prompt = ""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            user_prompt = msg.get("content", "").strip()
            break

    reasoning_text = extract_reasoning_text(record)
    if not reasoning_text:
        return []

    profile = get_profile(stem)
    paragraphs = profile.split_fn(reasoning_text)

    filtered = [para for para in paragraphs if len(para) >= min_chars]
    # Fall back to all paragraphs if filtering would discard everything
    # (e.g. a record whose reasoning is entirely made up of short fragments).
    kept = filtered if filtered else paragraphs

    return [
        SubText(
            composite_id=composite_id,
            model_stem=stem,
            sample_id=sample_id,
            sub_text_idx=idx,
            text=para,
            user_prompt=user_prompt,
        )
        for idx, para in enumerate(kept)
    ]


# ── Lazy loading ──────────────────────────────────────────────────────────────

def build_index(model_stems: list[str], data_dir: Path = DATA_DIR) -> dict[str, RecordRef]:
    """
    Fast index build: scan JSONL files for record IDs only, skip records without reasoning.
    Returns {composite_id: RecordRef}. Much faster than load_sub_texts for large datasets.
    """
    index: dict[str, RecordRef] = {}

    for stem in model_stems:
        path = data_dir / f"{stem}.jsonl"
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue

        loaded = skipped = 0
        with path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("JSON parse error in %s line %d: %s", stem, line_num, e)
                    continue

                # Skip records without a reasoning role
                has_reasoning = any(
                    m.get("role") == "reasoning" for m in record.get("messages", [])
                )
                if not has_reasoning:
                    skipped += 1
                    continue

                sample_id = record.get("id", f"line{line_num}")
                composite_id = f"{stem}::{sample_id}"
                index[composite_id] = RecordRef(
                    composite_id=composite_id,
                    stem=stem,
                    sample_id=sample_id,
                    line_num=line_num,
                )
                loaded += 1

        logger.info("%s: indexed %d records (%d skipped no-reasoning)", stem, loaded, skipped)

    return index


def balance_index(
    index: dict[str, RecordRef],
    rng: random.Random,
) -> dict[str, RecordRef]:
    """
    Cap each dataset file (stem) to the size of the smallest dataset by randomly
    subsampling. This ensures equal sampling probability across datasets regardless
    of their original sizes.

    Args:
        index: The full record index from build_index.
        rng: Random number generator for reproducible sampling.

    Returns:
        A new index dict with exactly min_count records per stem.
    """
    by_stem: dict[str, list[str]] = {}
    for cid, ref in index.items():
        by_stem.setdefault(ref.stem, []).append(cid)

    if len(by_stem) <= 1:
        return index

    min_count = min(len(ids) for ids in by_stem.values())

    result: dict[str, RecordRef] = {}
    for stem, ids in by_stem.items():
        sampled = rng.sample(ids, min_count) if len(ids) > min_count else ids
        for cid in sampled:
            result[cid] = index[cid]
        if len(ids) > min_count:
            logger.info("%s: balanced %d → %d records (min across datasets)", stem, len(ids), min_count)

    return result


def load_record_paragraphs(
    ref: RecordRef,
    data_dir: Path = DATA_DIR,
    min_chars: int = 250,
) -> list[SubText]:
    """
    Load and split paragraphs for a single record, identified by its RecordRef.
    Returns empty list if the record cannot be loaded or has no reasoning text.

    Args:
        ref: Reference to the record to load.
        data_dir: Directory containing the JSONL files.
        min_chars: Minimum number of characters a paragraph must have to be
            included. Paragraphs shorter than this are skipped (e.g. lone
            list items). Defaults to 250.
    """
    path = data_dir / f"{ref.stem}.jsonl"
    if not path.exists():
        logger.warning("File not found for record %s: %s", ref.composite_id, path)
        return []

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i != ref.line_num:
                continue
            line = line.strip()
            if not line:
                return []
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("JSON parse error loading %s: %s", ref.composite_id, e)
                return []
            return _record_to_sub_texts(
                record, ref.composite_id, ref.stem, ref.sample_id, min_chars=min_chars
            )

    logger.warning("Line %d not found in %s", ref.line_num, path)
    return []


# ── Eager loading (kept for backwards compat / inspect command) ───────────────

def load_sub_texts(model_stems: list[str], data_dir: Path = DATA_DIR) -> list[SubText]:
    """
    Eagerly load all sub-texts from all JSONL files matching the given stems.
    Returns a flat list of SubText objects.
    Use build_index + load_record_paragraphs for faster lazy loading.
    """
    sub_texts: list[SubText] = []

    for stem in model_stems:
        path = data_dir / f"{stem}.jsonl"
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue

        profile = get_profile(stem)
        skipped_no_reasoning = 0
        skipped_short = 0
        loaded = 0

        with path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning("JSON parse error in %s line %d: %s", stem, line_num, e)
                    continue

                sample_id = record.get("id", f"line{line_num}")
                composite_id = f"{stem}::{sample_id}"

                user_prompt = ""
                for msg in record.get("messages", []):
                    if msg.get("role") == "user":
                        user_prompt = msg.get("content", "").strip()
                        break

                reasoning_text = extract_reasoning_text(record)
                if not reasoning_text:
                    skipped_no_reasoning += 1
                    continue

                paragraphs = profile.split_fn(reasoning_text)
                if not paragraphs:
                    skipped_short += 1
                    continue

                for idx, para in enumerate(paragraphs):
                    sub_texts.append(SubText(
                        composite_id=composite_id,
                        model_stem=stem,
                        sample_id=sample_id,
                        sub_text_idx=idx,
                        text=para,
                        user_prompt=user_prompt,
                    ))
                loaded += 1

        logger.info(
            "%s: loaded %d records, %d sub-texts | skipped: %d no-reasoning, %d all-short",
            stem, loaded, sum(1 for st in sub_texts if st.model_stem == stem),
            skipped_no_reasoning, skipped_short,
        )

    return sub_texts
