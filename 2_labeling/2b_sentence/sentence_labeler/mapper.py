"""mapper.py — Map LLM-returned sentence texts to character offsets.

Design:
- Sentences are assumed to appear sequentially in the reasoning text.
- A `search_from` cursor starts at 0 and advances only on successful matches.
  On a miss, the cursor is NOT advanced, preventing cascading failures.
- Each sentence is searched within a bounded window:
    window = [search_from, search_from + max_lookahead]
  where max_lookahead = max(max_lookahead_min, multiplier * len(sentence)).
- Three-tier fallback per sentence:
    1. Exact substring match
    2. Normalized match (strip whitespace from both)
    3. Fuzzy sliding-window match (rapidfuzz partial_ratio >= threshold)
- Sentences that fail all tiers are skipped (logged as warnings); the cursor
  does not advance, so subsequent sentences are still searched from the same
  position.
"""

from __future__ import annotations

from rapidfuzz import fuzz
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import MapperConfig, SentenceAnnotation

from thesis_schema import AnnotatedSpan


def _find_exact(text: str, haystack: str, start: int, end: int) -> tuple[int, int] | None:
    idx = haystack.find(text, start, end)
    if idx == -1:
        return None
    return idx, idx + len(text)


def _find_normalized(text: str, haystack: str, start: int, end: int) -> tuple[int, int] | None:
    stripped = text.strip()
    if not stripped:
        return None
    idx = haystack.find(stripped, start, end)
    if idx == -1:
        return None
    return idx, idx + len(stripped)


def _find_fuzzy(
    text: str,
    haystack: str,
    start: int,
    end: int,
    threshold: float,
) -> tuple[int, int] | None:
    """Sliding-window fuzzy match within haystack[start:end].

    Uses rapidfuzz's C++-backed partial_ratio_alignment which performs an
    optimized sliding-window search internally.  The threshold (0.0–1.0) is
    converted to rapidfuzz's 0–100 scale.
    """
    if len(text) == 0:
        return None

    region = haystack[start:end]
    if len(region) < len(text) // 2:
        return None

    # rapidfuzz uses 0–100 scale; our threshold is 0.0–1.0
    alignment = fuzz.partial_ratio_alignment(
        text, region, score_cutoff=threshold * 100,
    )
    if alignment is None:
        return None

    # alignment.dest_start / .dest_end are offsets within `region`;
    # translate back to `haystack` coordinates.
    return (start + alignment.dest_start, start + alignment.dest_end)

def _snap_to_sentence_end(haystack: str, start: int, end: int) -> int:
    """Return the position just after the last sentence boundary in haystack[start:end].

    Includes any closing quotes/brackets immediately following the terminal
    punctuation (e.g. ." or !' are consumed too).
    Falls back to `end` unchanged if no sentence boundary is found.
    """
    _TRAIL_CHARS = frozenset({')', '}', ']'})
    region = haystack[start:end]
    best = max(region.rfind(p) for p in '.!?')
    if best == -1:
        return end
    # Advance past any closing quote/bracket chars that follow the punctuation
    pos = best + 1
    while pos < len(region) and region[pos] in _TRAIL_CHARS:
        pos += 1
    return start + pos

def _lookahead(sentence_len: int, multiplier: int, minimum: int) -> int:
    return max(minimum, multiplier * sentence_len)


def map_sentences_to_spans(
    sentence_annotations: list[SentenceAnnotation],
    reasoning_text: str,
    message_idx: int,
    cfg: MapperConfig,
    warn_fn: callable | None = None,
) -> list[AnnotatedSpan]:
    """Convert SentenceAnnotation objects to AnnotatedSpan objects.

    Args:
        sentence_annotations: List of labeled sentences from the LLM.
        reasoning_text: The full reasoning message content.
        message_idx: Index of the reasoning message in the ConversationEntry.
        cfg: MapperConfig with fuzzy_threshold, max_lookahead_multiplier, max_lookahead_min.
        warn_fn: Optional callable(str) for logging warnings.

    Returns:
        List of AnnotatedSpan objects. Unmappable sentences are excluded.
    """
    spans: list[AnnotatedSpan] = []
    search_from = 0
    total = len(reasoning_text)

    for i, sent in enumerate(sentence_annotations):
        text = sent.text
        if not text.strip():
            continue

        lookahead = _lookahead(
            len(text),
            cfg.max_lookahead_multiplier,
            cfg.max_lookahead_min,
        )
        search_end = min(total, search_from + lookahead)
        # Snap the region end to the last sentence boundary so we don't
        # force the matcher to consider partial-sentence windows.
        search_end = _snap_to_sentence_end(reasoning_text, search_from, search_end) + 2

        result: tuple[int, int] | None = None

        # Tier 1: exact
        result = _find_exact(text, reasoning_text, search_from, search_end)

        # Tier 2: normalized
        if result is None:
            result = _find_normalized(text, reasoning_text, search_from, search_end)

        # Tier 3: fuzzy
        if result is None:
            result = _find_fuzzy(
                text, reasoning_text, search_from, search_end, cfg.fuzzy_threshold
            )

        if result is None:
            if warn_fn:
                preview = text[:60].replace("\n", " ")
                warn_fn(
                    f"[mapper] Sentence {i} not found within lookahead "
                    f"({lookahead} chars from pos {search_from}): {preview!r}"
                )
            # Do NOT advance search_from — preserve position for next sentence
            continue

        char_start, char_end = result
        # Advance cursor only on success
        search_from = char_end

        # Map found text back to the actual substring (fuzzy may differ from sent.text)
        actual_text = reasoning_text[char_start:char_end]

        spans.append(
            AnnotatedSpan(
                text=actual_text,
                message_idx=message_idx,
                char_start=char_start,
                char_end=char_end,
                labels=sent.labels,
                score=float(sent.safety_score),
                meta={
                    "judge_text": text,  # what the LLM said
                    "sentence_index": i,
                },
            )
        )

    return spans
