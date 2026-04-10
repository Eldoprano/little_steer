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
    3. Fuzzy sliding-window match (SequenceMatcher ratio >= threshold)
- Sentences that fail all tiers are skipped (logged as warnings); the cursor
  does not advance, so subsequent sentences are still searched from the same
  position.
"""

from __future__ import annotations

from difflib import SequenceMatcher
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

    Window width varies from 70% to 130% of len(text). Returns the window
    with the highest SequenceMatcher ratio if >= threshold, else None.
    """
    text_len = len(text)
    if text_len == 0:
        return None

    region = haystack[start:end]
    if len(region) < text_len // 2:
        return None

    min_w = max(1, int(0.7 * text_len))
    max_w = int(1.3 * text_len)

    best_ratio = 0.0
    best_pos: tuple[int, int] | None = None

    for w in range(min_w, max_w + 1):
        if w > len(region):
            break
        for i in range(len(region) - w + 1):
            window = region[i : i + w]
            ratio = SequenceMatcher(None, text, window, autojunk=False).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = (start + i, start + i + w)

    if best_ratio >= threshold and best_pos is not None:
        return best_pos
    return None


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
