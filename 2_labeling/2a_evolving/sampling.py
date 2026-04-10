"""
Sampling: lazy, seed-reproducible sub-text selection.

sample_next_lazy uses a record-level index (build_index) and loads
paragraphs on demand, avoiding the need to pre-load all sub-texts.

Strategy:
  90% of the time: pick from records not yet visited (sample a random paragraph).
  10% of the time: revisit a previously seen record (prefer less-visited ones).
  If all records visited: pure revisit mode.
"""
from __future__ import annotations

import random

from data_loader import RecordRef, SubText, load_record_paragraphs
from state import RunState


def sample_next_lazy(
    index: dict[str, RecordRef],
    state: RunState,
    rng: random.Random,
    revisit_fraction: float = 0.10,
) -> tuple[SubText, SubText | None, bool]:
    """
    Lazy sampling: select a record, load it on demand, pick a random paragraph.

    Returns:
        (sub_text, prev_sub_text_or_None, is_revisit)
        prev_sub_text is the immediately preceding paragraph in the same record
        (None if the chosen paragraph is the first one).
    """
    all_ids = list(index.keys())
    processed_ids = set(state.processed.keys())
    unprocessed_ids = [cid for cid in all_ids if cid not in processed_ids]

    # Decide whether to revisit
    force_revisit = not unprocessed_ids
    do_revisit = force_revisit or (bool(processed_ids) and rng.random() < revisit_fraction)

    if do_revisit and processed_ids:
        composite_id = _pick_revisit(processed_ids, state, rng)
        is_revisit = True
    else:
        if not unprocessed_ids:
            raise RuntimeError("No sub-texts available to sample")
        composite_id = rng.choice(unprocessed_ids)
        is_revisit = False

    ref = index[composite_id]
    paragraphs = load_record_paragraphs(ref)
    if not paragraphs:
        raise RuntimeError(f"No paragraphs loaded for {composite_id}")

    chosen_idx = rng.randint(0, len(paragraphs) - 1)
    sub_text = paragraphs[chosen_idx]
    prev_sub_text = paragraphs[chosen_idx - 1] if chosen_idx > 0 else None

    return sub_text, prev_sub_text, is_revisit


def _pick_revisit(processed_ids: set[str], state: RunState, rng: random.Random) -> str:
    """Pick a processed composite_id, weighted by 1/(times a paragraph from it was processed)."""
    id_list = list(processed_ids)
    weights = []
    for cid in id_list:
        total = sum(
            e.get("times_processed", 1)
            for e in state.processed[cid].get("sub_texts", [])
        )
        weights.append(1.0 / max(total, 1))
    return rng.choices(id_list, weights=weights, k=1)[0]


# ── Legacy helpers (used by inspect command) ───────────────────────────────────

def _times_processed(sub_text: SubText, state: RunState) -> int:
    cid = sub_text.composite_id
    if cid not in state.processed:
        return 0
    for entry in state.processed[cid].get("sub_texts", []):
        if entry["idx"] == sub_text.sub_text_idx:
            return entry["times_processed"]
    return 0


def _is_processed(sub_text: SubText, state: RunState) -> bool:
    return _times_processed(sub_text, state) >= 1


def unprocessed_pool(all_sub_texts: list[SubText], state: RunState) -> list[SubText]:
    return [st for st in all_sub_texts if not _is_processed(st, state)]


def processed_pool(all_sub_texts: list[SubText], state: RunState) -> list[SubText]:
    return [st for st in all_sub_texts if _is_processed(st, state)]
