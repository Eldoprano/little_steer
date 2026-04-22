"""IAA computation for sentence-level reasoning annotation.

Agreement is computed at character level using interval intersection.
This naturally handles the case where two annotators split text into
different numbers of sentences: only the overlapping character regions
are compared, regardless of where each annotator drew boundaries.

Three comparison modes:
  "score"         — compare the scalar score field (-1 / 0 / 1)
  "primary_label" — compare only labels[0] (the most important label)
  "label_set"     — compare the full label set (order-invariant)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

# (start, end, score, labels_tuple) — compact in-memory representation
Span = tuple[int, int, int, tuple[str, ...]]
# entry_key -> annotator_name -> sorted list of Span
EntryData = dict[str, dict[str, list[Span]]]
AnnotatorCounts = dict[str, int]

MODES = ("score", "primary_label", "label_set")


def _extract_spans(run: dict, reasoning_idx: int) -> list[Span]:
    """Return sorted (start, end, score, labels) tuples for the reasoning message."""
    spans: list[Span] = []
    for s in run.get("spans") or []:
        if s.get("message_idx") != reasoning_idx:
            continue
        score = s.get("score")
        start = s.get("char_start", 0)
        end = s.get("char_end", 0)
        labels = tuple(s.get("labels") or [])
        if score is not None and start < end:
            spans.append((start, end, int(float(score)), labels))
    return sorted(spans, key=lambda x: x[0])


def load_iaa_data(dataset_path: Path) -> tuple[EntryData, AnnotatorCounts]:
    """Stream dataset.jsonl and build compact span data for IAA.

    For entries where the same annotator has multiple label runs
    (e.g., different taxonomy versions), the most recent run is used.
    """
    # entry_key -> judge_name -> (labeled_at, [Span])
    raw: dict[str, dict[str, tuple[str, list[Span]]]] = defaultdict(dict)
    annotator_sets: dict[str, set[str]] = defaultdict(set)

    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_id = entry.get("id", "")
            if not entry_id:
                continue

            messages = entry.get("messages") or []
            reasoning_idx = next(
                (i for i, m in enumerate(messages) if m.get("role") == "reasoning"),
                None,
            )
            if reasoning_idx is None:
                continue

            gen_hash = (entry.get("metadata") or {}).get("generation_hash", "")
            key = f"{entry_id}::{gen_hash}"

            for run in entry.get("label_runs") or []:
                judge = run.get("judge_name", "")
                if not judge:
                    continue
                spans = _extract_spans(run, reasoning_idx)
                if not spans:
                    continue

                labeled_at = run.get("labeled_at", "")
                existing = raw[key].get(judge)
                # Keep most recent run (lexicographic timestamp comparison)
                if existing is None or labeled_at >= existing[0]:
                    raw[key][judge] = (labeled_at, spans)
                    annotator_sets[judge].add(key)

    # Flatten to final structure
    entry_data: EntryData = {
        key: {judge: ts_spans[1] for judge, ts_spans in by_judge.items()}
        for key, by_judge in raw.items()
    }
    annotator_counts: AnnotatorCounts = {
        name: len(keys) for name, keys in annotator_sets.items()
    }
    return entry_data, annotator_counts


def _interval_intersect(
    a: list[Span], b: list[Span]
) -> list[tuple[int, int, Span, Span]]:
    """Sweep-line interval intersection returning (start, end, span_a, span_b).

    Works correctly even when one interval in A spans multiple intervals in B
    (the "1 sentence vs 2 sentences" case).  Assumes each list is sorted and
    non-overlapping within itself.
    """
    result: list[tuple[int, int, Span, Span]] = []
    ia = ib = 0
    while ia < len(a) and ib < len(b):
        lo = max(a[ia][0], b[ib][0])
        hi = min(a[ia][1], b[ib][1])
        if lo < hi:
            result.append((lo, hi, a[ia], b[ib]))
        if a[ia][1] <= b[ib][1]:
            ia += 1
        else:
            ib += 1
    return result


def _span_categories(
    span_a: Span, span_b: Span, mode: str
) -> tuple | None:
    """Return (cat_a, cat_b) for the given comparison mode, or None to skip."""
    if mode == "score":
        return span_a[2], span_b[2]
    if mode == "primary_label":
        la, lb = span_a[3], span_b[3]
        if not la or not lb:
            return None
        return la[0], lb[0]
    if mode == "label_set":
        la, lb = span_a[3], span_b[3]
        if not la or not lb:
            return None
        return frozenset(la), frozenset(lb)
    return None


def _kappa_from_pairs(
    pairs: list[tuple],  # (cat_a, cat_b, length)
    shared: int,
    total: float,
) -> dict:
    """Compute Cohen's kappa from weighted (cat_a, cat_b, length) pairs."""
    if total == 0 or shared == 0:
        return {"kappa": None, "pct": None, "shared": shared, "chars": 0}

    # Build dynamic category index
    cats: list = list(dict.fromkeys(c for a, b, _ in pairs for c in (a, b)))
    idx = {c: i for i, c in enumerate(cats)}
    n = len(cats)

    conf = [[0.0] * n for _ in range(n)]
    for ca, cb, length in pairs:
        conf[idx[ca]][idx[cb]] += length

    agreed = sum(conf[i][i] for i in range(n))
    pct = agreed / total
    row_sums = [sum(conf[i]) / total for i in range(n)]
    col_sums = [sum(conf[j][i] for j in range(n)) / total for i in range(n)]
    p_e = sum(row_sums[i] * col_sums[i] for i in range(n))
    denom = 1.0 - p_e
    kappa = (pct - p_e) / denom if denom > 1e-9 else 1.0

    return {
        "kappa": round(kappa, 4),
        "pct": round(pct, 4),
        "shared": shared,
        "chars": int(total),
    }


def _pairwise_kappa(
    entry_data: EntryData, ann_a: str, ann_b: str, mode: str = "score"
) -> dict:
    """Compute Cohen's kappa between two annotators (character-weighted).

    Only characters covered by BOTH annotators are included — this avoids
    penalising annotators for different coverage extents and focuses the
    comparison on actual labelling decisions.
    """
    pairs: list[tuple] = []
    total = 0.0
    shared = 0

    for by_ann in entry_data.values():
        if ann_a not in by_ann or ann_b not in by_ann:
            continue
        shared += 1
        for lo, hi, span_a, span_b in _interval_intersect(by_ann[ann_a], by_ann[ann_b]):
            cats = _span_categories(span_a, span_b, mode)
            if cats is None:
                continue
            length = hi - lo
            pairs.append((*cats, length))
            total += length

    return _kappa_from_pairs(pairs, shared, total)


def compute_agreement_matrix(
    entry_data: EntryData,
    annotators: list[str],
    mode: str = "score",
    pair_cache: dict | None = None,
) -> dict:
    """Compute full pairwise agreement matrix for selected annotators."""
    if pair_cache is None:
        pair_cache = {}
    if mode not in MODES:
        mode = "score"

    pairs: dict[str, dict] = {}
    for i, a in enumerate(annotators):
        for j, b in enumerate(annotators):
            key = f"{a}||{b}"
            if i == j:
                pairs[key] = {"kappa": 1.0, "pct": 1.0, "shared": None, "chars": 0}
            elif j > i:
                cache_key = f"{a}||{b}||{mode}"
                if cache_key not in pair_cache:
                    pair_cache[cache_key] = _pairwise_kappa(entry_data, a, b, mode)
                r = pair_cache[cache_key]
                pairs[key] = r
                pairs[f"{b}||{a}"] = r

    return {"annotators": annotators, "pairs": pairs, "mode": mode}


def count_shared_all(entry_data: EntryData, annotators: list[str]) -> int:
    """Count entries where ALL given annotators have at least one span."""
    if not annotators:
        return 0
    ann_set = set(annotators)
    return sum(
        1 for by_ann in entry_data.values()
        if ann_set.issubset(by_ann.keys())
    )
