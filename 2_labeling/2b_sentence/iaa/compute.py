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

MODES = ("score", "primary_label", "label_set", "any_overlap")


def filter_stable_labels(entry_data: EntryData) -> EntryData:
    """For annotators with a _pass2 counterpart, keep only stable spans/labels.

    A span is kept when:
    - At least one pass2 span overlaps it, AND
    - The overlap-weighted majority pass2 score matches pass1's score.

    Labels are filtered to those appearing in at least one overlapping pass2 span.
    Spans with no remaining labels are dropped.
    Annotators without a _pass2 counterpart are left unchanged.
    """
    all_anns: set[str] = set()
    for by_ann in entry_data.values():
        all_anns.update(by_ann.keys())

    pass_map = {j: j + "_pass2" for j in all_anns if j + "_pass2" in all_anns}
    if not pass_map:
        return entry_data

    result: EntryData = {}
    for key, by_ann in entry_data.items():
        new_by_ann: dict[str, list[Span]] = {}
        for ann, spans in by_ann.items():
            p2_name = pass_map.get(ann)
            if p2_name is None or p2_name not in by_ann:
                new_by_ann[ann] = spans
                continue

            spans2 = by_ann[p2_name]
            stable: list[Span] = []
            for sp1 in spans:
                s1, e1, score1, lbls1 = sp1
                p2_lbls: set[str] = set()
                score_weight: dict[int, float] = {}
                for sp2 in spans2:
                    if sp2[0] >= e1:
                        break
                    if sp2[1] > s1:
                        lo = max(s1, sp2[0])
                        hi = min(e1, sp2[1])
                        score_weight[sp2[2]] = score_weight.get(sp2[2], 0.0) + hi - lo
                        p2_lbls.update(sp2[3])
                if not score_weight:
                    continue
                if max(score_weight, key=score_weight.get) != score1:
                    continue
                stable_lbls = tuple(l for l in lbls1 if l in p2_lbls)
                if stable_lbls:
                    stable.append((s1, e1, score1, stable_lbls))

            new_by_ann[ann] = stable
        result[key] = new_by_ann
    return result


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
    if mode == "any_overlap":
        la, lb = span_a[3], span_b[3]
        if not la or not lb:
            return None
        # Shared label → both get same sentinel (= agreement); no shared label →
        # keep their distinct frozensets so the nominal distance stays 1.
        if set(la) & set(lb):
            return "__overlap__", "__overlap__"
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


def compute_substitution_matrix(
    entry_data: EntryData,
    annotators: list[str],
) -> dict[tuple[str, str], int]:
    """Cross-annotator label substitution frequencies.

    For each pair of overlapping spans where label sets differ, records how
    often each (label_from_A, label_from_B) pair appears as a substitute.
    Result is symmetric: (a, b) and (b, a) are merged into tuple(sorted(...)).
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for by_ann in entry_data.values():
        present = [a for a in annotators if a in by_ann]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                for _, _, sp_a, sp_b in _interval_intersect(by_ann[present[i]], by_ann[present[j]]):
                    extra_a = set(sp_a[3]) - set(sp_b[3])
                    extra_b = set(sp_b[3]) - set(sp_a[3])
                    for la in extra_a:
                        for lb in extra_b:
                            counts[tuple(sorted([la, lb]))] += 1
    return dict(counts)


def normalize_labels(entry_data: EntryData, label_map: dict[str, str]) -> EntryData:
    """Remap labels through label_map before IAA computation.

    Labels not in label_map pass through unchanged. If two labels in a span
    map to the same canonical form, duplicates are removed (order preserved).
    """
    if not label_map:
        return entry_data
    result: EntryData = {}
    for key, by_ann in entry_data.items():
        new_by_ann: dict[str, list[Span]] = {}
        for ann, spans in by_ann.items():
            new_spans: list[Span] = []
            for s, e, score, labels in spans:
                new_labels = tuple(dict.fromkeys(label_map.get(l, l) for l in labels))
                if new_labels:
                    new_spans.append((s, e, score, new_labels))
            new_by_ann[ann] = new_spans
        result[key] = new_by_ann
    return result


def count_shared_all(entry_data: EntryData, annotators: list[str]) -> int:
    """Count entries where ALL given annotators have at least one span."""
    if not annotators:
        return 0
    ann_set = set(annotators)
    return sum(
        1 for by_ann in entry_data.values()
        if ann_set.issubset(by_ann.keys())
    )


def compute_krippendorff_alpha(
    entry_data: EntryData,
    annotators: list[str],
    mode: str = "score",
) -> dict:
    """Compute Krippendorff's alpha across all selected annotators.

    Character-weighted: each shared character position contributes equally.
    Handles missing data naturally — units not labeled by both annotators
    in a pair are simply skipped.

    Distance functions:
    - "score": ordinal (k − l)² on values {−1, 0, 1}; appropriate for an
      ordered scale. Thesis note: with ~25 human annotations expect wide
      confidence intervals — report alpha AND n_coincidences.
    - "primary_label" / "label_set": nominal (0 if equal, 1 if not).

    Preferred over pairwise Cohen's kappa when there are 3+ annotators,
    because it produces a single summary statistic and handles missing data
    without requiring a complete all-pairs annotation design.

    Returns dict with keys: alpha, n_coincidences, total_chars, mode.
    alpha is None when there are fewer than 2 annotators with shared data.
    """
    if mode not in MODES:
        mode = "score"

    # Collect (val_a, val_b, weight) coincidence pairs from all annotator pairs
    all_pairs: list[tuple] = []
    for by_ann in entry_data.values():
        present = [a for a in annotators if a in by_ann]
        if len(present) < 2:
            continue
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a_name, b_name = present[i], present[j]
                for lo, hi, span_a, span_b in _interval_intersect(by_ann[a_name], by_ann[b_name]):
                    cats = _span_categories(span_a, span_b, mode)
                    if cats is None:
                        continue
                    all_pairs.append((*cats, hi - lo))

    if not all_pairs:
        return {"alpha": None, "n_coincidences": 0, "total_chars": 0, "mode": mode}

    # Value vocabulary
    all_vals = list(dict.fromkeys(c for ca, cb, _ in all_pairs for c in (ca, cb)))
    val_idx = {v: i for i, v in enumerate(all_vals)}
    n_v = len(all_vals)

    if mode == "score":
        def dist(a, b): return float((a - b) ** 2)
    else:
        def dist(a, b): return 0.0 if a == b else 1.0

    # Build coincidence matrix — count both orderings for each pair (Krippendorff convention).
    # Diagonal entries are doubled (one count per direction) so marginals stay consistent.
    C = [[0.0] * n_v for _ in range(n_v)]
    for ca, cb, w in all_pairs:
        i, j = val_idx[ca], val_idx[cb]
        C[i][j] += w
        C[j][i] += w  # symmetric — counts (A→B) and (B→A) separately

    n = sum(C[i][j] for i in range(n_v) for j in range(n_v))
    if n < 2:
        return {"alpha": None, "n_coincidences": len(all_pairs), "total_chars": 0, "mode": mode}

    n_k = [sum(C[i]) for i in range(n_v)]  # marginals

    d_o = sum(C[i][j] * dist(all_vals[i], all_vals[j])
              for i in range(n_v) for j in range(n_v)) / n
    d_e = sum(n_k[i] * n_k[j] * dist(all_vals[i], all_vals[j])
              for i in range(n_v) for j in range(n_v)) / (n * (n - 1))

    alpha = 1.0 if d_e < 1e-9 else 1.0 - d_o / d_e

    return {
        "alpha": round(alpha, 4),
        "n_coincidences": len(all_pairs),
        "total_chars": int(sum(w for _, _, w in all_pairs)),
        "mode": mode,
    }
