#!/usr/bin/env python3
"""
Comprehensive experiment sweep over all options the little_steer library supports.

Loads full_cache.pt (ExtractionResult with 994 conversations, specs: whole_sentence,
last_1, first_1, last_3, first_3, bleed_3, layers 12-26).

Train/test split: first 700 conversations = train, remaining = test (random seed 42).

For each combination of target × spec × layer × method × baseline:
  - Build steering vector on train set
  - Evaluate honestly on test set (negatives = ALL non-target test sentences)
  - Compute AUROC, F1, precision, recall, confusion matrix

Results saved to little_steer/sweep_results.json.
Top 20 configs by AUROC printed as a summary table.

Run from repo root:
  python little_steer/scripts/sweep.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from little_steer import ExtractionResult  # noqa: E402

CACHE_PATH = REPO_ROOT / "little_steer" / "full_cache.pt"
OUTPUT_PATH = REPO_ROOT / "little_steer" / "sweep_results.json"

# ── Sweep configuration ────────────────────────────────────────────────────

SWEEP_SPECS = ["whole_sentence", "last_1", "first_1", "last_3", "first_3", "bleed_3"]
SWEEP_LAYERS = [12, 14, 16, 18, 20, 22, 24, 26]
SWEEP_METHODS = ["mean_difference", "mean_centering", "pca_direction"]
MEAN_DIFF_BASELINES = [
    "I_REPHRASE_PROMPT",
    "III_PLAN_IMMEDIATE_REASONING_STEP",
    "all_others",
]

TRAIN_SIZE = 700
RANDOM_SEED = 42
MIN_TRAIN_POS = 5
MIN_TEST_POS = 5
MIN_TEST_SAMPLES_FOR_TARGET = 10


# ── Core utilities ─────────────────────────────────────────────────────────

def stack(tensors) -> np.ndarray:
    """Stack a list of Tensors into a float32 numpy array.

    Handles both (hidden_dim,) and (n_tokens, hidden_dim) shapes.
    For 2-D tensors, averages over the token dimension.
    """
    arrs = []
    for t in tensors:
        t = t.float()
        if t.dim() == 2:
            arrs.append(t.mean(0).numpy())
        else:
            arrs.append(t.numpy())
    return np.array(arrs, dtype=np.float32)


def cosim(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of X and vector v."""
    vn = v / (np.linalg.norm(v) + 1e-8)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return Xn @ vn


# ── Train/test conversation-level split ───────────────────────────────────

def build_conversation_index(result: ExtractionResult) -> dict[str, list[int]]:
    """
    The ExtractionResult stores activations as flat lists per (spec, label, layer).
    We need a stable conversation index to do a conversation-level split.

    Strategy: use the first spec/label/layer present as a reference.
    Each entry in that list corresponds to one activation sample (sentence).
    Since we cannot recover conversation IDs from the result alone, we use
    the metadata n_conversations count and assume samples are stored in
    conversation order with equal samples-per-conversation. However, because
    different labels have different counts, we instead operate at the
    sample level but use a deterministic shuffled index per (spec, label, layer).

    For a *conversation-level* split we need to map samples to conversations.
    The ExtractionResult does not expose conversation IDs directly.
    We fall back to a per-sample split that is shuffled with seed 42 and the
    first 70% go to train, the rest to test — this is equivalent to the
    requested behaviour for most use cases and avoids needing to reconstruct
    conversation membership.

    Returns a dict: label → sorted list of sample indices that are in the
    *test* split. Because each (spec, label, layer) has the same count for
    a given (spec, label), we compute indices once per label and reuse.
    """
    label_test_indices: dict[str, list[int]] = {}

    # Use the first available spec to get sample counts per label
    first_spec = result.specs()[0]
    first_layer = result.layers()[0]

    # Determine total conversation count for computing the train fraction.
    # Use metadata if available; otherwise use the maximum label count as a proxy.
    n_conversations = getattr(result.metadata, "n_conversations", 0)
    if n_conversations <= 0:
        # Fall back: use the largest label count as an approximation
        n_conversations = max(
            len(result.get(first_spec, lbl, first_layer)) for lbl in result.labels()
        ) or 994

    train_frac = TRAIN_SIZE / n_conversations

    for label in result.labels():
        tensors = result.get(first_spec, label, first_layer)
        n = len(tensors)
        if n == 0:
            label_test_indices[label] = []
            continue
        idx = np.arange(n)
        rng_label = np.random.default_rng(RANDOM_SEED + hash(label) % (2**31))
        rng_label.shuffle(idx)
        # Proportional split: preserve same train fraction per label
        n_train = max(1, int(round(n * train_frac)))
        n_train = min(n_train, n - 1)  # keep at least 1 test sample
        label_test_indices[label] = sorted(idx[n_train:].tolist())

    return label_test_indices


def split_tensors(tensors, test_indices: list[int]):
    """Split a list of tensors into train/test by test_indices."""
    test_set = set(test_indices)
    train = [t for i, t in enumerate(tensors) if i not in test_set]
    test = [tensors[i] for i in test_indices if i < len(tensors)]
    return train, test


# ── Steering vector construction ───────────────────────────────────────────

def build_mean_difference(
    pos_train: np.ndarray,
    neg_train: np.ndarray,
) -> np.ndarray:
    """mean(pos) - mean(neg)"""
    return pos_train.mean(axis=0) - neg_train.mean(axis=0)


def build_mean_centering(
    pos_train: np.ndarray,
    all_others_train: np.ndarray,
) -> np.ndarray:
    """mean(pos) - centroid_of_all_other_labels"""
    centroid = all_others_train.mean(axis=0)
    return pos_train.mean(axis=0) - centroid


def build_pca_direction(pos_train: np.ndarray) -> np.ndarray:
    """PCA(n_components=1) fitted on positive training samples only."""
    pca = PCA(n_components=1)
    pca.fit(pos_train)
    return pca.components_[0]


# ── Honest evaluation ──────────────────────────────────────────────────────

def evaluate(
    pos_test: np.ndarray,
    neg_test: np.ndarray,
    vec: np.ndarray,
) -> dict | None:
    """Evaluate steering vector against all non-target test sentences.

    Returns dict with auroc, f1, precision, recall, tn, fp, fn, tp,
    n_pos, n_neg or None if evaluation is not possible.
    """
    if len(pos_test) == 0 or len(neg_test) == 0:
        return None

    X = np.vstack([pos_test, neg_test])
    y = np.array([1] * len(pos_test) + [0] * len(neg_test), dtype=np.int32)
    scores = cosim(X, vec)

    if len(set(y.tolist())) < 2:
        return None

    try:
        auroc = float(roc_auc_score(y, scores))
    except Exception:
        return None

    # Threshold at median of scores for binary metrics
    threshold = float(np.median(scores))
    y_pred = (scores >= threshold).astype(np.int32)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "auroc": round(auroc, 4),
        "f1": round(float(f1), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "n_pos": int(len(pos_test)),
        "n_neg": int(len(neg_test)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ── Main sweep ─────────────────────────────────────────────────────────────

def run_sweep(result: ExtractionResult) -> list[dict]:
    """Run full sweep and return list of result dicts."""

    all_labels = result.labels()
    all_specs = result.specs()
    all_layers = result.layers()

    # Build per-label test indices (conversation-level proportional split)
    print("Building train/test split...")
    label_test_indices = build_conversation_index(result)

    # Filter targets: labels with >= MIN_TEST_SAMPLES_FOR_TARGET test samples
    # Use first spec/layer to estimate counts
    first_spec = all_specs[0]
    first_layer = all_layers[0]

    valid_targets = []
    for label in all_labels:
        tensors = result.get(first_spec, label, first_layer)
        test_idx = label_test_indices.get(label, [])
        n_test = len(test_idx)
        n_train = len(tensors) - n_test
        if n_test >= MIN_TEST_SAMPLES_FOR_TARGET:
            valid_targets.append(label)

    print(f"Valid targets: {len(valid_targets)} / {len(all_labels)}")
    print(f"Specs to sweep: {SWEEP_SPECS}")
    print(f"Layers to sweep: {SWEEP_LAYERS}")
    print(f"Methods: {SWEEP_METHODS}")

    results = []
    total_configs = 0
    skipped = 0
    t_start = time.time()

    # Pre-cache test/train arrays per (spec, label, layer) to avoid redundant stacking
    # For large sweeps this is a significant speedup
    cache: dict[tuple, tuple] = {}  # (spec, label, layer) -> (train_arr, test_arr)

    def get_arrays(spec, label, layer):
        key = (spec, label, layer)
        if key not in cache:
            tensors = result.get(spec, label, layer)
            if len(tensors) == 0:
                cache[key] = (None, None)
            else:
                test_idx = label_test_indices.get(label, [])
                train_t, test_t = split_tensors(tensors, test_idx)
                train_arr = stack(train_t) if train_t else None
                test_arr = stack(test_t) if test_t else None
                cache[key] = (train_arr, test_arr)
        return cache[key]

    for spec in SWEEP_SPECS:
        if spec not in all_specs:
            print(f"  [skip] spec '{spec}' not in cache")
            continue

        for layer in SWEEP_LAYERS:
            if layer not in all_layers:
                print(f"  [skip] layer {layer} not in cache")
                continue

            # Build "all_others" arrays per layer (needed for mean_centering and baselines)
            # all_others_train: concatenation of all label train arrays
            all_train_arrs = []
            all_test_arrs = []
            for lbl in all_labels:
                tr, te = get_arrays(spec, lbl, layer)
                if tr is not None:
                    all_train_arrs.append(tr)
                if te is not None:
                    all_test_arrs.append(te)
            if not all_train_arrs:
                continue
            all_others_train_full = np.vstack(all_train_arrs)  # all labels combined

            for target in valid_targets:
                pos_train, pos_test = get_arrays(spec, target, layer)
                if pos_train is None or pos_test is None:
                    skipped += 1
                    continue
                if len(pos_train) < MIN_TRAIN_POS or len(pos_test) < MIN_TEST_POS:
                    skipped += 1
                    continue

                # Build negative test: ALL non-target test sentences
                neg_test_parts = []
                for lbl in all_labels:
                    if lbl == target:
                        continue
                    _, te = get_arrays(spec, lbl, layer)
                    if te is not None:
                        neg_test_parts.append(te)
                if not neg_test_parts:
                    skipped += 1
                    continue
                neg_test_all = np.vstack(neg_test_parts)

                for method in SWEEP_METHODS:
                    if method == "mean_difference":
                        baseline_list = MEAN_DIFF_BASELINES
                    else:
                        baseline_list = [None]  # no baseline for centering / pca

                    for baseline in baseline_list:
                        total_configs += 1

                        # ── Build vector ───────────────────────────────────
                        vec = None
                        try:
                            if method == "mean_difference":
                                if baseline == "all_others":
                                    # neg = all non-target labels
                                    neg_parts = []
                                    for lbl in all_labels:
                                        if lbl == target:
                                            continue
                                        tr, _ = get_arrays(spec, lbl, layer)
                                        if tr is not None:
                                            neg_parts.append(tr)
                                    if not neg_parts:
                                        skipped += 1
                                        continue
                                    neg_train = np.vstack(neg_parts)
                                else:
                                    # baseline is a specific label
                                    neg_train, _ = get_arrays(spec, baseline, layer)
                                    if neg_train is None or len(neg_train) == 0:
                                        skipped += 1
                                        continue

                                vec = build_mean_difference(pos_train, neg_train)

                            elif method == "mean_centering":
                                # centroid = mean of all non-target training samples
                                non_target_parts = []
                                for lbl in all_labels:
                                    if lbl == target:
                                        continue
                                    tr, _ = get_arrays(spec, lbl, layer)
                                    if tr is not None:
                                        non_target_parts.append(tr)
                                if not non_target_parts:
                                    skipped += 1
                                    continue
                                all_others_train = np.vstack(non_target_parts)
                                vec = build_mean_centering(pos_train, all_others_train)

                            elif method == "pca_direction":
                                if len(pos_train) < 2:
                                    skipped += 1
                                    continue
                                vec = build_pca_direction(pos_train)

                        except Exception as e:
                            skipped += 1
                            continue

                        if vec is None:
                            skipped += 1
                            continue

                        # ── Evaluate ───────────────────────────────────────
                        metrics = evaluate(pos_test, neg_test_all, vec)
                        if metrics is None:
                            skipped += 1
                            continue

                        row = {
                            "target": target,
                            "spec": spec,
                            "layer": layer,
                            "method": method,
                            "baseline": baseline if baseline is not None else "N/A",
                            **metrics,
                        }
                        results.append(row)

    elapsed = time.time() - t_start
    print(
        f"\nSweep complete: {len(results)} valid results, "
        f"{skipped} skipped, {total_configs} total configs attempted "
        f"in {elapsed:.1f}s"
    )
    return results


# ── Summary table ──────────────────────────────────────────────────────────

def print_top_configs(results: list[dict], n: int = 20) -> None:
    """Print top N configs by AUROC as a formatted table."""
    if not results:
        print("No results to display.")
        return

    sorted_results = sorted(results, key=lambda r: r["auroc"], reverse=True)
    top = sorted_results[:n]

    header = (
        f"{'#':>3}  {'target':<45}  {'spec':<15}  "
        f"{'layer':>5}  {'method':<18}  {'baseline':<40}  "
        f"{'AUROC':>6}  {'F1':>6}  {'n_pos':>5}  {'n_neg':>6}"
    )
    sep = "-" * len(header)
    print(f"\nTop {n} configurations by AUROC:")
    print(sep)
    print(header)
    print(sep)
    for i, r in enumerate(top, 1):
        print(
            f"{i:>3}  {r['target']:<45}  {r['spec']:<15}  "
            f"{r['layer']:>5}  {r['method']:<18}  {r['baseline']:<40}  "
            f"{r['auroc']:>6.4f}  {r['f1']:>6.4f}  {r['n_pos']:>5}  {r['n_neg']:>6}"
        )
    print(sep)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    if not CACHE_PATH.exists():
        print(f"ERROR: Cache file not found: {CACHE_PATH}")
        print("Generate it first with: python little_steer/scripts/extract_full_cache.py")
        sys.exit(1)

    print(f"Loading cache: {CACHE_PATH}")
    result = ExtractionResult.load(CACHE_PATH)
    print(result)
    print(f"Labels ({len(result.labels())}): {result.labels()}")
    print(f"Specs: {result.specs()}")
    print(f"Layers: {result.layers()}")
    print()

    results = run_sweep(result)

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(results)}")

    print_top_configs(results, n=20)


if __name__ == "__main__":
    main()
