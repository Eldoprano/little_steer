#!/usr/bin/env python3
"""
Comprehensive sweep — memory-efficient version.

Processes one (spec, layer) at a time to stay within 16GB RAM.
31K samples × 2560 dims × float32 = ~320MB per (spec, layer) slice.

Run from: /home/eldoprano/Studium/GeneralCode/Reasoning_behaviours/
  python little_steer/scripts/sweep.py
"""
from __future__ import annotations
import gc, json, sys, time, warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, confusion_matrix

warnings.filterwarnings("ignore")
REPO_ROOT  = Path(__file__).parent.parent.parent
CACHE_PATH = REPO_ROOT / "little_steer" / "full_cache.pt"
OUT_PATH   = REPO_ROOT / "little_steer" / "sweep_results.json"
sys.path.insert(0, str(REPO_ROOT))
from little_steer import ExtractionResult

METHODS   = ["mean_difference", "mean_centering", "pca_direction"]
BASELINES = ["I_REPHRASE_PROMPT", "III_PLAN_IMMEDIATE_REASONING_STEP", "all_others"]
TRAIN_FRAC = 0.70
MIN_POS    = 5

# ── Utilities ─────────────────────────────────────────────────────────────────

def stack(tensors) -> np.ndarray:
    import torch
    # Fast path: all tensors same shape → single torch.stack then numpy
    if tensors and tensors[0].dim() == 1:
        return torch.stack([t.float() for t in tensors]).numpy().astype(np.float32)
    # Slow path: 2D tensors (mean over token dim first)
    arrs = []
    for t in tensors:
        t = t.float()
        arrs.append(t.mean(0).numpy() if t.dim() == 2 else t.numpy())
    return np.array(arrs, dtype=np.float32)

def cosim(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    vn = v / (np.linalg.norm(v) + 1e-8)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return Xn @ vn

def split_arr(arr: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(arr))
    rng.shuffle(idx)
    n_train = max(1, int(len(arr) * TRAIN_FRAC))
    return arr[idx[:n_train]], arr[idx[n_train:]]

def best_threshold_metrics(y, scores):
    n_pos = int(y.sum())
    if len(set(y.tolist())) < 2 or n_pos < MIN_POS:
        return None
    if scores[y == 1].mean() < scores[y == 0].mean():
        scores = -scores
    try:
        auroc = float(roc_auc_score(y, scores))
    except Exception:
        return None

    # Vectorized F1 search — avoid 300× sklearn calls
    thresholds = np.percentile(scores, np.linspace(0, 100, 200))
    thresholds = np.unique(thresholds)
    # preds[i] = (scores >= thr[i]) for all thresholds at once
    # shape: (n_thr, n_samples)
    preds_all = (scores[None, :] >= thresholds[:, None])  # bool (n_thr, N)
    tp_all = (preds_all & (y == 1)[None, :]).sum(1).astype(float)
    fp_all = (preds_all & (y == 0)[None, :]).sum(1).astype(float)
    fn_all = ((~preds_all) & (y == 1)[None, :]).sum(1).astype(float)
    denom = 2 * tp_all + fp_all + fn_all
    f1_all = np.where(denom > 0, 2 * tp_all / denom, 0.0)
    best_i = int(f1_all.argmax())
    best_thr = float(thresholds[best_i])
    best_f1 = float(f1_all[best_i])

    preds = (scores >= best_thr).astype(int)
    tp = int(tp_all[best_i]); fp = int(fp_all[best_i]); fn = int(fn_all[best_i])
    tn = len(y) - tp - fp - fn
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return dict(auroc=round(auroc, 4), f1=round(best_f1, 4),
                precision=round(p, 4), recall=round(r, 4),
                tn=tn, fp=fp, fn=fn, tp=tp,
                n_pos=n_pos, n_neg=int((y == 0).sum()))

# ── Slice processing on pre-converted numpy cache ─────────────────────────────

def process_slice_np(np_cache: dict, all_labels: list, spec: str, layer: int, seed: int = 42) -> list[dict]:
    """Fast version: all tensors already numpy arrays."""
    results = []
    train_parts, test_parts, label_ids, label_list = [], [], [], []

    for lbl in all_labels:
        arr = np_cache.get((spec, lbl, layer))
        if arr is None or len(arr) < MIN_POS:
            continue
        tr, te = split_arr(arr, seed + hash(lbl) % (2**20))
        if len(tr) >= MIN_POS:
            train_parts.append(tr)
            test_parts.append(te)
            label_ids.append(np.full(len(te), len(label_list), dtype=np.int32))
            label_list.append(lbl)

    if not label_list:
        return []

    X_te_all = np.vstack(test_parts)
    y_te_ids = np.concatenate(label_ids)
    train    = {lbl: train_parts[i] for i, lbl in enumerate(label_list)}

    def eval_honest(target_lbl, vec):
        target_idx = label_list.index(target_lbl)
        pos_mask = y_te_ids == target_idx
        if pos_mask.sum() < MIN_POS:
            return None
        return best_threshold_metrics(pos_mask.astype(np.int8), cosim(X_te_all, vec))

    for target in label_list:
        pos_tr = train[target]

        for baseline in BASELINES:
            if baseline == "all_others":
                neg_tr = np.vstack([train[l] for l in train if l != target])
            else:
                neg_tr = train.get(baseline)
                if neg_tr is None or len(neg_tr) < MIN_POS:
                    continue
            vec = pos_tr.mean(0) - neg_tr.mean(0)
            m = eval_honest(target, vec)
            if m:
                results.append(dict(target=target, spec=spec, layer=layer,
                                    method="mean_difference", baseline=baseline, **m))

        other_means = [train[l].mean(0) for l in train if l != target]
        if other_means:
            vec = pos_tr.mean(0) - np.stack(other_means).mean(0)
            m = eval_honest(target, vec)
            if m:
                results.append(dict(target=target, spec=spec, layer=layer,
                                    method="mean_centering", baseline="N/A", **m))

        if len(pos_tr) >= 3:
            Xc = pos_tr - pos_tr.mean(0)
            try:
                pca = PCA(n_components=1, svd_solver="randomized", random_state=42)
                pca.fit(Xc)
                vec = pca.components_[0]
                m = eval_honest(target, vec)
                if m:
                    results.append(dict(target=target, spec=spec, layer=layer,
                                        method="pca_direction", baseline="N/A", **m))
            except Exception:
                pass

    return results


# ── One slice: all labels for one (spec, layer) ───────────────────────────────

def process_slice(res: ExtractionResult, spec: str, layer: int, seed: int = 42) -> list[dict]:
    results = []
    all_labels = sorted(res.labels())

    # Stack all labels, track label boundaries for efficient indexing
    # One pass through all tensors → one big matrix per split
    train_parts, test_parts, label_ids = [], [], []
    label_list = []
    for lbl in all_labels:
        tensors = res.get(spec, lbl, layer)
        if not tensors:
            continue
        arr = stack(tensors)
        tr, te = split_arr(arr, seed + hash(lbl) % (2**20))
        if len(tr) >= MIN_POS:
            train_parts.append(tr)
            test_parts.append(te)
            label_ids.append(np.full(len(te), len(label_list), dtype=np.int32))
            label_list.append(lbl)

    if not label_list:
        return []

    # Single big matrices — O(N×D) memory once, not O(N_targets × N × D)
    X_tr_all = np.vstack(train_parts)   # shape (N_train, D)
    X_te_all = np.vstack(test_parts)    # shape (N_test,  D)
    y_te_ids = np.concatenate(label_ids)  # which label each test row is

    # Pre-compute cosim of full test set with ANY vector (reuse below)
    def eval_honest(target_lbl, vec):
        target_idx = label_list.index(target_lbl)
        pos_mask = y_te_ids == target_idx
        if pos_mask.sum() < MIN_POS:
            return None
        y = pos_mask.astype(np.int8)
        scores = cosim(X_te_all, vec)  # N_test cosim ops — single matmul
        return best_threshold_metrics(y, scores)

    # Build per-label train arrays dict for easy access
    train = {lbl: train_parts[i] for i, lbl in enumerate(label_list)}

    for target in label_list:
        pos_tr = train[target]
        if len(pos_tr) < MIN_POS:
            continue

        for baseline in BASELINES:
            if baseline == "all_others":
                neg_parts_tr = [train[l] for l in train if l != target]
                neg_tr = np.vstack(neg_parts_tr) if neg_parts_tr else None
            else:
                neg_tr = train.get(baseline)
            if neg_tr is None or len(neg_tr) < MIN_POS:
                continue
            vec = pos_tr.mean(0) - neg_tr.mean(0)
            m = eval_honest(target, vec)
            if m:
                results.append(dict(target=target, spec=spec, layer=layer,
                                    method="mean_difference", baseline=baseline, **m))

        other_means = [train[l].mean(0) for l in train if l != target]
        if other_means:
            vec = pos_tr.mean(0) - np.stack(other_means).mean(0)
            m = eval_honest(target, vec)
            if m:
                results.append(dict(target=target, spec=spec, layer=layer,
                                    method="mean_centering", baseline="N/A", **m))

        if len(pos_tr) >= 3:
            Xc = pos_tr - pos_tr.mean(0)
            try:
                pca = PCA(n_components=1, svd_solver="randomized", random_state=42)
                pca.fit(Xc)
                vec = pca.components_[0]
                m = eval_honest(target, vec)
                if m:
                    results.append(dict(target=target, spec=spec, layer=layer,
                                        method="pca_direction", baseline="N/A", **m))
            except Exception:
                pass

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    print(f"Loading {CACHE_PATH}...")
    res = ExtractionResult.load(str(CACHE_PATH))
    specs  = [s for s in ["whole_sentence","last_1","first_1","last_3","first_3","bleed_3"]
              if s in res.specs()]
    layers = sorted(res.layers())
    all_labels = sorted(res.labels())
    print(f"  {res.metadata.n_conversations} convs, {len(all_labels)} labels")
    print(f"  Specs: {specs}  Layers: {layers}", flush=True)

    # Pre-convert ALL tensors to numpy arrays once (eliminates bfloat16→float32 per slice)
    print("  Pre-converting tensors to numpy... ", end="", flush=True)
    t_pre = time.time()
    np_cache: dict[tuple, np.ndarray] = {}  # (spec, label, layer) -> np.ndarray
    for spec_s in specs:
        for lbl in all_labels:
            for lyr in layers:
                tensors = res.get(spec_s, lbl, lyr)
                if tensors:
                    # Batch convert: single torch.stack then .numpy()
                    if tensors[0].dim() == 1:
                        arr = torch.stack([t.float() for t in tensors]).numpy().astype(np.float32)
                    else:
                        arr = torch.stack([t.float().mean(0) for t in tensors]).numpy().astype(np.float32)
                    np_cache[(spec_s, lbl, lyr)] = arr
    print(f"done in {time.time()-t_pre:.1f}s  ({len(np_cache)} arrays)", flush=True)
    del res; gc.collect()  # free original torch tensors

    all_results = []
    t0 = time.time()

    for spec in specs:
        for layer in layers:
            t1 = time.time()
            slice_results = process_slice_np(np_cache, all_labels, spec, layer)
            all_results.extend(slice_results)
            if slice_results:
                best = max(slice_results, key=lambda x: x["auroc"])
                t_short = best['target'].replace('II_','').replace('IV_','').replace('III_','').replace('I_','')
                print(f"  {spec:<18} L{layer}  {len(slice_results):>4} results  "
                      f"best AUROC={best['auroc']:.4f} ({t_short[:22]}, "
                      f"{best['method'][:14]}, {best['baseline'][:20]})  "
                      f"[{time.time()-t1:.1f}s]", flush=True)
            gc.collect()

    elapsed = time.time() - t0
    print(f"\nTotal: {len(all_results)} results in {elapsed:.1f}s")

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved → {OUT_PATH}")

    print(f"\n{'='*95}")
    print(f"  {'Target':<38} {'AUROC':>7} {'F1':>6}  {'Spec':<16} {'L':>2}  {'Method':<16} {'Baseline'}")
    print(f"  {'='*93}")
    for r in sorted(all_results, key=lambda x: -x["auroc"])[:20]:
        t = r["target"].replace("II_","").replace("IV_","").replace("III_","").replace("I_","")
        print(f"  {t:<38} {r['auroc']:>7.4f} {r['f1']:>6.4f}  {r['spec']:<16} "
              f"{r['layer']:>2}  {r['method']:<16} {r['baseline']}")

if __name__ == "__main__":
    main()
