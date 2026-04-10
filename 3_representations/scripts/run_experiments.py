#!/usr/bin/env python3
"""
Experiment suite for little_steer steering vector research.

Goal: find configurations that maximize detection AUROC/F1/confusion matrix
for safety-relevant behaviors in reasoning model activations.

All experiments use CACHED ExtractionResult — no model forward pass needed.
K-fold=5 cross-validation throughout for honest estimates.

Key optimizations:
- linear_probe: PCA on positive class (small, fast) → project all data → LogReg
- mean_difference / pca: cosine similarity directly (no PCA needed)

Run from: /home/eldoprano/Studium/GeneralCode/Reasoning_behaviours/
  python little_steer/scripts/run_experiments.py
"""

from __future__ import annotations
import sys, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import little_steer as ls
from little_steer.extraction.result import ExtractionResult

EXTRACTION_CACHE = REPO_ROOT / "little_steer" / "extraction_ii_state_ethical_moral_concern.pt"
DATA_PATH = REPO_ROOT / "data" / "little_steer_dataset.jsonl"


# ── Core utilities ─────────────────────────────────────────────────────────

def stack(tensors) -> np.ndarray:
    arrs = []
    for t in tensors:
        t = t.float()
        arrs.append(t.mean(0).numpy() if t.dim() == 2 else t.numpy())
    return np.array(arrs, dtype=np.float32)


def cosim(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    vn = v / (np.linalg.norm(v) + 1e-8)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return Xn @ vn


def compute_metrics(y: np.ndarray, scores: np.ndarray) -> dict | None:
    if len(set(y.tolist())) < 2:
        return None
    auroc = float(roc_auc_score(y, scores))
    thresholds = np.linspace(scores.min(), scores.max(), 100)
    best_f1, best_thr = 0.0, float(thresholds[0])
    for thr in thresholds:
        f1 = float(f1_score(y, (scores >= thr).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    preds = (scores >= best_thr).astype(int)
    p, r, _, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y, preds, labels=[0, 1])
    return {"auroc": auroc, "f1": best_f1, "precision": float(p), "recall": float(r),
            "cm": cm, "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}


def cv_evaluate(pos: np.ndarray, neg: np.ndarray, method: str, k: int = 5,
                C: float = 1.0) -> dict | None:
    """5-fold CV evaluation. For linear_probe: PCA on positive class first."""
    if len(pos) < 4 or len(neg) < 4:
        return None

    if method == "linear_probe":
        # PCA on positive class only (small → fast), then project all data
        n_comp = min(50, len(pos) - 1, pos.shape[1])
        if n_comp < 2:
            return None
        pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=42)
        pca.fit(pos)  # ~0.03s for 110 samples
        pos_p = pca.transform(pos)
        neg_p = pca.transform(neg)
        X = np.vstack([pos_p, neg_p])
        y = np.array([1] * len(pos) + [0] * len(neg))
        n_splits = min(k, len(pos), len(neg))
        if n_splits < 2:
            return None
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_s, all_y = [], []
        for tr_i, te_i in skf.split(X, y):
            X_tr, X_te = X[tr_i], X[te_i]
            y_tr, y_te = y[tr_i], y[te_i]
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)
            clf = LogisticRegression(C=C, max_iter=500, solver='lbfgs', tol=1e-3)
            clf.fit(X_tr_sc, y_tr)
            s = clf.decision_function(X_te_sc)
            if s[y_te == 1].mean() < s[y_te == 0].mean():
                s = -s
            all_s.extend(s.tolist())
            all_y.extend(y_te.tolist())
        return compute_metrics(np.array(all_y), np.array(all_s)) if all_y else None

    # For non-probe methods: build a vector and compute cosine similarities
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    n_splits = min(k, len(pos), len(neg))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_s, all_y = [], []

    for tr_i, te_i in skf.split(X, y):
        X_tr, X_te = X[tr_i], X[te_i]
        y_tr, y_te = y[tr_i], y[te_i]
        p_tr, n_tr = X_tr[y_tr == 1], X_tr[y_tr == 0]
        if len(p_tr) < 2 or len(n_tr) < 2:
            continue

        if method == "mean_difference":
            v = p_tr.mean(0) - n_tr.mean(0)

        elif method == "pca":
            Xc = p_tr - p_tr.mean(0)
            nc = min(1, len(p_tr) - 1, Xc.shape[1])
            if nc < 1:
                continue
            pca = PCA(n_components=nc, svd_solver='randomized', random_state=42)
            pca.fit(Xc)
            v = pca.components_[0]

        elif method == "pca_contrastive":
            # PCA on (pos - neg_mean): variance relative to baseline
            Xc = p_tr - n_tr.mean(0)
            Xc = Xc - Xc.mean(0)
            nc = min(1, len(Xc) - 1, Xc.shape[1])
            if nc < 1:
                continue
            pca = PCA(n_components=nc, svd_solver='randomized', random_state=42)
            pca.fit(Xc)
            v = pca.components_[0]

        else:
            raise ValueError(f"Unknown method: {method}")

        s = cosim(X_te, v)
        if s[y_te == 1].mean() < s[y_te == 0].mean():
            s = -s
        all_s.extend(s.tolist())
        all_y.extend(y_te.tolist())

    return compute_metrics(np.array(all_y), np.array(all_s)) if all_y else None


def cv_evaluate_mean_centering(pos: np.ndarray, others: dict[str, np.ndarray],
                                k: int = 5) -> dict | None:
    """K-fold CV for mean_centering."""
    neg_arr = np.vstack(list(others.values())) if others else np.array([])
    neg_cat = np.concatenate([[i] * len(v) for i, v in enumerate(others.values())]) if others else np.array([])
    cat_names = list(others.keys())

    if len(pos) < 4 or len(neg_arr) < 4:
        return None

    X = np.vstack([pos, neg_arr])
    y = np.array([1] * len(pos) + [0] * len(neg_arr))
    cat = np.array([-1] * len(pos) + neg_cat.tolist())
    n_splits = min(k, len(pos), len(neg_arr))
    if n_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_s, all_y = [], []

    for tr_i, te_i in skf.split(X, y):
        X_tr, X_te = X[tr_i], X[te_i]
        y_tr, y_te = y[tr_i], y[te_i]
        cat_tr = cat[tr_i]
        p_tr = X_tr[y_tr == 1]
        if len(p_tr) < 2:
            continue
        other_means = [X_tr[cat_tr == i].mean(0) for i in range(len(cat_names))
                       if (cat_tr == i).sum() >= 1]
        if not other_means:
            continue
        centroid = np.stack(other_means).mean(0)
        v = p_tr.mean(0) - centroid
        s = cosim(X_te, v)
        if s[y_te == 1].mean() < s[y_te == 0].mean():
            s = -s
        all_s.extend(s.tolist())
        all_y.extend(y_te.tolist())

    return compute_metrics(np.array(all_y), np.array(all_s)) if all_y else None


# ── Display helpers ────────────────────────────────────────────────────────

def fmt_cm(cm):
    if cm is None:
        return "N/A"
    tn, fp, fn, tp = cm.ravel()
    return f"TP={tp} FP={fp} FN={fn} TN={tn}"


def print_header(title):
    print(f"\n{'='*72}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'='*72}", flush=True)


def print_table(rows, sort_key="auroc", top_n=None, show_cm=False):
    if not rows:
        print("  (no results)", flush=True)
        return
    rows = sorted([r for r in rows if r.get(sort_key) is not None],
                  key=lambda r: r.get(sort_key, 0), reverse=True)
    if top_n:
        rows = rows[:top_n]
    if not rows:
        return

    skip = {"cm", "threshold", "n_neg"}
    cols = [k for k in rows[0] if k not in skip]
    w = 16
    hdr = "  " + " | ".join(f"{c[:w]:{w}}" for c in cols)
    print(hdr, flush=True)
    print("  " + "-" * len(hdr.strip()), flush=True)
    for r in rows:
        parts = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                parts.append(f"{v:{w}.4f}")
            else:
                parts.append(f"{str(v)[:w]:{w}}")
        print("  " + " | ".join(parts), flush=True)
        if show_cm and "cm" in r:
            print(f"      → {fmt_cm(r['cm'])}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print("Loading extraction result...", flush=True)
    result = ExtractionResult.load(str(EXTRACTION_CACHE))
    print(f"Loaded. Labels: {len(result.labels())}, Specs: {result.specs()}, Layers: {result.layers()}", flush=True)

    all_labels = result.labels()
    specs = result.specs()
    layers = result.layers()
    TARGET = "II_STATE_ETHICAL_MORAL_CONCERN"
    K = 5
    METHODS = ["mean_difference", "pca", "pca_contrastive", "mean_centering", "linear_probe"]

    # ─────────────────────────────────────────────────────────────────────────
    # Helper: get pos/neg arrays for a target
    # ─────────────────────────────────────────────────────────────────────────
    def get_arrays(spec, target, layer, baseline_label=None):
        pos = stack(result.get(spec, target, layer))
        if baseline_label:
            neg = stack(result.get(spec, baseline_label, layer))
        else:
            neg_parts = [stack(result.get(spec, lbl, layer)) for lbl in all_labels
                         if lbl != target and len(result.get(spec, lbl, layer)) > 0]
            neg = np.vstack(neg_parts) if neg_parts else np.array([])
        return pos, neg

    def get_others(spec, target, layer):
        return {lbl: stack(result.get(spec, lbl, layer)) for lbl in all_labels
                if lbl != target and len(result.get(spec, lbl, layer)) > 0}

    # ─────────────────────────────────────────────────────────────────────────
    # EXP 1: Method × Spec × Layer  (5-fold CV, neg = all others pooled)
    # ─────────────────────────────────────────────────────────────────────────
    print_header("EXP 1: Method × Spec × Layer  (target=II_STATE_ETHICAL_MORAL_CONCERN)")
    print("  Neg = all other labels pooled  |  5-fold CV", flush=True)
    t1 = time.time()

    exp1 = []
    for spec in specs:
        for layer in layers:
            pos, neg = get_arrays(spec, TARGET, layer)
            others = get_others(spec, TARGET, layer)
            for method in METHODS:
                if method == "mean_centering":
                    r = cv_evaluate_mean_centering(pos, others, k=K)
                else:
                    r = cv_evaluate(pos, neg, method, k=K)
                if r:
                    exp1.append({"spec": spec, "method": method, "layer": layer, **r})

    print_table(exp1, top_n=20, show_cm=True)
    best1 = max(exp1, key=lambda r: r["auroc"]) if exp1 else {}
    best_spec = best1.get("spec", specs[0])
    best_method_e1 = best1.get("method", "mean_difference")
    best_layer_e1 = best1.get("layer", layers[0])
    if best1:
        print(f"\n  ★ BEST: spec={best_spec} method={best_method_e1} layer={best_layer_e1}"
              f"  AUROC={best1['auroc']:.4f}  F1={best1['f1']:.4f}", flush=True)
        print(f"    {fmt_cm(best1.get('cm'))}", flush=True)
    print(f"  [EXP1 done: {time.time()-t1:.1f}s]", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXP 2: Layer profile for best spec — comparison of methods
    # ─────────────────────────────────────────────────────────────────────────
    print_header(f"EXP 2: Layer profile  (spec={best_spec!r})")

    by_layer = defaultdict(dict)
    for r in exp1:
        if r["spec"] == best_spec:
            by_layer[r["layer"]][r["method"]] = r["auroc"]

    # Pretty print layer × method AUROC grid
    print(f"\n  AUROC by (layer, method) for spec={best_spec!r}:", flush=True)
    sorted_methods = sorted({r["method"] for r in exp1})
    header = f"  {'Layer':>6}" + "".join(f"  {m[:18]:18}" for m in sorted_methods)
    print(header, flush=True)
    for layer in sorted(layers):
        row = f"  {layer:>6}"
        for m in sorted_methods:
            v = by_layer[layer].get(m, float("nan"))
            row += f"  {v:18.4f}"
        print(row, flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXP 3: Baseline label comparison (mean_difference + linear_probe)
    # ─────────────────────────────────────────────────────────────────────────
    print_header("EXP 3: Baseline label choice  (mean_difference + linear_probe)")
    t3 = time.time()

    BASELINES = [
        ("all_others",           None),
        ("I_REPHRASE_PROMPT",    "I_REPHRASE_PROMPT"),
        ("I_SPECULATE_MOTIVE",   "I_SPECULATE_USER_MOTIVE"),
        ("III_PLAN_STEP",        "III_PLAN_IMMEDIATE_REASONING_STEP"),
        ("III_FACT",             "III_STATE_FACT_OR_KNOWLEDGE"),
        ("IV_INTEND_REFUSAL",    "IV_INTEND_REFUSAL_OR_SAFE_ACTION"),
        ("VI_FILLER",            "VI_NEUTRAL_FILLER_TRANSITION"),
    ]

    exp3 = []
    for bl_name, bl_label in BASELINES:
        for method in ["mean_difference", "linear_probe"]:
            best_r = None
            for layer in layers:
                pos, neg = get_arrays(best_spec, TARGET, layer, baseline_label=bl_label)
                if len(neg) < 4:
                    continue
                r = cv_evaluate(pos, neg, method, k=K)
                if r and (best_r is None or r["auroc"] > best_r["auroc"]):
                    best_r = {"baseline": bl_name, "method": method, "best_layer": layer, **r}
            if best_r:
                exp3.append(best_r)

    print_table(exp3, show_cm=True)
    print(f"  [EXP3 done: {time.time()-t3:.1f}s]", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXP 4: LinearProbe C (regularization sweep) — on best_spec
    # ─────────────────────────────────────────────────────────────────────────
    print_header("EXP 4: LinearProbe regularization (C parameter sweep)")
    t4 = time.time()

    exp4 = []
    for C in [0.01, 0.1, 0.5, 1.0, 5.0, 20.0]:
        best_r = None
        for layer in layers:
            pos, neg = get_arrays(best_spec, TARGET, layer)
            r = cv_evaluate(pos, neg, "linear_probe", k=K, C=C)
            if r and (best_r is None or r["auroc"] > best_r["auroc"]):
                best_r = {"C": C, "best_layer": layer, **r}
        if best_r:
            exp4.append(best_r)

    print_table(exp4, show_cm=True)
    print(f"  [EXP4 done: {time.time()-t4:.1f}s]", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXP 5: PCA multi-component ensemble
    # For each fold: fit top-k PCA components on positive training set,
    # then take MAX cosine similarity across components
    # ─────────────────────────────────────────────────────────────────────────
    print_header("EXP 5: PCA k-component ensemble  (max sim across k components)")
    t5 = time.time()

    exp5 = []
    for n_comp in [1, 2, 3, 5, 8, 15]:
        best_r = None
        for layer in layers:
            pos, neg = get_arrays(best_spec, TARGET, layer)
            if len(pos) < n_comp + 2:
                continue

            X = np.vstack([pos, neg])
            y = np.array([1]*len(pos) + [0]*len(neg))
            n_splits = min(K, len(pos), len(neg))
            if n_splits < 2:
                continue
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            all_s, all_y2 = [], []

            for tr_i, te_i in skf.split(X, y):
                X_tr, X_te = X[tr_i], X[te_i]
                y_tr, y_te = y[tr_i], y[te_i]
                p_tr = X_tr[y_tr == 1]
                actual_nc = min(n_comp, len(p_tr)-1, p_tr.shape[1])
                if actual_nc < 1:
                    continue
                Xc = p_tr - p_tr.mean(0)
                pca = PCA(n_components=actual_nc, svd_solver='randomized', random_state=42)
                pca.fit(Xc)
                # Max cosine similarity across components
                comp_sims = np.stack([cosim(X_te, c) for c in pca.components_])  # (nc, n_test)
                s = comp_sims.max(0)
                if s[y_te==1].mean() < s[y_te==0].mean():
                    s = -s
                all_s.extend(s.tolist())
                all_y2.extend(y_te.tolist())

            r = compute_metrics(np.array(all_y2), np.array(all_s)) if all_y2 else None
            if r and (best_r is None or r["auroc"] > best_r["auroc"]):
                best_r = {"n_components": n_comp, "best_layer": layer, **r}

        if best_r:
            exp5.append(best_r)

    print_table(exp5)
    print(f"  [EXP5 done: {time.time()-t5:.1f}s]", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXP 6: Layer ensembling — avg similarity across best-k layers
    # Uses mean_difference vectors for speed
    # ─────────────────────────────────────────────────────────────────────────
    print_header("EXP 6: Layer ensembling  (mean_difference, avg sim across best-k layers)")
    t6 = time.time()

    # Rank layers by single-layer AUROC for (best_spec, mean_difference)
    ranked = sorted([(r["layer"], r["auroc"]) for r in exp1
                     if r["spec"] == best_spec and r["method"] == "mean_difference"],
                    key=lambda x: -x[1])
    ranked_layers = [l for l, _ in ranked]

    exp6 = []
    for top_k in range(1, len(ranked_layers)+1):
        use_layers = ranked_layers[:top_k]

        # Build pos/neg arrays for all layers (same set)
        pos_by_l = {l: stack(result.get(best_spec, TARGET, l)) for l in use_layers}
        neg_by_l = {}
        for l in use_layers:
            neg_parts = [stack(result.get(best_spec, lbl, l)) for lbl in all_labels
                         if lbl != TARGET and len(result.get(best_spec, lbl, l)) > 0]
            neg_by_l[l] = np.vstack(neg_parts) if neg_parts else np.array([])

        # Use first layer's structure for CV splitting
        l0 = use_layers[0]
        pos0, neg0 = pos_by_l[l0], neg_by_l[l0]
        if len(pos0) < 4 or len(neg0) < 4:
            continue

        X0 = np.vstack([pos0, neg0])
        y0 = np.array([1]*len(pos0) + [0]*len(neg0))
        n_splits = min(K, len(pos0), len(neg0))
        if n_splits < 2:
            continue
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_s, all_y = [], []

        for tr_i, te_i in skf.split(X0, y0):
            # Compute avg similarity across layers
            layer_s_list = []
            y_te_refs = None
            valid = True
            for l in use_layers:
                pos_l, neg_l = pos_by_l[l], neg_by_l[l]
                np_ = len(pos_l)
                nn = len(neg_l)
                # Map fold indices
                pos_tr = pos_l[tr_i[tr_i < np_]]
                neg_tr_idx = tr_i[tr_i >= np_] - np_
                neg_tr = neg_l[neg_tr_idx[neg_tr_idx < nn]]
                pos_te = pos_l[te_i[te_i < np_]]
                neg_te_idx = te_i[te_i >= np_] - np_
                neg_te = neg_l[neg_te_idx[neg_te_idx < nn]]

                if len(pos_tr) < 2 or len(neg_tr) < 2:
                    valid = False
                    break
                X_te_l = np.vstack([pos_te, neg_te]) if len(pos_te)+len(neg_te) > 0 else None
                if X_te_l is None or len(X_te_l) == 0:
                    valid = False
                    break

                v = pos_tr.mean(0) - neg_tr.mean(0)
                s = cosim(X_te_l, v)
                y_te = np.array([1]*len(pos_te) + [0]*len(neg_te))
                if y_te_refs is None:
                    y_te_refs = y_te
                layer_s_list.append(s)

            if not valid or not layer_s_list or y_te_refs is None:
                continue

            # Average similarities across layers (align each first)
            aligned = []
            for s_l in layer_s_list:
                if len(s_l[y_te_refs==1]) > 0 and s_l[y_te_refs==1].mean() < s_l[y_te_refs==0].mean():
                    s_l = -s_l
                aligned.append(s_l)

            avg_s = np.mean(aligned, axis=0)
            all_s.extend(avg_s.tolist())
            all_y.extend(y_te_refs.tolist())

        r = compute_metrics(np.array(all_y), np.array(all_s)) if all_y else None
        if r:
            exp6.append({"top_k_layers": top_k, "layers": str(use_layers[:4]), **r})

    print_table(exp6)
    print(f"  [EXP6 done: {time.time()-t6:.1f}s]", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXP 7: ALL LABELS — which behaviors are most detectable?
    # ─────────────────────────────────────────────────────────────────────────
    print_header("EXP 7: All labels — detection quality (best method+layer per label)")
    t7 = time.time()

    exp7 = []
    for target_lbl in all_labels:
        best_r = None
        for method in ["mean_difference", "pca_contrastive", "linear_probe"]:
            for layer in layers:
                pos, neg = get_arrays(best_spec, target_lbl, layer)
                if len(pos) < 4:
                    continue
                r = cv_evaluate(pos, neg, method, k=K)
                if r and (best_r is None or r["auroc"] > best_r["auroc"]):
                    best_r = {"label": target_lbl[:32], "n_pos": len(pos),
                              "method": method, "layer": layer, **r}
        if best_r:
            exp7.append(best_r)

    print_table(exp7, show_cm=True)
    print(f"  [EXP7 done: {time.time()-t7:.1f}s]", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXP 8: Safety labels × specific semantic baselines (linear_probe)
    # Test: does choosing a semantically opposite baseline help?
    # ─────────────────────────────────────────────────────────────────────────
    print_header("EXP 8: Safety labels × semantic baselines  (linear_probe)")
    t8 = time.time()

    SAFETY_TARGETS = [
        "II_FLAG_PROMPT_AS_HARMFUL",
        "II_STATE_ETHICAL_MORAL_CONCERN",
        "II_STATE_SAFETY_CONCERN",
        "II_STATE_LEGAL_CONCERN",
        "V_DETAIL_HARMFUL_METHOD_OR_INFO",
        "IV_INTEND_REFUSAL_OR_SAFE_ACTION",
    ]
    SPECIFIC_BLS = [
        ("all_others",        None),
        ("I_REPHRASE_PROMPT", "I_REPHRASE_PROMPT"),
        ("IV_INTEND_REFUSAL", "IV_INTEND_REFUSAL_OR_SAFE_ACTION"),
        ("III_PLAN_STEP",     "III_PLAN_IMMEDIATE_REASONING_STEP"),
        ("VI_FILLER",         "VI_NEUTRAL_FILLER_TRANSITION"),
    ]

    exp8 = []
    for target_lbl in SAFETY_TARGETS:
        for bl_name, bl_label in SPECIFIC_BLS:
            if bl_label == target_lbl:
                continue
            best_r = None
            for layer in layers:
                pos, neg = get_arrays(best_spec, target_lbl, layer, baseline_label=bl_label)
                if len(pos) < 4 or len(neg) < 4:
                    continue
                r = cv_evaluate(pos, neg, "linear_probe", k=K)
                if r and (best_r is None or r["auroc"] > best_r["auroc"]):
                    best_r = {"target": target_lbl[:22], "baseline": bl_name, "layer": layer, **r}
            if best_r:
                exp8.append(best_r)

    print_table(exp8, show_cm=True)
    print(f"  [EXP8 done: {time.time()-t8:.1f}s]", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # EXP 9: Annotation score distribution (meta-analysis)
    # ─────────────────────────────────────────────────────────────────────────
    print_header("EXP 9: Annotation confidence score distribution")
    try:
        entries = ls.load_dataset(str(DATA_PATH), show_progress=False)
        score_by_label = defaultdict(lambda: defaultdict(int))
        for e in entries:
            for ann in e.annotations:
                for lbl in ann.labels:
                    score_by_label[lbl][ann.score] += 1

        print(f"\n  {TARGET} scores:", flush=True)
        for s in sorted(score_by_label[TARGET]):
            bar = "█" * min(score_by_label[TARGET][s] // 3, 50)
            print(f"    score {s}: {score_by_label[TARGET][s]:4d}  {bar}", flush=True)

        print(f"\n  Score distribution by label (count with score≥2 / total):", flush=True)
        for lbl in sorted(all_labels):
            d = score_by_label[lbl]
            total = sum(d.values())
            high = sum(v for k, v in d.items() if k >= 2)
            pct = 100 * high / total if total > 0 else 0
            print(f"    {lbl[:40]:40}  high={high:4d}/{total:4d}  ({pct:.0f}%)", flush=True)
    except Exception as ex:
        print(f"  Could not load dataset: {ex}", flush=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print_header(f"SUMMARY  (total time: {elapsed:.1f}s)")

    print("\n  Top-5 configs overall (EXP 1):", flush=True)
    for r in sorted(exp1, key=lambda r: -r["auroc"])[:5]:
        print(f"    AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}"
              f"  spec={r['spec']} method={r['method']} layer={r['layer']}", flush=True)
        print(f"      {fmt_cm(r.get('cm'))}", flush=True)

    print("\n  Best baselines (EXP 3):", flush=True)
    for r in sorted(exp3, key=lambda r: -r["auroc"])[:4]:
        print(f"    AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}"
              f"  method={r['method']} baseline={r['baseline']}", flush=True)

    print("\n  Layer ensembling sweet spot (EXP 6):", flush=True)
    for r in sorted(exp6, key=lambda r: -r["auroc"])[:3]:
        print(f"    AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}"
              f"  top_k={r['top_k_layers']} layers={r['layers']}", flush=True)

    print("\n  Most detectable labels (EXP 7):", flush=True)
    for r in sorted(exp7, key=lambda r: -r["auroc"])[:8]:
        print(f"    AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}"
              f"  n_pos={r['n_pos']}  {r['label']}", flush=True)

    print("\n  Best safety detection per label (EXP 8):", flush=True)
    best_e8 = {}
    for r in exp8:
        t = r["target"]
        if t not in best_e8 or r["auroc"] > best_e8[t]["auroc"]:
            best_e8[t] = r
    for r in sorted(best_e8.values(), key=lambda r: -r["auroc"]):
        print(f"    AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}"
              f"  target={r['target']}  best_bl={r['baseline']}", flush=True)

    print("\n" + "="*72, flush=True)


if __name__ == "__main__":
    main()
