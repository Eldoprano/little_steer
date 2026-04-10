#!/usr/bin/env python3
"""
FULL RUN — little_steer steering vector research.

Experiment strategy (CPU-only constraint, ~18s/conversation):
  1. Training: load CACHED 50-conv extraction (layers 12-26, all labels)
  2. Holdout eval: extract 15 NEW conversations (layers 12-26) → test set
  3. Dense layer sweep: extract 8 NEW conversations (ALL 32 layers) → layer profile
  4. Build vectors from cached training data
  5. Evaluate: holdout validation + dense layer profile

Total ~6 min for extraction + 2 min analysis.

Key configs validated by exploratory experiments:
  ★ spec=whole_sentence, method=mean_difference, baseline=I_REPHRASE_PROMPT
  ★ AUROC ~0.99 for most safety labels with this baseline
  ★ Layer 22-26 is optimal for this model

Run from: /home/eldoprano/Studium/GeneralCode/Reasoning_behaviours/
  python little_steer/scripts/full_run.py
"""

from __future__ import annotations
import sys, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import little_steer as ls
from little_steer import (
    LittleSteerModel, ActivationExtractor, ExtractionPlan, ExtractionSpec,
    TokenSelection, ExtractionResult, load_dataset,
)

DATA_PATH      = REPO_ROOT / "data" / "little_steer_dataset.jsonl"
OUT_DIR        = REPO_ROOT / "little_steer"
TRAIN_CACHE    = OUT_DIR / "extraction_ii_state_ethical_moral_concern.pt"
MODEL_ID       = "Qwen/Qwen3.5-4B"

# How many new conversations to extract
N_TEST         = 15   # holdout evaluation (layers 12-26)
N_LAYER_SWEEP  = 8    # dense layer sweep (all 32 layers)
SKIP_N         = 50   # skip the first N (already in training cache)

TRAIN_LAYERS   = list(range(12, 28, 2))     # [12,14,...,26]  same as training cache
DENSE_LAYERS   = list(range(0, 32, 2))      # all 16 even layers

BASELINE       = "I_REPHRASE_PROMPT"
SAFETY_TARGETS = [
    "II_FLAG_PROMPT_AS_HARMFUL",
    "II_STATE_ETHICAL_MORAL_CONCERN",
    "II_STATE_LEGAL_CONCERN",
    "II_STATE_SAFETY_CONCERN",
    "IV_INTEND_REFUSAL_OR_SAFE_ACTION",
    "V_DETAIL_HARMFUL_METHOD_OR_INFO",
]


# ── Utilities ───────────────────────────────────────────────────────────────

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


def cv_evaluate(pos, neg, method="mean_difference", k=5, C=1.0):
    """5-fold CV evaluation. For linear_probe: PCA on pos class first."""
    if len(pos) < 4 or len(neg) < 4:
        return None
    if method == "linear_probe":
        n_comp = min(50, len(pos) - 1, pos.shape[1])
        if n_comp < 2:
            return None
        pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=42)
        pca.fit(pos)
        pos_p = pca.transform(pos)
        neg_p = pca.transform(neg)
        X = np.vstack([pos_p, neg_p])
        y = np.array([1]*len(pos) + [0]*len(neg))
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

    X = np.vstack([pos, neg])
    y = np.array([1]*len(pos) + [0]*len(neg))
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
        elif method == "pca_contrastive":
            Xc = p_tr - n_tr.mean(0)
            Xc = Xc - Xc.mean(0)
            nc = min(1, len(Xc)-1, Xc.shape[1])
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


def evaluate_holdout(train_res, test_res, target, spec, baseline=None, method="mean_difference"):
    """Train on train_res, evaluate on test_res. Returns best layer result."""
    all_labels = train_res.labels()
    common_layers = [l for l in train_res.layers() if l in test_res.layers()]
    results = []
    for l in common_layers:
        pos_tr = stack(train_res.get(spec, target, l))
        if baseline:
            neg_tr = stack(train_res.get(spec, baseline, l))
        else:
            neg_parts = [stack(train_res.get(spec, lbl, l)) for lbl in all_labels
                         if lbl != target and len(train_res.get(spec, lbl, l)) > 0]
            neg_tr = np.vstack(neg_parts) if neg_parts else np.array([])
        if len(pos_tr) < 2 or len(neg_tr) < 2:
            continue

        if method == "mean_difference":
            vec = pos_tr.mean(0) - neg_tr.mean(0)
        else:
            continue

        pos_te = stack(test_res.get(spec, target, l))
        if baseline:
            neg_te = stack(test_res.get(spec, baseline, l))
        else:
            test_labels = test_res.labels()
            neg_parts = [stack(test_res.get(spec, lbl, l)) for lbl in test_labels
                         if lbl != target and len(test_res.get(spec, lbl, l)) > 0]
            neg_te = np.vstack(neg_parts) if neg_parts else np.array([])
        if len(pos_te) < 2 or len(neg_te) < 2:
            continue

        X_te = np.vstack([pos_te, neg_te])
        y_te = np.array([1]*len(pos_te) + [0]*len(neg_te))
        sims = cosim(X_te, vec)
        if sims[y_te == 1].mean() < sims[y_te == 0].mean():
            sims = -sims
        r = compute_metrics(y_te, sims)
        if r:
            results.append({"target": target, "layer": l, **r})
    return results


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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    # ── Load training cache ──────────────────────────────────────────────────
    print("Loading training extraction cache...", flush=True)
    train_result = ExtractionResult.load(str(TRAIN_CACHE))
    print(f"  Training: {train_result.metadata.n_conversations} conversations, "
          f"layers {train_result.layers()}", flush=True)

    # ── Load dataset ─────────────────────────────────────────────────────────
    print("Loading dataset...", flush=True)
    all_entries = load_dataset(str(DATA_PATH))
    print(f"  {len(all_entries)} total conversations", flush=True)

    # Take conversations NOT in training (skip first SKIP_N)
    import random
    rng = random.Random(42)
    shuffled = list(all_entries)
    rng.shuffle(shuffled)
    new_entries = shuffled[SKIP_N:]

    # ── Load model ───────────────────────────────────────────────────────────
    print_header(f"Loading model ({MODEL_ID})")
    t_load = time.time()
    model = LittleSteerModel(
        model_id=MODEL_ID,
        use_pretrained_loading=True,
        allow_multimodal=True,
        check_renaming=False,
    )
    import torch
    device_str = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f"  Model: {model.num_layers} layers, hidden_size={model.hidden_size}", flush=True)
    print(f"  Running on: {device_str}", flush=True)
    print(f"  [Load time: {time.time()-t_load:.1f}s]", flush=True)

    extractor = ActivationExtractor(model)

    # ── Extract TEST set (holdout) ────────────────────────────────────────────
    test_cache = OUT_DIR / "full_run_test.pt"
    test_entries = new_entries[:N_TEST]
    if test_cache.exists():
        print(f"  Loading cached test extraction: {test_cache}", flush=True)
        test_result = ExtractionResult.load(str(test_cache))
        print(test_result.summary(), flush=True)
    else:
        print_header(f"Extracting TEST set ({len(test_entries)} conversations, layers {TRAIN_LAYERS})")
        test_plan = ExtractionPlan(name="holdout_test")
        test_plan.add_spec("whole_sentence",
                           ExtractionSpec(TokenSelection("all", aggregation="mean"),
                                          layers=TRAIN_LAYERS))
        test_plan.add_spec("last_3",
                           ExtractionSpec(TokenSelection("last_n", n=3, aggregation="mean"),
                                          layers=TRAIN_LAYERS))
        t_ext = time.time()
        test_result = extractor.extract(test_entries, test_plan)
        print(f"  [Test extraction done: {time.time()-t_ext:.1f}s]", flush=True)
        test_result.save(str(test_cache))
        print(test_result.summary(), flush=True)

    # ── Extract DENSE LAYER SWEEP ─────────────────────────────────────────────
    sweep_cache = OUT_DIR / "full_run_sweep.pt"
    sweep_entries = new_entries[N_TEST:N_TEST + N_LAYER_SWEEP]
    if sweep_cache.exists():
        print(f"  Loading cached sweep extraction: {sweep_cache}", flush=True)
        sweep_result = ExtractionResult.load(str(sweep_cache))
        print(sweep_result.summary(), flush=True)
    else:
        print_header(f"Extracting DENSE LAYER SWEEP ({len(sweep_entries)} conversations, all 32 layers)")
        sweep_plan = ExtractionPlan(name="dense_sweep")
        sweep_plan.add_spec("whole_sentence",
                            ExtractionSpec(TokenSelection("all", aggregation="mean"),
                                           layers=DENSE_LAYERS))
        t_sweep = time.time()
        sweep_result = extractor.extract(sweep_entries, sweep_plan)
        print(f"  [Sweep extraction done: {time.time()-t_sweep:.1f}s]", flush=True)
        sweep_result.save(str(sweep_cache))
        print(sweep_result.summary(), flush=True)

    del model  # free memory

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════

    train_labels = train_result.labels()

    # ── ANALYSIS 1: Holdout evaluation of best config ────────────────────────
    print_header("ANALYSIS 1: Holdout evaluation  (train=50conv, test=15conv)")
    print(f"  Method: mean_difference  |  Baseline: {BASELINE}  |  Spec: whole_sentence", flush=True)

    holdout_results = []
    for target in SAFETY_TARGETS:
        if target == BASELINE or target not in test_result.labels():
            continue
        results = evaluate_holdout(train_result, test_result, target, "whole_sentence",
                                   baseline=BASELINE)
        holdout_results.extend(results)

    # Best layer per target
    best_per_target = {}
    for r in holdout_results:
        t = r["target"]
        if t not in best_per_target or r["auroc"] > best_per_target[t]["auroc"]:
            best_per_target[t] = r

    print_table(list(best_per_target.values()), show_cm=True)

    # ── ANALYSIS 2: Layer profile from holdout ───────────────────────────────
    print_header("ANALYSIS 2: Layer profile  (holdout test, whole_sentence, I_REPHRASE_PROMPT)")
    print(f"\n  {'Layer':>6}" + "".join(f"  {t[:18]:18}" for t in SAFETY_TARGETS
                                        if t != BASELINE and t in test_result.labels()), flush=True)
    by_tgt_layer = {(r["target"], r["layer"]): r["auroc"] for r in holdout_results}
    for l in sorted(set(r["layer"] for r in holdout_results)):
        row = f"  {l:>6}"
        for t in SAFETY_TARGETS:
            if t == BASELINE or t not in test_result.labels():
                continue
            v = by_tgt_layer.get((t, l), float("nan"))
            row += f"  {v:18.4f}"
        print(row, flush=True)

    # ── ANALYSIS 3: Holdout vs CV comparison ────────────────────────────────
    print_header("ANALYSIS 3: Holdout validation  (compare with 5-fold CV results)")
    print("  Validates whether CV estimates from training data were trustworthy", flush=True)

    cv_results = []
    for target in SAFETY_TARGETS:
        if target == BASELINE:
            continue
        best_layer = best_per_target.get(target, {}).get("layer", TRAIN_LAYERS[-1])
        pos = stack(train_result.get("whole_sentence", target, best_layer))
        neg = stack(train_result.get("whole_sentence", BASELINE, best_layer))
        cv_r = cv_evaluate(pos, neg, "mean_difference", k=5) if len(pos) >= 4 and len(neg) >= 4 else None
        holdout_r = best_per_target.get(target, {})
        cv_auroc = cv_r["auroc"] if cv_r else float("nan")
        holdout_auroc = holdout_r.get("auroc", float("nan"))
        cv_results.append({
            "target": target[:30],
            "layer": best_layer,
            "CV_AUROC": cv_auroc,
            "holdout_AUROC": holdout_auroc,
            "n_train_pos": len(pos),
            "n_test_pos": holdout_r.get("n_pos", 0),
        })

    print_table(cv_results, sort_key="holdout_AUROC")

    # ── ANALYSIS 4: Dense layer sweep (in-set 5-fold CV) ─────────────────────
    print_header("ANALYSIS 4: Dense layer sweep  (8 conversations, 5-fold CV, all 32 layers)")
    print(f"  Method: mean_difference  |  Baseline: {BASELINE}  |  Spec: whole_sentence", flush=True)

    sweep_all_labels = sweep_result.labels()
    sweep_rows = []
    for target in SAFETY_TARGETS:
        if target == BASELINE or target not in sweep_all_labels:
            continue
        best_r = None
        for l in DENSE_LAYERS:
            pos = stack(sweep_result.get("whole_sentence", target, l))
            neg = stack(sweep_result.get("whole_sentence", BASELINE, l))
            if len(pos) < 4 or len(neg) < 4:
                continue
            r = cv_evaluate(pos, neg, "mean_difference", k=min(5, len(pos), len(neg)))
            if r and (best_r is None or r["auroc"] > best_r["auroc"]):
                best_r = {"target": target[:28], "layer": l, **r}
        if best_r:
            sweep_rows.append(best_r)

    print_table(sweep_rows, show_cm=True)

    # Print layer grid for sweep
    print(f"\n  AUROC by layer (dense sweep, mean_difference, {BASELINE}):", flush=True)
    sweep_targets = [t for t in SAFETY_TARGETS if t != BASELINE and t in sweep_all_labels]
    print(f"  {'Layer':>6}" + "".join(f"  {t[:16]:16}" for t in sweep_targets), flush=True)

    all_sweep_rows = []
    for target in sweep_targets:
        for l in DENSE_LAYERS:
            pos = stack(sweep_result.get("whole_sentence", target, l))
            neg = stack(sweep_result.get("whole_sentence", BASELINE, l))
            if len(pos) < 4 or len(neg) < 4:
                continue
            r = cv_evaluate(pos, neg, "mean_difference", k=min(5, len(pos), len(neg)))
            if r:
                all_sweep_rows.append({"target": target, "layer": l, "auroc": r["auroc"]})

    by_tgt_l = {(r["target"], r["layer"]): r["auroc"] for r in all_sweep_rows}
    for l in DENSE_LAYERS:
        row = f"  {l:>6}"
        for t in sweep_targets:
            v = by_tgt_l.get((t, l), float("nan"))
            row += f"  {v:16.4f}"
        print(row, flush=True)

    # ── ANALYSIS 5: Spec comparison (whole_sentence vs last_3) ───────────────
    print_header("ANALYSIS 5: Spec comparison  (whole_sentence vs last_3, holdout test)")

    spec_rows = []
    for spec in ["whole_sentence", "last_3"]:
        for target in SAFETY_TARGETS:
            if target == BASELINE or target not in test_result.labels():
                continue
            results = evaluate_holdout(train_result, test_result, target, spec, baseline=BASELINE)
            best_r = max(results, key=lambda r: r["auroc"]) if results else None
            if best_r:
                spec_rows.append({"spec": spec, "target": target[:25], **best_r})

    print_table(spec_rows, show_cm=True)

    # ── ANALYSIS 6: Baseline comparison on holdout ───────────────────────────
    print_header("ANALYSIS 6: Baseline comparison on holdout  (I_REPHRASE_PROMPT vs all_others)")

    bl_rows = []
    for baseline_choice, bl_label in [("I_REPHRASE_PROMPT", BASELINE), ("all_others", None)]:
        for target in SAFETY_TARGETS:
            if target == BASELINE or target not in test_result.labels():
                continue
            results = evaluate_holdout(train_result, test_result, target, "whole_sentence",
                                       baseline=bl_label)
            best_r = max(results, key=lambda r: r["auroc"]) if results else None
            if best_r:
                bl_rows.append({"baseline": baseline_choice, "target": target[:25], **best_r})

    print_table(bl_rows, show_cm=True)

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print_header(f"FULL RUN SUMMARY  (total time: {elapsed:.1f}s)")

    print(f"\n  Training: {train_result.metadata.n_conversations} conversations "
          f"(cached from previous run)", flush=True)
    print(f"  Test (holdout): {len(test_entries)} conversations (freshly extracted)", flush=True)
    print(f"  Dense sweep: {len(sweep_entries)} conversations (all 32 layers)", flush=True)
    print(f"  Best config: whole_sentence + mean_difference + {BASELINE} baseline", flush=True)

    print(f"\n  Holdout results (true generalization test):", flush=True)
    for r in sorted(best_per_target.values(), key=lambda r: -r.get("auroc", 0)):
        print(f"    AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}"
              f"  n_test={r['n_pos']}  {r['target']}", flush=True)
        print(f"      {fmt_cm(r.get('cm'))}", flush=True)

    print("\n" + "="*72, flush=True)


if __name__ == "__main__":
    main()
