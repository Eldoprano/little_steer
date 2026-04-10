#!/usr/bin/env python3
"""
Visualization + honest evaluation of steering vectors.

1. Per-conversation sentence-level similarity plot
2. Distribution of similarities across ALL sentence types (realistic scenario)
3. Honest AUROC: vector applied to ALL test sentences, not just target vs one baseline
4. Token position comparison (whole_sentence vs last_3 vs last_1)

Run from: /home/eldoprano/Studium/GeneralCode/Reasoning_behaviours/
  python little_steer/scripts/visualize.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from little_steer import (
    LittleSteerModel, ActivationExtractor, ExtractionPlan, ExtractionSpec,
    TokenSelection, ExtractionResult, load_dataset,
)

OUT_DIR    = REPO_ROOT / "little_steer"
TRAIN_CACHE = OUT_DIR / "extraction_ii_state_ethical_moral_concern.pt"
TEST_CACHE  = OUT_DIR / "full_run_test.pt"
DATA_PATH   = REPO_ROOT / "data" / "little_steer_dataset.jsonl"
PLOTS_DIR   = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

LAYER        = 22
SPEC         = "whole_sentence"
BASELINE     = "I_REPHRASE_PROMPT"
SAFETY_TARGETS = [
    "II_FLAG_PROMPT_AS_HARMFUL",
    "II_STATE_ETHICAL_MORAL_CONCERN",
    "II_STATE_LEGAL_CONCERN",
    "II_STATE_SAFETY_CONCERN",
    "IV_INTEND_REFUSAL_OR_SAFE_ACTION",
]

# ── Utilities ────────────────────────────────────────────────────────────────

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


def build_vector(train_res: ExtractionResult, target: str, baseline: str, spec: str, layer: int) -> np.ndarray:
    pos = stack(train_res.get(spec, target, layer))
    neg = stack(train_res.get(spec, baseline, layer))
    return pos.mean(0) - neg.mean(0)


# ── PLOT 1: Per-conversation sentence-level similarity ────────────────────────

def plot_conversation_similarity(conv_entry, extractor, train_res, target: str,
                                  spec: str, layer: int, baseline: str, out_path: Path):
    """Extract activations for every sentence in one conversation and plot similarity."""
    print(f"  Extracting activations for conversation {conv_entry.id}...", flush=True)
    t0 = time.time()

    plan = ExtractionPlan(name="viz_single")
    plan.add_spec(spec, ExtractionSpec(
        TokenSelection("all", aggregation="mean") if spec == "whole_sentence" else
        TokenSelection("last_n", n=3, aggregation="mean"),
        layers=[layer]
    ))

    result = extractor.extract([conv_entry], plan)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    # Build steering vector from training data
    vec = build_vector(train_res, target, baseline, spec, layer)

    # Collect all sentence activations + labels from this conversation
    all_sims = []
    all_labels = []
    all_texts = []

    for ann in sorted(conv_entry.annotations, key=lambda a: (a.message_idx, a.char_start)):
        ann_labels = [l for l in ann.labels if l] if ann.labels else []
        # Look up the activation — we need to find which tensor matches this annotation.
        # Since result groups by label, look for each label this annotation has.
        # For unlabeled sentences (or labels not in result), we mark as "other"
        found_tensor = None
        matched_label = None
        for lbl in ann_labels:
            tensors = result.get(spec, lbl, layer)
            if tensors:
                # The tensor for this annotation is the last one added for this label
                # (because we're processing a single conversation)
                # To be safe, use the first available tensor
                found_tensor = tensors[0]
                matched_label = lbl
                break

        if found_tensor is None:
            # Unlabeled or label with no tensor — skip
            continue

        act = found_tensor.float().numpy()
        act_2d = np.expand_dims(act, 0)
        sim = float(cosim(act_2d, vec)[0])
        all_sims.append(sim)
        is_target = int(target in ann_labels)
        all_labels.append(is_target)
        short_text = ann.text[:60].replace("\n", " ")
        all_texts.append(short_text)

    if len(all_sims) < 2:
        print(f"  Not enough data for conversation plot", flush=True)
        return

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(all_sims))
    colors = ["#d62728" if l == 1 else "#1f77b4" for l in all_labels]
    ax.bar(xs, all_sims, color=colors, alpha=0.8, edgecolor="none")
    ax.axhline(0, color="black", lw=0.5, linestyle="--")

    ax.set_xlabel("Sentence index (ordered by position in conversation)", fontsize=11)
    ax.set_ylabel("Cosine similarity to steering vector", fontsize=11)
    ax.set_title(
        f"Per-sentence similarity: steering vector for '{target}'\n"
        f"Spec={spec}, Layer={layer}, Baseline={baseline}\n"
        f"Conversation {conv_entry.id[:12]}",
        fontsize=11
    )
    red_patch = mpatches.Patch(color="#d62728", label=f"Positive ({target})")
    blue_patch = mpatches.Patch(color="#1f77b4", label="Other labels")
    ax.legend(handles=[red_patch, blue_patch], fontsize=10)

    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    ax.text(0.02, 0.97, f"n_pos={n_pos}  n_neg={n_neg}", transform=ax.transAxes,
            fontsize=9, va="top")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}", flush=True)
    return all_sims, all_labels


# ── PLOT 2: Distribution plot (all test sentences) ────────────────────────────

def plot_similarity_distributions(test_res: ExtractionResult, train_res: ExtractionResult,
                                   target: str, spec: str, layer: int, baseline: str,
                                   out_path: Path):
    """Box/violin plot of similarity scores by label category across all test sentences."""
    vec = build_vector(train_res, target, baseline, spec, layer)

    all_labels = test_res.labels()
    label_sims = {}
    for lbl in sorted(all_labels):
        tensors = test_res.get(spec, lbl, layer)
        if len(tensors) < 3:
            continue
        X = stack(tensors)
        sims = cosim(X, vec)
        label_sims[lbl] = sims

    if not label_sims:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    labels_sorted = sorted(label_sims.keys(), key=lambda l: np.median(label_sims[l]), reverse=True)
    data = [label_sims[l] for l in labels_sorted]
    positions = np.arange(len(labels_sorted))

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    medianprops=dict(color="black", lw=2))

    for i, (lbl, patch) in enumerate(zip(labels_sorted, bp["boxes"])):
        color = "#d62728" if lbl == target else "#aec7e8"
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels_sorted], fontsize=7)
    ax.set_ylabel("Cosine similarity to steering vector", fontsize=11)
    ax.set_title(
        f"Similarity distribution by label category\n"
        f"Steering vector: '{target}' vs '{baseline}',  Spec={spec}, Layer={layer}",
        fontsize=11
    )
    ax.axhline(0, color="black", lw=0.5, linestyle="--")
    red_patch = mpatches.Patch(color="#d62728", label=f"Target: {target}")
    blue_patch = mpatches.Patch(color="#aec7e8", label="Other categories")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}", flush=True)


# ── PLOT 3: Honest AUROC — vector applied to ALL test sentences ───────────────

def honest_eval_all_sentences(test_res: ExtractionResult, train_res: ExtractionResult,
                               spec: str, layer: int, baseline: str, out_path: Path):
    """
    HONEST evaluation: build vector, then score ALL test sentences.
    Positive = sentences labeled with target.
    Negative = ALL other labeled sentences (not just I_REPHRASE_PROMPT).
    """
    print("\n  === HONEST EVALUATION (vector vs all test sentences) ===", flush=True)
    print(f"  {'Target':<40} {'AUROC':>7}  {'F1':>6}  {'n_pos':>6}  {'n_neg':>6}  {'n_neg_was_before':>16}", flush=True)
    print(f"  {'-'*90}", flush=True)

    all_labels = test_res.labels()
    results = {}

    for target in SAFETY_TARGETS:
        if target not in all_labels:
            continue

        vec = build_vector(train_res, target, baseline, spec, layer)

        # All sentences in test set
        all_X, all_y = [], []
        for lbl in all_labels:
            tensors = test_res.get(spec, lbl, layer)
            if not tensors:
                continue
            X = stack(tensors)
            y = np.ones(len(X)) if lbl == target else np.zeros(len(X))
            all_X.append(X)
            all_y.append(y)

        if not all_X:
            continue
        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        sims = cosim(X_all, vec)

        n_pos = int(y_all.sum())
        n_neg = int((y_all == 0).sum())
        if n_pos < 2:
            continue

        auroc = float(roc_auc_score(y_all, sims))

        thresholds = np.percentile(sims, np.linspace(0, 100, 200))
        best_f1, best_thr = 0.0, thresholds[0]
        for thr in thresholds:
            f1 = float(f1_score(y_all, (sims >= thr).astype(int), zero_division=0))
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)

        # Compare to the "biased" evaluation (just baseline negatives)
        neg_baseline = test_res.get(spec, baseline, layer)
        n_neg_before = len(neg_baseline)

        print(f"  {target:<40} {auroc:>7.4f}  {best_f1:>6.4f}  {n_pos:>6}  {n_neg:>6}  {n_neg_before:>16}", flush=True)
        results[target] = {"auroc": auroc, "f1": best_f1, "sims": sims, "y": y_all,
                           "n_pos": n_pos, "n_neg": n_neg}

    # ROC curve plot
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (target, res) in zip(axes, results.items()):
        fpr, tpr, _ = roc_curve(res["y"], res["sims"])
        ax.plot(fpr, tpr, lw=2, color="#d62728",
                label=f"AUROC={res['auroc']:.3f}, F1={res['f1']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("FPR", fontsize=9)
        ax.set_ylabel("TPR", fontsize=9)
        ax.set_title(target.replace("_", "\n"), fontsize=7)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"HONEST ROC curves (vs ALL test sentences)\nSpec={spec}, Layer={layer}, Baseline={baseline}",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}", flush=True)
    return results


# ── PLOT 4: Token position comparison ─────────────────────────────────────────

def plot_token_position_comparison(test_res: ExtractionResult, train_res: ExtractionResult,
                                    target: str, baseline: str, layer: int, out_path: Path):
    """Compare whole_sentence (all tokens mean) vs last_3 for the same data."""
    specs = [s for s in test_res.specs() if s in train_res.specs()]
    print(f"\n  === TOKEN POSITION COMPARISON for {target} ===", flush=True)

    all_labels = test_res.labels()
    fig, axes = plt.subplots(1, len(specs), figsize=(6 * len(specs), 4), sharey=False)
    if len(specs) == 1:
        axes = [axes]

    for ax, spec in zip(axes, specs):
        vec = build_vector(train_res, target, baseline, spec, layer)

        label_sims = {}
        for lbl in sorted(all_labels):
            tensors = test_res.get(spec, lbl, layer)
            if len(tensors) < 3:
                continue
            X = stack(tensors)
            sims = cosim(X, vec)
            label_sims[lbl] = sims

        labels_sorted = sorted(label_sims.keys(), key=lambda l: np.median(label_sims[l]), reverse=True)
        data = [label_sims[l] for l in labels_sorted]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        for lbl, patch in zip(labels_sorted, bp["boxes"]):
            patch.set_facecolor("#d62728" if lbl == target else "#aec7e8")
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(labels_sorted) + 1))
        ax.set_xticklabels([l.replace("_", "\n") for l in labels_sorted], fontsize=6)
        ax.set_title(f"Spec: {spec}", fontsize=10)
        ax.set_ylabel("Cosine similarity", fontsize=9)

        # Print AUROCs
        X_pos = stack(test_res.get(spec, target, layer)) if target in all_labels else None
        if X_pos is not None:
            sims_pos = cosim(X_pos, vec)
            for lbl in labels_sorted:
                if lbl != target:
                    X_neg = stack(test_res.get(spec, lbl, layer))
                    y = np.array([1]*len(sims_pos) + [0]*len(X_neg))
                    s = np.concatenate([sims_pos, cosim(X_neg, vec)])
                    if len(set(y)) == 2:
                        print(f"    {spec} vs {lbl}: AUROC={roc_auc_score(y, s):.4f} (n_pos={len(sims_pos)}, n_neg={len(X_neg)})", flush=True)

    fig.suptitle(
        f"Token position comparison: '{target}'\nBaseline={baseline}, Layer={layer}",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"  Saved: {out_path}", flush=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    import torch
    device_str = f"CUDA ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device_str}")

    # Load cached extractions
    print("Loading train extraction cache...", flush=True)
    train_res = ExtractionResult.load(str(TRAIN_CACHE))
    print("Loading test extraction cache...", flush=True)
    test_res = ExtractionResult.load(str(TEST_CACHE))

    # ── Load model for per-conversation extraction ────────────────────────────
    print("\nLoading model for per-conversation extraction...", flush=True)
    model = LittleSteerModel(
        model_id="Qwen/Qwen3.5-4B",
        use_pretrained_loading=True,
        allow_multimodal=True,
        check_renaming=False,
    )
    extractor = ActivationExtractor(model)

    # Pick a conversation that has a good number of target annotations
    entries = load_dataset(str(DATA_PATH))
    # Skip first 50 (training), find a test conv with many target annotations
    rng = np.random.default_rng(42)
    shuffled = list(entries)
    rng.shuffle(shuffled)
    new_entries = shuffled[50:]

    target = "II_STATE_ETHICAL_MORAL_CONCERN"
    good_conv = None
    for e in new_entries[:30]:
        n_target = sum(1 for a in e.annotations if target in (a.labels or []))
        if n_target >= 3:
            good_conv = e
            break

    print(f"\nSelected conversation: {good_conv.id} ({sum(1 for a in good_conv.annotations if target in (a.labels or []))} target annotations, {len(good_conv.annotations)} total)", flush=True)

    # ── PLOT 1: Per-conversation similarity ──────────────────────────────────
    print(f"\n[PLOT 1] Per-conversation sentence similarity", flush=True)
    plot_conversation_similarity(
        good_conv, extractor, train_res, target, SPEC, LAYER, BASELINE,
        PLOTS_DIR / f"plot1_conv_similarity_{target[:20]}.png"
    )

    del model  # free GPU memory

    # ── PLOT 2: Distribution across ALL label categories ─────────────────────
    print(f"\n[PLOT 2] Distribution of similarities by label category", flush=True)
    for tgt in SAFETY_TARGETS:
        if tgt in test_res.labels():
            plot_similarity_distributions(
                test_res, train_res, tgt, SPEC, LAYER, BASELINE,
                PLOTS_DIR / f"plot2_distribution_{tgt[:30]}.png"
            )

    # ── PLOT 3: HONEST evaluation (all sentences) ────────────────────────────
    print(f"\n[PLOT 3] Honest evaluation (vector vs ALL test sentences)", flush=True)
    honest_eval_all_sentences(
        test_res, train_res, SPEC, LAYER, BASELINE,
        PLOTS_DIR / "plot3_honest_roc.png"
    )

    # ── PLOT 4: Token position comparison ────────────────────────────────────
    print(f"\n[PLOT 4] Token position comparison", flush=True)
    plot_token_position_comparison(
        test_res, train_res, target, BASELINE, LAYER,
        PLOTS_DIR / f"plot4_token_positions_{target[:20]}.png"
    )

    print(f"\n✅ All plots saved to {PLOTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
