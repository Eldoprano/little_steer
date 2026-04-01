#!/usr/bin/env python3
"""
Clean plots matching the little_steer notebook style.
  - AUROC by layer (honest: vector vs ALL sentence types)
  - Confusion matrix at best layer
  - Per-sentence similarity for one conversation

Run from: /home/eldoprano/Studium/GeneralCode/Reasoning_behaviours/
  python little_steer/scripts/make_plots.py
"""
from __future__ import annotations
import sys, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from little_steer import (
    LittleSteerModel, ActivationExtractor, ExtractionPlan, ExtractionSpec,
    TokenSelection, ExtractionResult, load_dataset,
)
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support, confusion_matrix

# ── Style (matching the library) ─────────────────────────────────────────────
_PALETTE = ["#2166ac", "#d6604d", "#4dac26", "#8073ac", "#f1a340", "#1a9850"]
_SPINE_COLOR = "#aaa"

def _clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_SPINE_COLOR)
    ax.spines["bottom"].set_color(_SPINE_COLOR)
    ax.tick_params(colors=_SPINE_COLOR, labelsize=9)
    ax.yaxis.label.set_color("#555")
    ax.xaxis.label.set_color("#555")
    ax.title.set_color("#333")

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR     = REPO_ROOT / "little_steer"
TRAIN_CACHE = OUT_DIR / "extraction_ii_state_ethical_moral_concern.pt"
TEST_CACHE  = OUT_DIR / "full_run_test.pt"
DATA_PATH   = REPO_ROOT / "data" / "little_steer_dataset.jsonl"
PLOTS_DIR   = OUT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

BASELINE = "I_REPHRASE_PROMPT"
TARGETS  = [
    "II_FLAG_PROMPT_AS_HARMFUL",
    "II_STATE_ETHICAL_MORAL_CONCERN",
    "II_STATE_LEGAL_CONCERN",
    "II_STATE_SAFETY_CONCERN",
    "IV_INTEND_REFUSAL_OR_SAFE_ACTION",
]

# ── Utilities ─────────────────────────────────────────────────────────────────
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

def build_vector(train_res, target, baseline, spec, layer) -> np.ndarray:
    pos = stack(train_res.get(spec, target, layer))
    neg = stack(train_res.get(spec, baseline, layer))
    return pos.mean(0) - neg.mean(0)

def honest_metrics(test_res, train_res, target, spec, layer):
    """Compute metrics where negatives = ALL other labeled sentences."""
    vec = build_vector(train_res, target, BASELINE, spec, layer)
    all_X, all_y = [], []
    for lbl in test_res.labels():
        tensors = test_res.get(spec, lbl, layer)
        if not tensors:
            continue
        X = stack(tensors)
        y = np.ones(len(X)) if lbl == target else np.zeros(len(X))
        all_X.append(X); all_y.append(y)
    if not all_X:
        return None
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    if y_all.sum() < 2:
        return None
    sims = cosim(X_all, vec)
    auroc = float(roc_auc_score(y_all, sims))
    thresholds = np.percentile(sims, np.linspace(0, 100, 200))
    best_f1, best_thr = 0.0, thresholds[0]
    for thr in thresholds:
        f1 = float(f1_score(y_all, (sims >= thr).astype(int), zero_division=0))
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    preds = (sims >= best_thr).astype(int)
    p, r, _, _ = precision_recall_fscore_support(y_all, preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_all, preds, labels=[0, 1])
    mean_pos = float(sims[y_all == 1].mean())
    mean_neg = float(sims[y_all == 0].mean())
    return dict(auroc=auroc, f1=best_f1, precision=float(p), recall=float(r),
                cm=cm, sims=sims, y=y_all, thr=best_thr,
                mean_pos=mean_pos, mean_neg=mean_neg,
                n_pos=int(y_all.sum()), n_neg=int((y_all==0).sum()))


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — AUROC by layer (honest)
# ═══════════════════════════════════════════════════════════════════════════════
def plot1_auroc_by_layer(train_res, test_res, spec="whole_sentence"):
    layers = sorted(set(train_res.layers()) & set(test_res.layers()))
    fig, ax = plt.subplots(figsize=(10, 4))
    _clean_axes(ax)
    ax.axhline(0.5, color="#ddd", lw=0.8, ls="--", zorder=0)

    for i, target in enumerate(TARGETS):
        if target not in test_res.labels():
            continue
        aurocs = []
        for l in layers:
            m = honest_metrics(test_res, train_res, target, spec, l)
            aurocs.append(m["auroc"] if m else float("nan"))
        label_short = target.replace("II_", "").replace("IV_", "").replace("_", " ").title()
        ax.plot(layers, aurocs, color=_PALETTE[i], lw=1.8, marker="o", markersize=4,
                label=label_short)

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC (honest: vs all sentences)")
    ax.set_title(f"Honest AUROC by layer — spec={spec}, baseline={BASELINE}", fontsize=10)
    ax.set_ylim(0.3, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8, labelcolor="#444")
    fig.tight_layout()
    path = PLOTS_DIR / "plot1_auroc_by_layer.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Discrimination (mean_pos − mean_neg) by layer
# ═══════════════════════════════════════════════════════════════════════════════
def plot2_discrimination(train_res, test_res, spec="whole_sentence"):
    layers = sorted(set(train_res.layers()) & set(test_res.layers()))
    fig, ax = plt.subplots(figsize=(10, 4))
    _clean_axes(ax)
    ax.axhline(0, color="#ddd", lw=0.8, ls="--", zorder=0)

    for i, target in enumerate(TARGETS):
        if target not in test_res.labels():
            continue
        discs = []
        for l in layers:
            m = honest_metrics(test_res, train_res, target, spec, l)
            discs.append(m["mean_pos"] - m["mean_neg"] if m else float("nan"))
        label_short = target.replace("II_", "").replace("IV_", "").replace("_", " ").title()
        ax.plot(layers, discs, color=_PALETTE[i], lw=1.8, marker="o", markersize=4,
                label=label_short)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Discrimination (sim_pos − sim_neg)")
    ax.set_title(f"Discrimination by layer — spec={spec}, baseline={BASELINE}", fontsize=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, fontsize=8, labelcolor="#444")
    fig.tight_layout()
    path = PLOTS_DIR / "plot2_discrimination.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Confusion matrices at best layer (honest)
# ═══════════════════════════════════════════════════════════════════════════════
def plot3_confusion_matrices(train_res, test_res, spec="whole_sentence"):
    layers = sorted(set(train_res.layers()) & set(test_res.layers()))

    valid_targets = [t for t in TARGETS if t in test_res.labels()]
    n = len(valid_targets)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, valid_targets):
        # Find best layer by honest AUROC
        best, best_l = None, None
        for l in layers:
            m = honest_metrics(test_res, train_res, target, spec, l)
            if m and (best is None or m["auroc"] > best["auroc"]):
                best, best_l = m, l

        if best is None:
            ax.axis("off"); continue

        cm = best["cm"]  # [[TN, FP], [FN, TP]]
        total = cm.sum()
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        tick_labels = ["Absent (0)", "Present (1)"]
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)

        label_short = target.replace("II_", "").replace("IV_", "").replace("_", "\n")
        ax.set_title(
            f"{label_short}\nLayer {best_l}  F1={best['f1']:.3f}  AUROC={best['auroc']:.3f}",
            fontsize=8, color="#333", pad=8
        )
        thresh = cm.max() * 0.6
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = 100 * count / total if total > 0 else 0
                ax.text(j, i, f"{count}\n({pct:.1f}%)",
                        ha="center", va="center",
                        color="white" if count > thresh else "#333", fontsize=10)
        _clean_axes(ax)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"Confusion matrices at best layer — honest eval (all sentence types as negatives)\nSpec={spec}, Baseline={BASELINE}", fontsize=10)
    fig.tight_layout()
    path = PLOTS_DIR / "plot3_confusion_matrices.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Per-conversation sentence similarity (single conv, GPU extraction)
# ═══════════════════════════════════════════════════════════════════════════════
def plot4_per_conv_similarity(train_res, extractor, target, spec, layer):
    entries = load_dataset(str(DATA_PATH))
    rng = np.random.default_rng(42)
    shuffled = list(entries); rng.shuffle(shuffled)
    new_entries = shuffled[50:]

    # Pick a test conversation with multiple target + non-target annotations
    conv = None
    for e in new_entries[:40]:
        n_pos = sum(1 for a in e.annotations if target in (a.labels or []))
        n_neg = sum(1 for a in e.annotations if target not in (a.labels or []))
        if n_pos >= 3 and n_neg >= 4:
            conv = e; break
    if conv is None:
        print("  No suitable conversation found"); return

    print(f"  Extracting conv {conv.id[:10]}... ({len(conv.annotations)} sentences)", flush=True)
    plan = ExtractionPlan(name="viz")
    plan.add_spec(spec, ExtractionSpec(TokenSelection("all", aggregation="mean"), layers=[layer]))
    result = extractor.extract([conv], plan)

    vec = build_vector(train_res, target, BASELINE, spec, layer)

    # Collect per-annotation similarity in conversation order
    rows = []
    label_counts = {}  # lbl -> list of tensors consumed
    for ann in sorted(conv.annotations, key=lambda a: (a.message_idx, a.char_start)):
        ann_labels = ann.labels or []
        found = False
        for lbl in ann_labels:
            tensors = result.get(spec, lbl, layer)
            if tensors:
                idx = label_counts.get(lbl, 0)
                if idx < len(tensors):
                    label_counts[lbl] = idx + 1
                    act = tensors[idx].float().numpy()
                    sim = float(cosim(act.reshape(1, -1), vec)[0])
                    is_pos = int(target in ann_labels)
                    rows.append((sim, is_pos, ann.text[:55].replace("\n", " ")))
                    found = True
                    break
        # if annotation found but not matching any layer tensor, skip

    if not rows:
        print("  No rows to plot"); return

    sims_arr = np.array([r[0] for r in rows])
    labels_arr = [r[1] for r in rows]
    texts = [r[2] for r in rows]
    xs = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.55), 5))
    _clean_axes(ax)

    colors = [_PALETTE[1] if l == 1 else _PALETTE[0] for l in labels_arr]
    ax.bar(xs, sims_arr, color=colors, alpha=0.85, edgecolor="none")
    ax.axhline(0, color="black", lw=0.6, ls="--")

    # Draw threshold line (mean of the vector's training distribution)
    pos_sims = sims_arr[np.array(labels_arr) == 1]
    neg_sims = sims_arr[np.array(labels_arr) == 0]
    thr = (pos_sims.mean() + neg_sims.mean()) / 2 if len(pos_sims) > 0 and len(neg_sims) > 0 else None
    if thr is not None:
        ax.axhline(thr, color="#888", lw=1, ls=":", label=f"midpoint thr={thr:.2f}")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}" for i in xs], fontsize=7)
    ax.set_xlabel("Sentence index (in conversation order)")
    ax.set_ylabel("Cosine similarity to steering vector")
    label_short = target.replace("_", " ")
    ax.set_title(
        f"Per-sentence similarity — conv {conv.id[:10]}\n"
        f"Target: {label_short}   Layer={layer}  Spec={spec}  Baseline={BASELINE}",
        fontsize=9
    )
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=_PALETTE[1], label=f"Positive (n={sum(labels_arr)})"),
        Patch(color=_PALETTE[0], label=f"Other labels (n={len(labels_arr)-sum(labels_arr)})"),
    ] + ([ax.lines[-1]] if thr is not None else []), fontsize=9, frameon=False)

    # Annotate sentences below x-axis
    for xi, (sim, lbl, txt) in enumerate(zip(sims_arr, labels_arr, texts)):
        ax.text(xi, ax.get_ylim()[0] - 0.01, txt, rotation=45, ha="right", va="top",
                fontsize=5.5, color=_PALETTE[1] if lbl else _PALETTE[0])

    fig.tight_layout()
    path = PLOTS_DIR / "plot4_per_sentence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    import torch
    print(f"Device: {'CUDA ('+torch.cuda.get_device_name(0)+')' if torch.cuda.is_available() else 'CPU'}")

    train_res = ExtractionResult.load(str(TRAIN_CACHE))
    test_res  = ExtractionResult.load(str(TEST_CACHE))

    print("\n[1] AUROC by layer")
    plot1_auroc_by_layer(train_res, test_res)

    print("[2] Discrimination by layer")
    plot2_discrimination(train_res, test_res)

    print("[3] Confusion matrices")
    plot3_confusion_matrices(train_res, test_res)

    print("[4] Per-conversation similarity (loading model...)")
    model = LittleSteerModel("Qwen/Qwen3.5-4B", use_pretrained_loading=True,
                              allow_multimodal=True, check_renaming=False)
    extractor = ActivationExtractor(model)
    plot4_per_conv_similarity(train_res, extractor, "II_STATE_ETHICAL_MORAL_CONCERN",
                               "whole_sentence", layer=22)
    del model

    print(f"\nAll plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
