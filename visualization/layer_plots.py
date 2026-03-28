"""
little_steer.visualization.layer_plots

Matplotlib-based plots for analysing steering vector performance across layers.

Functions:
    plot_layer_discrimination(scores, ...)  — discrimination (present−absent) vs layer
    plot_layer_metrics(eval_results, ...)   — AUROC / F1 / precision / recall vs layer
    plot_confusion_matrix(eval_result, ...) — 2×2 confusion matrix heatmap
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ..probing import BehaviorScore, EvaluationResult


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

_PALETTE = [
    "#2166ac", "#d6604d", "#4dac26", "#8073ac",
    "#f1a340", "#1a9850", "#d01c8b", "#b35806",
    "#80cdc1", "#dfc27d",
]
_LINESTYLES = ["-", "--", ":", "-."]
_SPINE_COLOR = "#aaa"


def _clean_axes(ax: "Axes") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_SPINE_COLOR)
    ax.spines["bottom"].set_color(_SPINE_COLOR)
    ax.tick_params(colors=_SPINE_COLOR, labelsize=9)
    ax.yaxis.label.set_color("#555")
    ax.xaxis.label.set_color("#555")
    ax.title.set_color("#333")


# ---------------------------------------------------------------------------
# plot_layer_discrimination
# ---------------------------------------------------------------------------

def plot_layer_discrimination(
    scores: List["BehaviorScore"],
    labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple["Figure", "Axes"]:
    """Line chart of steering vector discrimination across layers.

    Plots ``mean_present − mean_absent`` on the y-axis and layer index on the
    x-axis.  Each unique ``(label, method)`` combination becomes one line.
    Higher discrimination means the vector direction more strongly separates
    spans with the target label from those without it.

    Args:
        scores:  List of :class:`~little_steer.probing.BehaviorScore` objects,
                 as returned by :func:`~little_steer.probing.score_dataset`.
        labels:  If set, only plot lines for these labels.
        figsize: Figure size override.  Defaults to ``(10, 4)``.

    Returns:
        ``(fig, ax)`` — a matplotlib figure and axes.

    Example:
        scores = ls.score_dataset(model, dataset, vectors)
        fig, ax = ls.plot_layer_discrimination(scores)
        plt.show()
    """
    if labels:
        scores = [s for s in scores if s.label in labels]
    if not scores:
        fig, ax = plt.subplots(figsize=figsize or (10, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    # Group by (label, method)
    groups: dict[tuple, dict[int, float]] = defaultdict(dict)
    for s in scores:
        groups[(s.label, s.method)][s.layer] = s.discrimination

    fig, ax = plt.subplots(figsize=figsize or (10, 4))
    _clean_axes(ax)
    ax.axhline(0, color="#ddd", lw=0.8, ls="--", zorder=0)

    unique_labels = sorted(set(k[0] for k in groups))
    unique_methods = sorted(set(k[1] for k in groups))

    for (label, method), layer_disc in sorted(groups.items()):
        layers_sorted = sorted(layer_disc)
        disc_vals = [layer_disc[l] for l in layers_sorted]
        color = _PALETTE[unique_labels.index(label) % len(_PALETTE)]
        ls = _LINESTYLES[unique_methods.index(method) % len(_LINESTYLES)]
        ax.plot(layers_sorted, disc_vals, color=color, ls=ls, lw=1.5, marker="o",
                markersize=3, label=f"{label} [{method}]")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Discrimination (present − absent)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
              fontsize=8, labelcolor="#444")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_layer_metrics
# ---------------------------------------------------------------------------

def plot_layer_metrics(
    eval_results: List["EvaluationResult"],
    metric: str = "auroc",
    labels: Optional[List[str]] = None,
    aggregations: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple["Figure", "Axes"]:
    """Line chart of a classification metric across layers.

    Each ``(label, aggregation, method)`` combination becomes one line.
    Includes a reference horizontal line at the random-baseline level (0.5
    for AUROC, 0 for other metrics).

    Args:
        eval_results: List of :class:`~little_steer.probing.EvaluationResult`
                      objects, as returned by :func:`~little_steer.probing.evaluate_dataset`.
        metric:       One of ``'auroc'``, ``'f1'``, ``'precision'``, ``'recall'``.
        labels:       If set, only plot lines for these labels.
        aggregations: If set, only plot lines for these aggregation modes.
        figsize:      Figure size override.  Defaults to ``(10, 4)``.

    Returns:
        ``(fig, ax)`` — a matplotlib figure and axes.

    Example:
        results = ls.evaluate_dataset(model, dataset, vectors)
        fig, ax = ls.plot_layer_metrics(results, metric="auroc")
        plt.show()
    """
    valid = {"auroc", "f1", "precision", "recall"}
    if metric not in valid:
        raise ValueError(f"metric must be one of {valid}, got {metric!r}")

    filtered = eval_results
    if labels:
        filtered = [r for r in filtered if r.label in labels]
    if aggregations:
        filtered = [r for r in filtered if r.aggregation in aggregations]
    if not filtered:
        fig, ax = plt.subplots(figsize=figsize or (10, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig, ax

    # Group by (label, aggregation, method)
    groups: dict[tuple, dict[int, float]] = defaultdict(dict)
    for r in filtered:
        val = getattr(r, metric)
        if not np.isnan(val):
            groups[(r.label, r.aggregation, r.method)][r.layer] = val

    unique_labels = sorted(set(k[0] for k in groups))
    unique_aggs = sorted(set(k[1] for k in groups))

    fig, ax = plt.subplots(figsize=figsize or (10, 4))
    _clean_axes(ax)

    ref_line = 0.5 if metric == "auroc" else 0.0
    ax.axhline(ref_line, color="#ddd", lw=0.8, ls="--", zorder=0)

    for (label, agg, method), layer_vals in sorted(groups.items()):
        layers_sorted = sorted(layer_vals)
        vals = [layer_vals[l] for l in layers_sorted]
        color = _PALETTE[unique_labels.index(label) % len(_PALETTE)]
        ls_style = _LINESTYLES[unique_aggs.index(agg) % len(_LINESTYLES)]
        ax.plot(layers_sorted, vals, color=color, ls=ls_style, lw=1.5, marker="o",
                markersize=3, label=f"{label} [{agg}, {method}]")

    ax.set_xlabel("Layer")
    ax.set_ylabel(metric.upper())
    if metric == "auroc":
        ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False,
              fontsize=8, labelcolor="#444")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    eval_result: "EvaluationResult",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple["Figure", "Axes"]:
    """2×2 confusion matrix heatmap for one evaluation result.

    The matrix is oriented as ``[[TN, FP], [FN, TP]]`` — rows are the true
    class, columns are the predicted class.  Each cell shows the count and
    its percentage of the total.

    Args:
        eval_result: A single :class:`~little_steer.probing.EvaluationResult`.
        figsize:     Figure size override.  Defaults to ``(5, 4)``.

    Returns:
        ``(fig, ax)`` — a matplotlib figure and axes.

    Example:
        results = ls.evaluate_dataset(model, dataset, vectors)
        fig, ax = ls.plot_confusion_matrix(results[0])
        plt.show()
    """
    cm = eval_result.confusion_matrix  # [[TN, FP], [FN, TP]]
    total = cm.sum()

    fig, ax = plt.subplots(figsize=figsize or (5, 4))
    _clean_axes(ax)

    im = ax.imshow(cm, cmap="Blues", vmin=0)

    tick_labels = ["Absent (0)", "Present (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)

    title_lines = [
        f"{eval_result.label}",
        f"Layer {eval_result.layer} · {eval_result.aggregation} · "
        f"F1={eval_result.f1:.3f} · AUROC={eval_result.auroc:.3f}",
    ]
    ax.set_title("\n".join(title_lines), fontsize=9, color="#333", pad=8)

    # Annotate cells
    thresh = cm.max() * 0.6
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100 * count / total if total > 0 else 0
            text_color = "white" if count > thresh else "#333"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", color=text_color, fontsize=11)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Count")
    fig.tight_layout()
    return fig, ax
