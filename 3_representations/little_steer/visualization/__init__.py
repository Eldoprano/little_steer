"""
little_steer.visualization — Notebook-friendly visualizations for steering vectors.

Token view (HTML, displayable in Jupyter):
    show_token_similarity(model, entry, vector, ...)
    render_token_similarity_html(token_sims, layer=None, show_labels=True)
    render_multilayer_html(token_sims, layers=None, max_display_tokens=200)

Layer plots (matplotlib figures):
    plot_layer_discrimination(scores, ...)
    plot_layer_metrics(eval_results, metric="auroc", ...)
    plot_confusion_matrix(eval_result, ...)
"""

from .token_view import (
    render_token_similarity_html,
    render_multilayer_html,
    show_token_similarity,
)
from .layer_plots import (
    plot_layer_discrimination,
    plot_layer_metrics,
    plot_confusion_matrix,
    plot_vector_similarity,
)
from .probe_view import render_probe_detection_html, legend_html

__all__ = [
    "render_token_similarity_html",
    "render_multilayer_html",
    "show_token_similarity",
    "plot_layer_discrimination",
    "plot_layer_metrics",
    "plot_confusion_matrix",
    "plot_vector_similarity",
    "render_probe_detection_html",
    "legend_html",
]
