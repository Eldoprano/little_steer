"""
little_steer.visualization.token_view

HTML-based token-level similarity visualizations for Jupyter notebooks.

Two rendering modes:
  - Single layer:  render_token_similarity_html(token_sims, layer=None)
  - Multi-layer:   render_multilayer_html(token_sims, layers=None)

High-level convenience:
  - show_token_similarity(model, entry, vector, ...) → IPython.display.HTML
"""

from __future__ import annotations

import html as html_module
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

if TYPE_CHECKING:
    from ..probing import TokenSimilarities
    from ..models.model import LittleSteerModel
    from ..vectors.steering_vector import SteeringVector
    from ..data.schema import ConversationEntry


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _sim_to_css_color(sim: float, vmin: float, vmax: float) -> str:
    """Map a similarity value to a CSS rgba string using the RdBu_r colormap."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    r, g, b, _ = cm.RdBu_r(norm(sim))
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},0.85)"


def _color_scale_css(vmin: float, vmax: float, n: int = 128) -> str:
    """Build a CSS linear-gradient string for the legend."""
    stops = []
    for i in range(n + 1):
        t = i / n
        sim = vmin + t * (vmax - vmin)
        color = _sim_to_css_color(sim, vmin, vmax)
        stops.append(f"{color} {t * 100:.0f}%")
    return f"linear-gradient(to right, {', '.join(stops)})"


# ---------------------------------------------------------------------------
# Label token map
# ---------------------------------------------------------------------------

def _label_token_set(token_sims: "TokenSimilarities", target_label: str) -> set:
    """Return set of token indices that fall inside a span with target_label."""
    labeled: set[int] = set()
    for ts in token_sims.token_spans:
        if target_label in ts.labels:
            labeled.update(range(ts.token_start, ts.token_end))
    return labeled


# ---------------------------------------------------------------------------
# Single-layer HTML rendering
# ---------------------------------------------------------------------------

def render_token_similarity_html(
    token_sims: "TokenSimilarities",
    layer: Optional[int] = None,
    show_labels: bool = True,
) -> str:
    """Render token-level cosine similarities as an HTML string.

    Each token is displayed as an inline ``<span>`` with a background colour
    from the ``RdBu_r`` diverging colormap:
      - **Red** → high positive similarity (behaviour likely present)
      - **White** → near zero
      - **Blue** → negative similarity (behaviour absent)

    Newlines in the formatted text are preserved so the layout matches the
    original conversation structure.  Annotated spans for the target label
    are underlined.  Hovering over a token shows its exact similarity value.

    Args:
        token_sims:  Output of :func:`~little_steer.probing.get_token_similarities`.
        layer:       Which layer's similarities to display.
                     Defaults to ``token_sims.layer`` (the vector's native layer).
        show_labels: If True, underline tokens inside annotated spans.

    Returns:
        HTML string — pass to ``IPython.display.HTML(...)`` to render in a notebook.
    """
    display_layer = layer if layer is not None else token_sims.layer
    sims = token_sims.similarities[display_layer]

    # Symmetric colour range
    abs_max = max(abs(min(sims)), abs(max(sims)), 1e-6)
    vmin, vmax = -abs_max, abs_max

    labeled_tokens = _label_token_set(token_sims, token_sims.label) if show_labels else set()
    text = token_sims.formatted_text
    char_spans = token_sims.token_char_spans
    n = len(token_sims.tokens)

    # Build legend gradient
    grad = _color_scale_css(vmin, vmax)
    legend_html = (
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;'
        f'font-size:12px;color:#555;">'
        f'<b style="white-space:nowrap">{html_module.escape(token_sims.label)}</b>'
        f'<span style="white-space:nowrap">· Layer {display_layer}</span>'
        f'<span style="white-space:nowrap">· [{vmin:.3f}, {vmax:.3f}]</span>'
        f'<div style="display:flex;align-items:center;gap:4px;flex-shrink:0;">'
        f'<span style="color:#2166ac;font-size:11px">absent</span>'
        f'<div style="width:80px;height:10px;background:{grad};border-radius:3px;border:1px solid #ccc;"></div>'
        f'<span style="color:#b2182b;font-size:11px">present</span>'
        f'</div>'
    )
    if show_labels and labeled_tokens:
        legend_html += (
            f'<span style="border-bottom:2px solid #2980b9;padding-bottom:1px;'
            f'font-size:11px;white-space:nowrap"> = labeled span</span>'
        )
    legend_html += "</div>"

    # Build token spans, preserving newlines
    token_parts: list[str] = []
    for i, (tok, sim) in enumerate(zip(token_sims.tokens, sims)):
        color = _sim_to_css_color(sim, vmin, vmax)
        is_labeled = i in labeled_tokens

        # Tooltip
        label_note = f" [{html_module.escape(token_sims.label)}]" if is_labeled else ""
        title = f"sim={sim:.4f}{label_note}"

        # Token display text
        display = html_module.escape(tok) if tok.strip() else "·"

        underline = (
            "border-bottom:2px solid #2980b9;padding-bottom:1px;"
            if is_labeled else ""
        )
        span = (
            f'<span style="background:{color};padding:1px 0;{underline}"'
            f' title="{title}">{display}</span>'
        )
        token_parts.append(span)

        # Emit newlines between tokens by checking the gap in formatted text
        if i < n - 1:
            end_cur = char_spans[i][1]
            start_next = char_spans[i + 1][0]
            if end_cur < start_next and "\n" in text[end_cur:start_next]:
                token_parts.append("\n")

    tokens_html = "".join(token_parts)

    return (
        f'<div style="font-family:monospace;font-size:13px;line-height:1.8;'
        f'padding:12px;background:#fafafa;border-radius:8px;border:1px solid #e0e0e0;">'
        f'{legend_html}'
        f'<div style="white-space:pre-wrap;word-break:break-word;">'
        f'{tokens_html}'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Multi-layer HTML rendering
# ---------------------------------------------------------------------------

def render_multilayer_html(
    token_sims: "TokenSimilarities",
    layers: Optional[List[int]] = None,
    max_display_tokens: int = 200,
) -> str:
    """Render a token × layer similarity heatmap as HTML.

    Shows the text at the top (coloured by the vector's native layer) and a
    compact grid below where each row is a transformer layer and each column
    is a token.  The same ``RdBu_r`` colour scale is used across all layers.

    Args:
        token_sims:         Output of :func:`~little_steer.probing.get_token_similarities`.
        layers:             Which layers to include in the grid.
                            Defaults to all layers present in ``token_sims.similarities``.
        max_display_tokens: Truncate the token axis to this length to keep the
                            grid manageable (default 200).

    Returns:
        HTML string — pass to ``IPython.display.HTML(...)`` to render in a notebook.
    """
    display_layers = sorted(layers if layers else token_sims.similarities.keys())
    n_tokens = len(token_sims.tokens)
    truncated = n_tokens > max_display_tokens
    display_n = min(n_tokens, max_display_tokens)

    # Global colour range across all layers
    all_sims: list[float] = []
    for layer_idx in display_layers:
        all_sims.extend(token_sims.similarities[layer_idx][:display_n])
    if not all_sims:
        return "<p>No similarity data to display.</p>"
    abs_max = max(abs(min(all_sims)), abs(max(all_sims)), 1e-6)
    vmin, vmax = -abs_max, abs_max

    labeled_tokens = _label_token_set(token_sims, token_sims.label)

    # ── Text reference (top section, using native layer) ────────────────────
    text_html = render_token_similarity_html(token_sims, show_labels=True)

    # ── Truncation notice ────────────────────────────────────────────────────
    trunc_note = ""
    if truncated:
        trunc_note = (
            f'<p style="font-size:11px;color:#888;margin:4px 0;">'
            f'Showing first {max_display_tokens} of {n_tokens} tokens.</p>'
        )

    # ── Header row: abbreviated token strings ────────────────────────────────
    header_cells = ['<th style="width:50px;text-align:right;padding-right:6px;'
                    'color:#aaa;font-weight:normal;vertical-align:bottom;">Layer</th>']
    tokens_display = token_sims.tokens[:display_n]
    for i, tok in enumerate(tokens_display):
        tok_abbr = html_module.escape(tok[:5].replace("\n", "↵").replace(" ", "·"))
        dot = (
            '<span style="color:#2980b9;font-size:8px;">●</span>'
            if i in labeled_tokens else ""
        )
        header_cells.append(
            f'<th style="writing-mode:vertical-rl;transform:rotate(180deg);'
            f'white-space:nowrap;padding:1px 2px;font-size:8px;color:#666;'
            f'font-weight:normal;max-height:50px;vertical-align:bottom;">'
            f'{dot}{tok_abbr}</th>'
        )
    header_row = "<tr>" + "".join(header_cells) + "</tr>"

    # ── Data rows: one per layer ──────────────────────────────────────────────
    data_rows: list[str] = []
    for layer_idx in display_layers:
        row_sims = token_sims.similarities[layer_idx][:display_n]
        cells = [
            f'<td style="text-align:right;padding-right:6px;font-size:10px;'
            f'color:#888;white-space:nowrap;">L{layer_idx}</td>'
        ]
        for i, sim in enumerate(row_sims):
            color = _sim_to_css_color(sim, vmin, vmax)
            cells.append(
                f'<td style="width:8px;height:14px;background:{color};'
                f'border:1px solid rgba(255,255,255,0.15);" '
                f'title="L{layer_idx} tok={i} sim={sim:.4f}"></td>'
            )
        data_rows.append("<tr>" + "".join(cells) + "</tr>")

    grid_html = (
        f'<div style="overflow-x:auto;margin-top:12px;">'
        f'{trunc_note}'
        f'<table style="border-collapse:collapse;font-family:monospace;font-size:9px;">'
        f'<thead>{header_row}</thead>'
        f'<tbody>{"".join(data_rows)}</tbody>'
        f'</table>'
        f'</div>'
    )

    return (
        f'<div style="font-family:monospace;">'
        f'{text_html}'
        f'{grid_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Convenience: show_token_similarity
# ---------------------------------------------------------------------------

def show_token_similarity(
    model: "LittleSteerModel",
    entry: "ConversationEntry",
    vector: "SteeringVector",
    layers: Optional[List[int]] = None,
    multilayer: bool = False,
):
    """Run one forward pass and display token-level similarities in a notebook.

    High-level convenience combining
    :func:`~little_steer.probing.get_token_similarities` with
    :func:`render_token_similarity_html` or :func:`render_multilayer_html`.

    Args:
        model:      LittleSteerModel instance.
        entry:      Conversation entry to visualise.
        vector:     Steering vector to probe against.
        layers:     Layer indices to collect and display.
                    Defaults to ``[vector.layer]``.
        multilayer: If True, render the multi-layer grid view.
                    If False (default), render the single-layer text view.

    Returns:
        ``IPython.display.HTML`` object — just return it from a notebook cell.
        Falls back to the raw HTML string if IPython is not available.

    Example:
        # Single-layer view
        show_token_similarity(model, entry, vec)

        # Multi-layer grid
        show_token_similarity(model, entry, vec, layers=list(range(0, 32, 2)), multilayer=True)
    """
    from ..probing import get_token_similarities

    token_sims = get_token_similarities(model, entry, vector, layers=layers)

    if multilayer:
        html_str = render_multilayer_html(token_sims, layers=layers)
    else:
        display_layer = layers[0] if layers else vector.layer
        html_str = render_token_similarity_html(token_sims, layer=display_layer)

    try:
        from IPython.display import HTML
        return HTML(html_str)
    except ImportError:
        return html_str
