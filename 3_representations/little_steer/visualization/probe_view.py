"""
little_steer.visualization.probe_view

Self-contained HTML renderer for multi-label behavior detection.

Works with any detection source that produces per-token scores:
  - MLPProbe / LinearProbeMultilabel  →  sigmoid probabilities [0, 1]
  - SteeringVectorSet  →  cosine similarities [-1, 1], use normalize_scores=True

Two display modes:
  ``"token"``    — color each token independently by its highest-scoring label.
  ``"sentence"`` — aggregate scores across each annotated span and color the
                   whole span uniformly (uses token_spans for boundaries).

The output is a self-contained HTML string with no external dependencies.
Embed in Marimo with ``mo.Html(...)``, in a web page directly, or save to
a ``.html`` file.

Main entry point:
    render_probe_detection_html(tokens, token_char_spans, scores, labels, ...)
"""

from __future__ import annotations

import html as html_module
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..data.tokenizer_utils import TokenSpan


# ---------------------------------------------------------------------------
# Fixed label → RGB color palette for all 25 v6 taxonomy labels
# ---------------------------------------------------------------------------
# Colors are given as "R, G, B" strings so they can be used inside rgba(...)

_LABEL_PALETTE: dict[str, str] = {
    # Group A: Prompt and User  (warm orange/amber)
    "rephrasePrompt":             "210, 130,  70",
    "speculateUserMotive":        "220, 160,  60",
    "flagEvaluationAwareness":    "200, 100,  80",
    "reframeTowardSafety":        "215, 185,  60",
    # Group B: Norms and Values  (red / rose / purple)
    "flagAsHarmful":              "210,  70,  70",
    "enumerateHarms":             "200,  80, 100",
    "stateSafetyConcern":         "195,  80, 130",
    "stateLegalConcern":          "185,  80, 160",
    "stateEthicalConcern":        "170,  80, 180",
    "referenceOwnPolicy":         "155,  80, 195",
    "cautiousFraming":            "140,  80, 210",
    # Group C: Facts and Content  (cyan / blue)
    "stateFactOrKnowledge":       " 60, 160, 210",
    "stateFalseClaim":            " 70, 130, 210",
    "detailHarmfulMethod":        " 50, 190, 190",
    "noteRiskWhileDetailingHarm": " 50, 210, 160",
    # Group D: Response Planning  (green)
    "intendRefusal":              " 60, 185, 100",
    "intendHarmfulCompliance":    " 90, 190,  80",
    "planResponseStructure":      " 50, 175, 130",
    "suggestSafeAlternative":     " 50, 160, 150",
    "produceResponseDraft":       "120, 200,  70",
    # Group E: Thinking Process  (indigo / muted steel)
    "expressUncertainty":         "120, 120, 210",
    "selfCorrect":                "140, 100, 215",
    "planReasoningStep":          "100, 140, 210",
    "summarizeReasoning":         " 80, 150, 205",
    "neutralFiller":              "160, 170, 170",
}

_FALLBACK_COLORS = [
    " 90, 120, 200",
    "200, 120,  90",
    " 90, 200, 120",
    "200,  90, 170",
    "120, 200, 200",
    "200, 200,  90",
    "160,  90, 200",
    " 90, 160, 200",
]


def _label_rgb(label: str, idx: int) -> str:
    return _LABEL_PALETTE.get(label, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def _normalize_scores_per_label(scores: np.ndarray) -> np.ndarray:
    """Per-label min-max normalization → [0, 1]."""
    out = np.empty_like(scores, dtype=np.float32)
    for j in range(scores.shape[1]):
        col = scores[:, j]
        lo, hi = float(col.min()), float(col.max())
        if hi > lo:
            out[:, j] = (col - lo) / (hi - lo)
        else:
            out[:, j] = 0.5
    return out


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------

def legend_html(labels: list[str], threshold: float) -> str:
    return _legend_html(labels, threshold)


def _legend_html(labels: list[str], threshold: float) -> str:
    swatches = []
    for idx, label in enumerate(labels):
        rgb = _label_rgb(label, idx)
        swatches.append(
            f'<span style="display:inline-flex;align-items:center;gap:3px;'
            f'margin:2px 6px 2px 0;font-size:11px;white-space:nowrap;">'
            f'<span style="display:inline-block;width:11px;height:11px;'
            f'border-radius:3px;background:rgba({rgb},0.9);flex-shrink:0;'
            f'border:1px solid rgba(0,0,0,0.15);"></span>'
            f'{html_module.escape(label)}'
            f'</span>'
        )
    return (
        f'<div class="ls-legend" style="display:flex;flex-wrap:wrap;align-items:center;'
        f'padding:8px 10px;margin-bottom:8px;background:#f5f5f5;'
        f'border-radius:6px;border:1px solid #e0e0e0;">'
        f'<style>'
        f'@media (prefers-color-scheme: dark) {{'
        f'.ls-legend {{ background: #252525 !important; border-color: #444 !important; }}'
        f'.ls-legend span {{ color: #aaa !important; }}'
        f'}}'
        f'</style>'
        f'<span style="font-size:11px;color:#888;margin-right:10px;white-space:nowrap;">'
        f'threshold&nbsp;≥&nbsp;<b>{threshold:.2f}</b></span>'
        + "".join(swatches)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_probe_detection_html(
    tokens: List[str],
    token_char_spans: List[Tuple[int, int]],
    scores: np.ndarray,
    labels: List[str],
    formatted_text: str,
    *,
    token_spans: Optional[List] = None,
    threshold: float = 0.4,
    mode: Literal["token", "sentence"] = "token",
    show_ground_truth: bool = True,
    normalize_scores: bool = True,
    section_markers: Optional[dict] = None,
    show_legend: bool = True,
    show_header: bool = True,
) -> str:
    """Render multi-label behavior detection as a self-contained HTML string.

    Accepts scores from any source — MLPProbe probabilities, linear probe
    outputs, or cosine similarities from steering vectors — as long as the
    array has shape ``(seq_len, n_labels)``.

    Args:
        tokens:           Decoded token strings, length = seq_len.
        token_char_spans: (char_start, char_end) per token in ``formatted_text``.
        scores:           ``(seq_len, n_labels)`` float array.
                          Probe path → sigmoid probabilities in [0, 1].
                          Vector path → cosine similarities in [-1, 1].
        labels:           Label names matching the columns of ``scores``.
        formatted_text:   Full chat-formatted text (preserves newlines in the output).
        token_spans:      Ground-truth annotation spans (from TokenPositionMapper).
                          Used for underlines (both modes) and sentence-level grouping.
        threshold:        Minimum score to assign a label color.  Tokens/spans where
                          all label scores fall below this stay uncolored.
        mode:             ``"token"``  — color each token by its top label score.
                          ``"sentence"`` — average scores across each span's tokens
                          and color the whole span uniformly.  Requires ``token_spans``.
        show_ground_truth: Draw a thin underline under ground-truth annotated spans.
        normalize_scores: If True (default), apply per-label min-max normalization
                          so the threshold is always interpreted in [0, 1].
                          Set to False if your scores are already probabilities.
        section_markers:  Optional ``{token_index: label_str}`` dict.  Before each
                          keyed token position a section header is inserted (e.g.
                          ``{0: "System", 12: "User", 40: "Reasoning", 80: "Response"}``).
        show_legend:      Whether to include the color legend (default True).  Set to
                          False when sharing one legend across multiple panels.

    Returns:
        Self-contained HTML string.  Render with ``mo.Html(...)`` in Marimo,
        embed directly in a web page, or save to a ``.html`` file.
    """
    if len(tokens) == 0:
        return "<p style='color:#888;font-family:monospace;'>No tokens to display.</p>"

    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError(
            f"scores must be (seq_len, n_labels), got {scores.shape} "
            f"with {len(labels)} labels."
        )

    disp = _normalize_scores_per_label(scores) if normalize_scores else scores.copy()
    n = len(tokens)

    # ── Resolve spans and mode ────────────────────────────────────────────
    # Case-insensitive robust check for 'sentence' in mode string
    _is_sentence = mode is None or "sentence" in str(mode).lower()
    
    # If in sentence mode but no spans provided, generate them automatically
    if _is_sentence and not token_spans:
        generated_spans = []
        start = 0
        from collections import namedtuple
        _TS = namedtuple("TokenSpan", ["token_start", "token_end", "labels"])
        for i, tok in enumerate(tokens):
            if "\n" in tok or (tok.strip() and tok.strip()[-1] in (".", "!", "?")):
                generated_spans.append(_TS(token_start=start, token_end=i+1, labels=[]))
                start = i + 1
        if start < len(tokens):
            generated_spans.append(_TS(token_start=start, token_end=len(tokens), labels=[]))
        token_spans = generated_spans

    # ── Per-token display color ────────────────────────────────────────────
    color_info: list[tuple[str, float, str] | None] = [None] * n
    
    # If in token mode, fill per-token colors first
    if not _is_sentence:
        for i in range(min(n, disp.shape[0])):
            row = disp[i]
            best_j = int(np.argmax(row))
            best_score = float(row[best_j])
            if best_score >= threshold:
                rgb = _label_rgb(labels[best_j], best_j)
                span_above = max(1.0 - threshold, 1e-6)
                alpha = 0.30 + 0.55 * min(1.0, (best_score - threshold) / span_above)
                color_info[i] = (rgb, alpha, labels[best_j])

    # ── Sentence mode: override with span-level aggregates ────────────────
    if _is_sentence and token_spans:
        for ts in token_spans:
            t_start = max(0, ts.token_start)
            t_end = min(n, ts.token_end)
            if t_start >= t_end or t_start >= disp.shape[0]:
                continue
            eff_end = min(t_end, disp.shape[0])
            
            # Average scores across the entire span
            mean_row = disp[t_start:eff_end].mean(axis=0)  # (n_labels,)
            disp[t_start:eff_end] = mean_row
            
            # Tie-break: np.argmax returns first index (label order)
            best_j = int(np.argmax(mean_row))
            best_score = float(mean_row[best_j])
            
            if best_score >= threshold:
                rgb = _label_rgb(labels[best_j], best_j)
                span_above = max(1.0 - threshold, 1e-6)
                alpha = 0.30 + 0.55 * min(1.0, (best_score - threshold) / span_above)
                fill = (rgb, alpha, labels[best_j])
            else:
                fill = None
                
            for i in range(t_start, t_end):
                color_info[i] = fill

    # ── Ground-truth token index set ──────────────────────────────────────
    gt_tokens: set[int] = set()
    if show_ground_truth and token_spans:
        for ts in token_spans:
            for i in range(max(0, ts.token_start), min(n, ts.token_end)):
                gt_tokens.add(i)

    # ── Tooltip per token ─────────────────────────────────────────────────
    def _tooltip_content(i: int) -> str:
        if i >= disp.shape[0]:
            return ""
        
        # Get all labels above threshold, sorted by score descending, then by index ascending
        active = []
        for j, lbl in enumerate(labels):
            s = float(disp[i, j])
            if s >= threshold:
                active.append((s, lbl, j))
        # Sort by score descending (-x[0]), then by original index ascending (x[2])
        active.sort(key=lambda x: (-x[0], x[2]))
        active = active[:3]  # top 3 only to keep HTML size manageable
        
        if not active:
            return ""
            
        parts = []
        for s, lbl, j in active:
            rgb = _label_rgb(lbl, j)
            parts.append(
                f'<div style="display:flex;align-items:center;gap:6px;margin:2px 0;">'
                f'<span style="display:inline-block;width:8px;height:8px;border-radius:2px;'
                f'background:rgb({rgb});border:1px solid rgba(0,0,0,0.1);"></span>'
                f'<span>{html_module.escape(lbl)}: <b>{s:.3f}</b></span>'
                f'</div>'
            )
        return "".join(parts)

    # ── Resolve section_markers ───────────────────────────────────────────
    _section_markers: dict[int, str] = section_markers or {}

    # ── Build token HTML ──────────────────────────────────────────────────
    parts: list[str] = []
    for i, tok in enumerate(tokens):
        if i in _section_markers:
            sec_label = html_module.escape(_section_markers[i])
            parts.append(
                f'<div class="ls-section-header" style="width:100%;margin:10px 0 4px 0;padding:3px 8px;'
                f'font-size:10px;font-weight:600;letter-spacing:0.08em;'
                f'text-transform:uppercase;color:#666;border-left:3px solid #aaa;'
                f'background:#efefef;border-radius:0 3px 3px 0;">'
                f'{sec_label}</div>'
                f'<style>'
                f'@media (prefers-color-scheme: dark) {{'
                f'.ls-section-header {{ background: #2e2e2e !important; border-left-color: #666 !important; color: #999 !important; }}'
                f'}}'
                f'</style>'
            )
        ci = color_info[i]
        is_gt = i in gt_tokens

        underline = (
            "border-bottom:2px solid rgba(50,50,50,0.45);padding-bottom:1px;"
            if is_gt else ""
        )
        bg = f"rgba({ci[0]},{ci[1]:.2f})" if ci else "transparent"
        
        # Tooltip data
        tooltip_html = _tooltip_content(i)
        
        # Use data attributes for a custom CSS-based tooltip
        # We also keep 'title' as a fallback if the custom tooltip isn't implemented
        display_token = html_module.escape(tok) if tok.strip() else "·"
        
        # Handle line breaks: if token is exactly a newline, we add it as a <br>
        # but also keep it for white-space: pre-wrap
        if tok == "\n":
             parts.append('<br/>')
             continue
        elif tok == "\r\n":
             parts.append('<br/>')
             continue

        if tooltip_html:
             # Wrap in a container for the tooltip
             parts.append(
                f'<span class="ls-token" style="background:{bg};padding:1px 0;{underline}'
                f'border-radius:2px;position:relative;cursor:help;">'
                f'{display_token}'
                f'<span class="ls-tooltip">{tooltip_html}</span>'
                f'</span>'
             )
        else:
             parts.append(
                f'<span style="background:{bg};padding:1px 0;{underline}'
                f'border-radius:2px;">{display_token}</span>'
             )

        # Preserve newlines between tokens that might have been stripped by the tokenizer
        if i < n - 1:
            end_cur = token_char_spans[i][1]
            start_next = token_char_spans[i + 1][0]
            if end_cur < start_next:
                gap_text = formatted_text[end_cur:start_next]
                if "\n" in gap_text:
                    parts.append("<br/>" * gap_text.count("\n"))

    mode_label = "sentence-level" if mode == "sentence" else "token-level"
    legend = _legend_html(labels, threshold) if show_legend else ""
    header = (
        f'<div style="margin-bottom:6px;font-size:11px;color:#888;">'
        f'Behavior detection &nbsp;·&nbsp; {mode_label} &nbsp;·&nbsp; '
        f'{len(labels)} label{"s" if len(labels) != 1 else ""}'
        f'</div>'
    ) if show_header else ""
    tokens_html = "".join(parts)

    styles = """
    <style>
    .ls-token:hover .ls-tooltip { display: block; }
    .ls-tooltip {
        display: none;
        position: absolute;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        background: #fff;
        color: #333;
        padding: 8px 12px;
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15), 0 0 0 1px rgba(0,0,0,0.05);
        z-index: 1000;
        min-width: 180px;
        pointer-events: none;
        font-family: sans-serif;
        font-size: 11px;
    }
    .ls-tooltip::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #fff transparent transparent transparent;
    }
    .ls-container {
        line-height: 1.8;
        white-space: pre-wrap;
        word-break: break-word;
        padding: 12px;
        background: #fafafa;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        color: #222;
    }
    @media (prefers-color-scheme: dark) {
        .ls-container {
            background: #1e1e1e;
            border-color: #444;
            color: #ddd;
        }
        .ls-tooltip {
            background: #2a2a2a;
            color: #e0e0e0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.08);
        }
        .ls-tooltip::after {
            border-color: #2a2a2a transparent transparent transparent;
        }
    }
    </style>
    """

    return (
        f'{styles}'
        f'<div style="font-family:monospace;font-size:13px;">'
        f'{header}'
        f'{legend}'
        f'<div class="ls-container">'
        f'{tokens_html}'
        f'</div>'
        f'</div>'
    )
