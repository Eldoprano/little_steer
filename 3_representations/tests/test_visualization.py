"""
Tests for little_steer.visualization — no GPU required.

Tests colour mapping, HTML rendering, and matplotlib plots using mock data
so they can run without a model or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock objects
# ---------------------------------------------------------------------------

def make_token_similarities(n_tokens=20, layers=(5, 10, 15)):
    """Create a mock TokenSimilarities object without running a model."""
    from little_steer.probing import TokenSimilarities
    from little_steer.data.tokenizer_utils import TokenSpan

    tokens = [f"tok{i}" for i in range(n_tokens)]
    similarities = {
        layer: list(np.linspace(-0.5, 0.5, n_tokens) + 0.01 * layer)
        for layer in layers
    }
    token_char_spans = [(i * 5, i * 5 + 4) for i in range(n_tokens)]
    token_spans = [
        TokenSpan(token_start=0, token_end=5, labels=["TARGET_LABEL"]),
        TokenSpan(token_start=10, token_end=15, labels=["OTHER_LABEL"]),
    ]
    return TokenSimilarities(
        tokens=tokens,
        similarities=similarities,
        token_char_spans=token_char_spans,
        token_spans=token_spans,
        formatted_text="     ".join(tokens),  # spaces between
        label="TARGET_LABEL",
        layer=10,
    )


def make_behavior_scores():
    from little_steer.probing import BehaviorScore
    scores = []
    for layer in range(0, 30, 5):
        scores.append(BehaviorScore(
            label="I_REPHRASE_PROMPT",
            layer=layer,
            method="pca",
            mean_present=0.3 + 0.01 * layer,
            mean_absent=0.1,
            n_present=10,
            n_absent=20,
        ))
        scores.append(BehaviorScore(
            label="II_STATE_SAFETY_CONCERN",
            layer=layer,
            method="mean_difference",
            mean_present=0.2 + 0.005 * layer,
            mean_absent=0.05,
            n_present=8,
            n_absent=18,
        ))
    return scores


def make_evaluation_result():
    from little_steer.probing import EvaluationResult
    return EvaluationResult(
        label="I_REPHRASE_PROMPT",
        layer=15,
        method="pca",
        aggregation="mean",
        auroc=0.78,
        f1=0.65,
        precision=0.7,
        recall=0.6,
        threshold=0.25,
        confusion_matrix=np.array([[40, 10], [5, 45]]),
        mean_present=0.35,
        mean_absent=0.1,
        n_present=50,
        n_absent=50,
    )


# ---------------------------------------------------------------------------
# Token view tests
# ---------------------------------------------------------------------------

class TestColorMapping:
    def test_returns_rgba_string(self):
        from little_steer.visualization.token_view import _sim_to_css_color
        color = _sim_to_css_color(0.5, -1.0, 1.0)
        assert color.startswith("rgba(")
        assert color.endswith(")")

    def test_positive_is_reddish(self):
        from little_steer.visualization.token_view import _sim_to_css_color
        color = _sim_to_css_color(1.0, -1.0, 1.0)
        # RdBu_r: max → red → R high, B low
        # Parse rgba(R,G,B,A)
        parts = color.replace("rgba(", "").replace(")", "").split(",")
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        assert r > b, f"Expected reddish for sim=1.0, got r={r} b={b}"

    def test_negative_is_bluish(self):
        from little_steer.visualization.token_view import _sim_to_css_color
        color = _sim_to_css_color(-1.0, -1.0, 1.0)
        parts = color.replace("rgba(", "").replace(")", "").split(",")
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        assert b > r, f"Expected bluish for sim=-1.0, got r={r} b={b}"


class TestRenderSingleLayer:
    def test_produces_html(self):
        from little_steer.visualization.token_view import render_token_similarity_html
        ts = make_token_similarities()
        html = render_token_similarity_html(ts)
        assert isinstance(html, str)
        assert "<span" in html
        assert "background:" in html

    def test_contains_label(self):
        from little_steer.visualization.token_view import render_token_similarity_html
        ts = make_token_similarities()
        html = render_token_similarity_html(ts)
        assert "TARGET_LABEL" in html

    def test_contains_layer_info(self):
        from little_steer.visualization.token_view import render_token_similarity_html
        ts = make_token_similarities()
        html = render_token_similarity_html(ts, layer=5)
        assert "Layer 5" in html

    def test_labeled_tokens_get_underline(self):
        from little_steer.visualization.token_view import render_token_similarity_html
        ts = make_token_similarities()
        html = render_token_similarity_html(ts, show_labels=True)
        assert "border-bottom" in html

    def test_html_escaping(self):
        """Tokens with special HTML characters must be escaped."""
        from little_steer.visualization.token_view import render_token_similarity_html
        from little_steer.probing import TokenSimilarities
        from little_steer.data.tokenizer_utils import TokenSpan

        ts = TokenSimilarities(
            tokens=["<b>"],
            similarities={0: [0.3]},
            token_char_spans=[(0, 3)],
            token_spans=[],
            formatted_text="<b>",
            label="X",
            layer=0,
        )
        html = render_token_similarity_html(ts)
        assert "&lt;b&gt;" in html
        assert "<b>" not in html.split("TARGET")[0]  # not as raw tag in output


class TestRenderMultilayer:
    def test_produces_html_with_table(self):
        from little_steer.visualization.token_view import render_multilayer_html
        ts = make_token_similarities()
        html = render_multilayer_html(ts)
        assert "<table" in html

    def test_layer_labels_present(self):
        from little_steer.visualization.token_view import render_multilayer_html
        ts = make_token_similarities(layers=[5, 10, 15])
        html = render_multilayer_html(ts)
        assert "L5" in html
        assert "L10" in html
        assert "L15" in html

    def test_max_display_tokens_truncates(self):
        from little_steer.visualization.token_view import render_multilayer_html
        ts = make_token_similarities(n_tokens=50)
        html = render_multilayer_html(ts, max_display_tokens=10)
        # Should mention truncation
        assert "50" in html or "10" in html


# ---------------------------------------------------------------------------
# Layer plot tests
# ---------------------------------------------------------------------------

class TestPlotLayerDiscrimination:
    def test_returns_fig_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        from little_steer.visualization.layer_plots import plot_layer_discrimination
        scores = make_behavior_scores()
        fig, ax = plot_layer_discrimination(scores)
        assert fig is not None
        assert ax is not None

    def test_label_filter(self):
        import matplotlib
        matplotlib.use("Agg")
        from little_steer.visualization.layer_plots import plot_layer_discrimination
        scores = make_behavior_scores()
        fig, ax = plot_layer_discrimination(scores, labels=["I_REPHRASE_PROMPT"])
        # Only one label should be in legend
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert all("I_REPHRASE_PROMPT" in t for t in legend_texts)

    def test_empty_scores_no_crash(self):
        import matplotlib
        matplotlib.use("Agg")
        from little_steer.visualization.layer_plots import plot_layer_discrimination
        fig, ax = plot_layer_discrimination([])
        assert fig is not None


class TestPlotLayerMetrics:
    def test_returns_fig_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        from little_steer.visualization.layer_plots import plot_layer_metrics
        from little_steer.probing import EvaluationResult

        results = [
            EvaluationResult(
                label="I_REPHRASE_PROMPT", layer=l, method="pca", aggregation="mean",
                auroc=0.5 + 0.01 * l, f1=0.4, precision=0.5, recall=0.3,
                threshold=0.2, confusion_matrix=np.array([[5, 2], [1, 4]]),
                mean_present=0.3, mean_absent=0.1, n_present=5, n_absent=7,
            )
            for l in range(0, 30, 5)
        ]
        fig, ax = plot_layer_metrics(results, metric="auroc")
        assert fig is not None

    def test_invalid_metric_raises(self):
        import matplotlib
        matplotlib.use("Agg")
        from little_steer.visualization.layer_plots import plot_layer_metrics
        with pytest.raises(ValueError):
            plot_layer_metrics([], metric="invalid_metric")


class TestPlotConfusionMatrix:
    def test_returns_fig_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        from little_steer.visualization.layer_plots import plot_confusion_matrix
        er = make_evaluation_result()
        fig, ax = plot_confusion_matrix(er)
        assert fig is not None

    def test_title_contains_label(self):
        import matplotlib
        matplotlib.use("Agg")
        from little_steer.visualization.layer_plots import plot_confusion_matrix
        er = make_evaluation_result()
        fig, ax = plot_confusion_matrix(er)
        assert "I_REPHRASE_PROMPT" in ax.get_title()


class TestPlotVectorSimilarity:
    def test_returns_fig_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        from little_steer.visualization.layer_plots import plot_vector_similarity
        import torch
        sim_mat = np.array([[1.0, 0.5], [0.5, 1.0]])
        labels = ["vec_A|pca|L20|last", "vec_B|mean|L20|last"]
        fig, ax = plot_vector_similarity(sim_mat, labels)
        assert fig is not None
        assert ax is not None

    def test_empty_no_crash(self):
        import matplotlib
        matplotlib.use("Agg")
        from little_steer.visualization.layer_plots import plot_vector_similarity
        fig, ax = plot_vector_similarity(np.array([]).reshape(0, 0), [])
        assert fig is not None
