"""
little_steer.extraction.config

Configuration for what and how to extract activations from labeled spans.

TokenSelection design:
  - bleed_before / bleed_after FIRST shape/window the span boundaries
    (positive = extend outward, negative = shrink inward / skip tokens)
  - The extraction strategy then picks from WITHIN that shaped window
  - This makes it intuitive: bleed shapes the sentence, strategy picks from it

Example:
  TokenSelection("first", bleed_before=-2)
    → Skip 2 tokens from the start, then take the first token of what remains.

  TokenSelection("all", bleed_before=5, bleed_after=2)
    → Extend the window 5 tokens left and 2 tokens right, take all of them.

  TokenSelection("last_n", n=3, bleed_after=-1)
    → Shrink 1 token from the end of the span, then take the last 3 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TokenSelection:
    """Specifies how to select tokens from a labeled span.

    Two-phase process:
      Phase 1 — Bleed: reshape the span boundaries.
        bleed_before > 0: extend window left by N tokens (add context before)
        bleed_before < 0: shrink window by skipping the first N tokens
        bleed_after  > 0: extend window right by N tokens (add context after)
        bleed_after  < 0: shrink window by skipping the last N tokens

      Phase 2 — Strategy: pick tokens from the shaped window.
        "first"   — first token of the window
        "last"    — last token of the window
        "all"     — all tokens in the window (mean-aggregated or kept raw)
        "first_n" — first n tokens of the window (or all if window < n)
        "last_n"  — last n tokens of the window (or all if window < n)

    If after bleed the window is empty (start >= end / beyond sequence), the
    span is skipped with a warning but no error.

    Aggregation:
      "mean"  — average all selected tokens into one vector (default)
      "none"  — keep all selected tokens as a (n_tokens, hidden_dim) matrix
    """

    strategy: Literal["first", "last", "all", "first_n", "last_n"] = "all"
    n: int = 1
    """For first_n and last_n: number of tokens to take."""
    bleed_before: int = 0
    """Positive = extend left / add context. Negative = skip leading tokens."""
    bleed_after: int = 0
    """Positive = extend right / add context. Negative = skip trailing tokens."""
    aggregation: Literal[
        "mean",
        "none",
        "top_confident_mean",
        "entropy_weighted_mean",
        "decision_point_mean",
    ] = "mean"
    """How to aggregate selected tokens into a single vector.

    Probability-aware options (require a full model forward pass for logits):
      "top_confident_mean"   — mean over tokens where top-1 next-token prob >=
                               confidence_threshold. High confidence = clean signal.
      "entropy_weighted_mean" — weighted mean, weights = 1/entropy. Confident
                               positions contribute more, uncertain ones less.
      "decision_point_mean"  — mean over the highest-entropy tokens (top
                               high_entropy_fraction). High entropy = model is
                               choosing = potential behavioural fork.
    """
    confidence_threshold: float = 0.8
    """For top_confident_mean: minimum top-1 next-token probability to keep a token."""
    high_entropy_fraction: float = 0.5
    """For decision_point_mean: fraction of tokens to keep (highest entropy)."""

    @property
    def needs_logits(self) -> bool:
        """True if this aggregation strategy requires token probability distributions."""
        return self.aggregation in (
            "top_confident_mean",
            "entropy_weighted_mean",
            "decision_point_mean",
        )

    def apply(
        self,
        span_token_start: int,
        span_token_end: int,
        sequence_length: int,
    ) -> tuple[int, int] | None:
        """Compute the final (start, end) token indices to extract.

        Returns None if the window is empty after bleed (span will be skipped).

        Args:
            span_token_start: First token index of the annotated span.
            span_token_end: One-past-last token index of the annotated span.
            sequence_length: Total number of tokens in the sequence (for clamping).

        Returns:
            (start, end) token indices (exclusive end), or None if empty.
        """
        # Phase 1: Apply bleed to reshape window boundaries.
        # bleed_before > 0 → extend left (subtract from start)
        # bleed_before < 0 → shrink left (add to start = skip leading tokens)
        window_start = span_token_start - self.bleed_before
        window_end = span_token_end + self.bleed_after

        # Clamp to valid sequence range
        window_start = max(0, window_start)
        window_end = min(sequence_length, window_end)

        # Empty window → skip
        if window_start >= window_end:
            return None

        # Phase 2: Apply strategy within the shaped window
        window_len = window_end - window_start

        if self.strategy == "first":
            return (window_start, window_start + 1)

        elif self.strategy == "last":
            return (window_end - 1, window_end)

        elif self.strategy == "all":
            return (window_start, window_end)

        elif self.strategy == "first_n":
            actual_n = min(self.n, window_len)  # graceful for short spans
            return (window_start, window_start + actual_n)

        elif self.strategy == "last_n":
            actual_n = min(self.n, window_len)
            return (window_end - actual_n, window_end)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def describe(self) -> str:
        """Human-readable description of this selection."""
        parts = [f"strategy={self.strategy!r}"]
        if self.strategy in ("first_n", "last_n"):
            parts.append(f"n={self.n}")
        if self.bleed_before != 0:
            parts.append(f"bleed_before={self.bleed_before}")
        if self.bleed_after != 0:
            parts.append(f"bleed_after={self.bleed_after}")
        parts.append(f"aggregation={self.aggregation!r}")
        if self.aggregation == "top_confident_mean":
            parts.append(f"confidence_threshold={self.confidence_threshold}")
        elif self.aggregation == "decision_point_mean":
            parts.append(f"high_entropy_fraction={self.high_entropy_fraction}")
        return f"TokenSelection({', '.join(parts)})"

    def __repr__(self) -> str:
        return self.describe()


@dataclass
class ExtractionSpec:
    """Specifies WHAT to extract from each annotated span.

    Combines a TokenSelection (how to pick tokens) with a list of layers
    (which model layers to extract from).
    """

    token_selection: TokenSelection = field(default_factory=TokenSelection)
    """How to select and aggregate tokens from each span."""

    layers: list[int] | None = None
    """Which layer indices to extract from. None = all layers."""

    def __repr__(self) -> str:
        layer_str = "all" if self.layers is None else str(self.layers)
        return f"ExtractionSpec({self.token_selection!r}, layers={layer_str})"


@dataclass
class ExtractionPlan:
    """A named collection of ExtractionSpecs to run simultaneously.

    All specs share one model forward pass per conversation — the model runs
    ONCE and each spec slices/aggregates the cached activations differently.
    This is the key efficiency win: N specs cost 1x model pass, not Nx.

    Example:
        plan = ExtractionPlan("experiment_v1", specs={
            "last_token": ExtractionSpec(
                token_selection=TokenSelection("last"),
                layers=[20, 25]
            ),
            "whole_sentence": ExtractionSpec(
                token_selection=TokenSelection("all"),
                layers=[20, 25]
            ),
            "bleed_5_all": ExtractionSpec(
                token_selection=TokenSelection("all", bleed_before=5, bleed_after=5),
                layers=[20, 25]
            ),
        })
    """

    name: str
    """Human-readable name for this extraction plan."""

    specs: dict[str, ExtractionSpec] = field(default_factory=dict)
    """Named extraction specs. Keys become axes in ExtractionResult."""

    label_filter: list[str] | None = None
    """If set, only extract annotations with at least one of these labels."""

    def all_required_layers(self) -> list[int]:
        """Union of all layer indices required across all specs (sorted)."""
        layers: set[int] = set()
        for spec in self.specs.values():
            if spec.layers is not None:
                layers.update(spec.layers)
        return sorted(layers)

    def needs_all_layers(self) -> bool:
        """True if any spec has layers=None (requires a full forward pass)."""
        return any(s.layers is None for s in self.specs.values())

    def max_required_layer(self, total_layers: int) -> int:
        """Index of the deepest layer needed. Controls tracer.stop() placement."""
        if self.needs_all_layers():
            return total_layers - 1
        required = self.all_required_layers()
        return max(required) if required else total_layers - 1

    def add_spec(self, name: str, spec: ExtractionSpec) -> ExtractionPlan:
        """Fluent API for adding specs."""
        self.specs[name] = spec
        return self

    def __repr__(self) -> str:
        spec_names = list(self.specs.keys())
        return (
            f"ExtractionPlan(name={self.name!r}, "
            f"specs={spec_names}, "
            f"label_filter={self.label_filter!r})"
        )
