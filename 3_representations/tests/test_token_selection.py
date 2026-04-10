"""
Tests for little_steer.extraction.config — TokenSelection behavior.

Covers:
  - All strategies
  - Positive bleed (extend window)
  - Negative bleed (shrink / skip tokens)
  - Edge cases: empty window after bleed, spans shorter than n, clamping
"""

import pytest
from little_steer.extraction.config import TokenSelection, ExtractionSpec, ExtractionPlan


# Helper: run TokenSelection.apply with default sequence_length=100
def apply(sel: TokenSelection, start: int, end: int, seq_len: int = 100):
    return sel.apply(span_token_start=start, span_token_end=end, sequence_length=seq_len)


# ── Strategy: first ───────────────────────────────────────────────────────────

def test_first_basic():
    sel = TokenSelection("first")
    assert apply(sel, 10, 20) == (10, 11)


def test_first_single_token_span():
    sel = TokenSelection("first")
    assert apply(sel, 5, 6) == (5, 6)


# ── Strategy: last ────────────────────────────────────────────────────────────

def test_last_basic():
    sel = TokenSelection("last")
    assert apply(sel, 10, 20) == (19, 20)


def test_last_single_token_span():
    sel = TokenSelection("last")
    assert apply(sel, 5, 6) == (5, 6)


# ── Strategy: all ─────────────────────────────────────────────────────────────

def test_all_basic():
    sel = TokenSelection("all")
    assert apply(sel, 5, 15) == (5, 15)


# ── Strategy: first_n ────────────────────────────────────────────────────────

def test_first_n_basic():
    sel = TokenSelection("first_n", n=3)
    assert apply(sel, 10, 20) == (10, 13)


def test_first_n_clamped_to_span():
    """If span has 5 tokens but n=10, take all 5."""
    sel = TokenSelection("first_n", n=10)
    assert apply(sel, 10, 15) == (10, 15)


# ── Strategy: last_n ─────────────────────────────────────────────────────────

def test_last_n_basic():
    sel = TokenSelection("last_n", n=3)
    assert apply(sel, 10, 20) == (17, 20)


def test_last_n_clamped_to_span():
    """If span has 5 tokens but n=10, take all 5."""
    sel = TokenSelection("last_n", n=10)
    assert apply(sel, 10, 15) == (10, 15)


# ── Positive bleed (extend window) ──────────────────────────────────────────

def test_bleed_before_positive_extends_left():
    """bleed_before=5 extends the window 5 tokens to the left."""
    sel = TokenSelection("all", bleed_before=5)
    # Window: [10-5, 20) = [5, 20)
    assert apply(sel, 10, 20) == (5, 20)


def test_bleed_after_positive_extends_right():
    """bleed_after=3 extends the window 3 tokens to the right."""
    sel = TokenSelection("all", bleed_after=3)
    # Window: [10, 20+3) = [10, 23)
    assert apply(sel, 10, 20) == (10, 23)


def test_bleed_both_positive():
    sel = TokenSelection("all", bleed_before=5, bleed_after=5)
    assert apply(sel, 10, 20) == (5, 25)


def test_bleed_clamped_at_sequence_start():
    """Bleed that goes beyond sequence start should clamp to 0."""
    sel = TokenSelection("all", bleed_before=20)
    # Window would be [10-20, 20) = [-10, 20) → clamped to (0, 20)
    assert apply(sel, 10, 20) == (0, 20)


def test_bleed_clamped_at_sequence_end():
    """Bleed that goes beyond sequence end should clamp."""
    sel = TokenSelection("all", bleed_after=200)
    assert apply(sel, 10, 20, seq_len=50) == (10, 50)


# ── Negative bleed (shrink / skip tokens) ────────────────────────────────────

def test_bleed_before_negative_skips_leading_tokens():
    """bleed_before=-3 skips the first 3 tokens of the span."""
    sel = TokenSelection("all", bleed_before=-3)
    # Window: [10-(-3), 20) = [13, 20)
    assert apply(sel, 10, 20) == (13, 20)


def test_bleed_after_negative_skips_trailing_tokens():
    """bleed_after=-2 skips the last 2 tokens of the span."""
    sel = TokenSelection("all", bleed_after=-2)
    # Window: [10, 20+(-2)) = [10, 18)
    assert apply(sel, 10, 20) == (10, 18)


def test_bleed_both_negative():
    """Skip 3 leading and 2 trailing."""
    sel = TokenSelection("all", bleed_before=-3, bleed_after=-2)
    # Window: [13, 18)
    assert apply(sel, 10, 20) == (13, 18)


def test_negative_bleed_makes_window_empty_returns_none():
    """If skipping more tokens than the span has, window becomes empty → None."""
    sel = TokenSelection("all", bleed_before=-7, bleed_after=-5)
    # Span is [10, 17) = 7 tokens
    # Window: [17, 12) → start >= end → None
    result = apply(sel, 10, 17)
    assert result is None


def test_negative_bleed_exactly_empties_window_returns_none():
    """Skip exactly all tokens."""
    sel = TokenSelection("all", bleed_before=-5)
    # Span [10, 15) has 5 tokens → after: [15, 15) → empty → None
    result = apply(sel, 10, 15)
    assert result is None


# ── Combined bleed + strategy ────────────────────────────────────────────────

def test_bleed_before_negative_then_first():
    """Skip 2 leading, then take first token of what remains."""
    sel = TokenSelection("first", bleed_before=-2)
    # Window: [12, 20), then first → [12, 13)
    assert apply(sel, 10, 20) == (12, 13)


def test_bleed_before_positive_then_last_n():
    """Extend 5 tokens left, then take last 3."""
    sel = TokenSelection("last_n", n=3, bleed_before=5)
    # Window: [5, 20), then last_3 → [17, 20)
    assert apply(sel, 10, 20) == (17, 20)


def test_bleed_extend_then_first_n():
    """Extend window, then first_n from the extended window."""
    sel = TokenSelection("first_n", n=4, bleed_before=3, bleed_after=3)
    # Window: [7, 23), then first_4 → [7, 11)
    assert apply(sel, 10, 20) == (7, 11)


# ── ExtractionPlan helpers ────────────────────────────────────────────────────

def test_extraction_plan_all_required_layers():
    plan = ExtractionPlan("test", specs={
        "s1": ExtractionSpec(TokenSelection("last"), layers=[10, 20]),
        "s2": ExtractionSpec(TokenSelection("all"), layers=[15, 25]),
    })
    assert plan.all_required_layers() == [10, 15, 20, 25]


def test_extraction_plan_needs_all_layers():
    plan = ExtractionPlan("test", specs={
        "s1": ExtractionSpec(TokenSelection("last"), layers=None),
    })
    assert plan.needs_all_layers() is True


def test_extraction_plan_does_not_need_all_layers():
    plan = ExtractionPlan("test", specs={
        "s1": ExtractionSpec(TokenSelection("last"), layers=[5, 10]),
    })
    assert plan.needs_all_layers() is False


def test_extraction_plan_max_required_layer():
    plan = ExtractionPlan("test", specs={
        "s1": ExtractionSpec(TokenSelection("last"), layers=[10, 25]),
    })
    assert plan.max_required_layer(total_layers=32) == 25


def test_extraction_plan_fluent_api():
    plan = ExtractionPlan("test")
    plan.add_spec("s1", ExtractionSpec(TokenSelection("first"), layers=[5]))
    plan.add_spec("s2", ExtractionSpec(TokenSelection("last"), layers=[10]))
    assert "s1" in plan.specs
    assert "s2" in plan.specs
