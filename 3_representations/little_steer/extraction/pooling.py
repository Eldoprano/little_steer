"""
little_steer.extraction.pooling

Probability-aware aggregation functions for activation extraction.

All pooling functions take the same inputs and return the same output shape,
making them drop-in alternatives to plain mean-pooling.

Inputs:
    activations  : (n_tokens, hidden_dim) — activations for the selected window
    token_logits : (n_tokens, vocab_size) — model logits at those positions

Output:
    (hidden_dim,) — a single aggregated activation vector

Semantics of token_logits[i]:
    The logits at position i predict what token comes AFTER position i.
    High top-1 probability → model is confident about the next token (low entropy).
    High entropy → model is uncertain about the next token → potential decision point.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to probabilities. (n_tokens, vocab_size) → same shape."""
    return F.softmax(logits.float(), dim=-1)


def _token_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) per token. (n_tokens, vocab_size) → (n_tokens,)."""
    log_probs = torch.log(probs.clamp(min=1e-10))
    return -(probs * log_probs).sum(dim=-1)


def top_confident_mean(
    activations: torch.Tensor,
    token_logits: torch.Tensor,
    threshold: float = 0.8,
) -> torch.Tensor:
    """Mean-pool only tokens where the top-1 next-token probability >= threshold.

    High top-1 confidence means the model strongly expects one outcome — these
    positions carry a cleaner, low-noise signal. Falls back to plain mean when
    no tokens pass the threshold.

    Args:
        activations:   (n_tokens, hidden_dim)
        token_logits:  (n_tokens, vocab_size) — model logits at each position
        threshold:     Minimum top-1 probability to include a token (default 0.8).

    Returns:
        (hidden_dim,) aggregated activation.
    """
    probs = _probs_from_logits(token_logits)
    top1_probs = probs.max(dim=-1).values  # (n_tokens,)
    mask = top1_probs >= threshold
    if not mask.any():
        return activations.mean(dim=0)
    return activations[mask].mean(dim=0)


def entropy_weighted_mean(
    activations: torch.Tensor,
    token_logits: torch.Tensor,
) -> torch.Tensor:
    """Weighted mean with weights proportional to 1 / token entropy.

    Confident positions (low entropy) get higher weight; uncertain positions
    (high entropy) get lower weight. Unlike top_confident_mean, all tokens
    participate — uncertain ones are down-weighted rather than excluded.

    Args:
        activations:   (n_tokens, hidden_dim)
        token_logits:  (n_tokens, vocab_size) — model logits at each position

    Returns:
        (hidden_dim,) aggregated activation.
    """
    probs = _probs_from_logits(token_logits)
    ent = _token_entropy(probs)         # (n_tokens,)
    inv_ent = 1.0 / (ent + 1e-10)      # avoid division by zero
    weights = inv_ent / inv_ent.sum()   # normalise to sum=1
    return (activations.float() * weights.unsqueeze(-1)).sum(dim=0)


def decision_point_mean(
    activations: torch.Tensor,
    token_logits: torch.Tensor,
    high_entropy_fraction: float = 0.5,
) -> torch.Tensor:
    """Mean-pool only the highest-entropy tokens (model 'decision points').

    High entropy = model was most uncertain here = the model is choosing
    between multiple continuations. Hypothesis: at these forks the activations
    may already encode the behaviour being decided, even if the sampled token
    doesn't reveal it in the text.

    Args:
        activations:            (n_tokens, hidden_dim)
        token_logits:           (n_tokens, vocab_size) — model logits at each position
        high_entropy_fraction:  Fraction of tokens to keep by highest entropy
                                (0.5 = top-50%, default).

    Returns:
        (hidden_dim,) aggregated activation.
    """
    n = activations.shape[0]
    if n <= 1:
        return activations.mean(dim=0)
    probs = _probs_from_logits(token_logits)
    ent = _token_entropy(probs)  # (n_tokens,)
    cutoff = torch.quantile(ent, 1.0 - high_entropy_fraction)
    mask = ent >= cutoff
    if not mask.any():
        return activations.mean(dim=0)
    return activations[mask].mean(dim=0)


def apply_aggregation(
    activations: torch.Tensor,
    aggregation: str,
    token_logits: torch.Tensor | None = None,
    confidence_threshold: float = 0.8,
    high_entropy_fraction: float = 0.5,
) -> torch.Tensor:
    """Dispatch aggregation by name.

    For "mean" and "none", token_logits is ignored.
    For probability-based aggregations, token_logits must be provided.

    Returns:
        (hidden_dim,) for all aggregations except "none", which returns the
        raw (n_tokens, hidden_dim) tensor unchanged.
    """
    if aggregation == "mean":
        return activations.mean(dim=0)

    elif aggregation == "top_confident_mean":
        if token_logits is None:
            raise ValueError("top_confident_mean requires token_logits")
        return top_confident_mean(activations, token_logits, confidence_threshold)

    elif aggregation == "entropy_weighted_mean":
        if token_logits is None:
            raise ValueError("entropy_weighted_mean requires token_logits")
        return entropy_weighted_mean(activations, token_logits)

    elif aggregation == "decision_point_mean":
        if token_logits is None:
            raise ValueError("decision_point_mean requires token_logits")
        return decision_point_mean(activations, token_logits, high_entropy_fraction)

    else:
        # "none" — keep raw tensor
        return activations
