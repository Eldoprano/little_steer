"""Span-level activation extraction and vector evaluation helpers.

Centralises three previously-duplicated patterns (probing.py 211-324, 424-707,
715-918): formatting + tokenising an entry, slicing layer activations to a
span, and computing cosine similarities against a stack of vectors.

The headline win is `cosine_against_stack`: instead of looping over vectors
and computing one dot product at a time, we stack vectors into a (V, D)
matrix and let BLAS do (S, D) @ (D, V) → (S, V) once per layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import torch
import torch.nn.functional as F

from ..extraction.config import TokenSelection
from ..extraction.pooling import apply_aggregation

if TYPE_CHECKING:
    from ..data.tokenizer_utils import TokenSpan
    from ..models.model import LittleSteerModel
    from ..vectors.steering_vector import SteeringVector
    from thesis_schema import ConversationEntry


@dataclass
class EntryContext:
    """Everything we need about a conversation to do span-level evaluation.

    Built once per (entry, model) pair and reused across many vector
    operations to avoid redundant tokenisation / annotation mapping.
    """

    entry: "ConversationEntry"
    formatted_text: str
    message_offsets: dict[int, int]
    token_ids: torch.Tensor               # (seq_len,) on CPU
    token_spans: list                     # list[TokenSpan]
    offset_mapping: list[tuple[int, int]] | None  # (char_start, char_end) per token
    seq_len: int


@dataclass
class SpanActivation:
    """One aggregated activation per layer for a single span."""

    labels: set[str]
    by_layer: dict[int, torch.Tensor]    # layer_idx → (hidden_dim,)


def prepare_entry(
    model: "LittleSteerModel",
    entry: "ConversationEntry",
    *,
    max_seq_len: int = 4096,
    need_offset_mapping: bool = False,
) -> EntryContext | None:
    """Format → tokenise → map annotations to token spans.

    Returns None when there is nothing to evaluate (no annotations or no
    annotations survive the mapping step).
    """
    if not entry.annotations:
        return None

    formatted_text, message_offsets = model.format_messages_with_offsets(entry.messages)
    token_spans = model._token_mapper.map_annotations_to_tokens(
        entry=entry,
        formatted_text=formatted_text,
        message_offsets=message_offsets,
    )
    if not token_spans:
        return None

    encoding = model.tokenize(formatted_text, return_offsets_mapping=need_offset_mapping)
    token_ids = encoding["input_ids"][0]
    seq_len = min(len(token_ids), max_seq_len)
    token_ids = token_ids[:seq_len]

    offset_mapping = None
    if need_offset_mapping:
        offset_mapping = [
            (int(s), int(e)) for s, e in encoding["offset_mapping"][0].tolist()[:seq_len]
        ]

    return EntryContext(
        entry=entry,
        formatted_text=formatted_text,
        message_offsets=message_offsets,
        token_ids=token_ids,
        token_spans=token_spans,
        offset_mapping=offset_mapping,
        seq_len=seq_len,
    )


def select_span_activation(
    layer_act: torch.Tensor,
    token_span: "TokenSpan",
    token_selection: TokenSelection,
    *,
    seq_len: int,
    span_logits: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Apply a TokenSelection to a single span and return the aggregated activation.

    Returns None when the resolved window is empty.

    Returned tensor is always (hidden_dim,) — for the "none" aggregation we
    fall back to the last selected token to keep the output shape uniform
    across callers.
    """
    if token_span.token_start >= seq_len:
        return None

    eff_end = min(token_span.token_end, seq_len)
    resolved = token_selection.apply(token_span.token_start, eff_end, seq_len)
    if resolved is None:
        return None

    sel_start, sel_end = resolved
    selected = layer_act[sel_start:sel_end]
    if selected.shape[0] == 0:
        return None

    sel_logits = span_logits[sel_start:sel_end] if span_logits is not None else None
    raw = apply_aggregation(
        selected,
        token_selection.aggregation,
        token_logits=sel_logits,
        confidence_threshold=token_selection.confidence_threshold,
        high_entropy_fraction=token_selection.high_entropy_fraction,
    )
    # "none" aggregation returns (n_tokens, hidden_dim); fall back to last token.
    return raw.float() if raw.dim() == 1 else raw[-1].float()


def stack_vectors(
    vectors: Iterable["SteeringVector"],
    *,
    normalize: bool = True,
) -> tuple[dict[int, torch.Tensor], list[tuple[str, str, int]]]:
    """Group vectors by layer and stack each group into an (n_at_layer, D) tensor.

    Args:
        vectors:   Iterable of SteeringVector.
        normalize: If True, L2-normalise each vector. Required for cosine sim.

    Returns:
        (matrices, keys) where:
          matrices: layer_idx → (n_at_layer, hidden_dim) Tensor.
          keys:     Flat list of (label, method, layer) parallel to the rows
                    of matrices when iterated by layer in ascending order.

    Use with `cosine_against_stack`:
        sims = acts @ matrices[layer].T   # (n_spans, n_vecs_at_layer)
    """
    by_layer: dict[int, list[tuple[tuple[str, str, int], torch.Tensor]]] = {}
    for v in vectors:
        sv = v.vector.float()
        if normalize:
            sv = F.normalize(sv.unsqueeze(0), dim=-1).squeeze(0)
        by_layer.setdefault(v.layer, []).append(((v.label, v.method, v.layer), sv))

    matrices: dict[int, torch.Tensor] = {}
    keys: list[tuple[str, str, int]] = []
    for layer in sorted(by_layer):
        layer_keys, layer_vecs = zip(*by_layer[layer])
        matrices[layer] = torch.stack(list(layer_vecs))   # (n_at_layer, D)
        keys.extend(layer_keys)
    return matrices, keys


def cosine_against_stack(
    activations: torch.Tensor,
    vec_matrix: torch.Tensor,
    *,
    normalize_acts: bool = True,
) -> torch.Tensor:
    """Cosine similarity between many activations and many vectors in one matmul.

    Args:
        activations:    (..., D) tensor (e.g. (n_spans, D) or (seq_len, D)).
        vec_matrix:     (V, D) tensor of (already L2-normalised) vectors.
        normalize_acts: L2-normalise the activations before the matmul.

    Returns:
        Tensor of shape (..., V).
    """
    acts = activations.float()
    if normalize_acts:
        acts = F.normalize(acts, dim=-1)
    return acts @ vec_matrix.to(acts.device).T
