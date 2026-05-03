"""Single canonical forward-pass helper.

Used by `extraction.extractor`, `probing`, `scoring`, and the marimo notebook.
Centralises the nnsight tracing pattern, early-stop heuristic, and OOM/cleanup
so a fix or optimisation only happens in one place.

Reference implementation came from `extraction/extractor.py:245-276`.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import torch

if TYPE_CHECKING:
    from ..models.model import LittleSteerModel


@dataclass
class ForwardPassResult:
    """Activations and (optional) logits collected from a single forward pass.

    Attributes:
        layer_acts:  Dict {layer_idx: Tensor (seq_len, hidden_dim)} on CPU,
                     float dtype preserved (typically bfloat16 for nnterp models;
                     callers should `.float()` before sklearn/numpy ops).
        logits:      None unless ``need_logits=True``. Tensor (seq_len, vocab_size).
        seq_len:     Length of the input token sequence (post-truncation).
    """

    layer_acts: dict[int, torch.Tensor]
    logits: torch.Tensor | None
    seq_len: int


def run_forward_pass(
    model: "LittleSteerModel",
    token_ids: torch.Tensor,
    required_layers: Iterable[int],
    *,
    need_logits: bool = False,
    stop_after_max: bool = True,
) -> ForwardPassResult:
    """Run one nnsight-traced forward pass and return the requested activations.

    Args:
        model:           LittleSteerModel instance.
        token_ids:       1-D int tensor of shape (seq_len,). Caller is
                         responsible for truncation; this function passes the
                         tensor through verbatim.
        required_layers: Layer indices to capture. Order does not matter for
                         the caller, but layers are accessed in ascending
                         order internally (CRITICAL for nnsight).
        need_logits:     If True, also capture LM-head logits. Forces a full
                         forward pass (no `tracer.stop()`).
        stop_after_max:  If True (default) and ``need_logits`` is False, call
                         ``tracer.stop()`` after the highest required layer
                         to skip the remaining transformer blocks.

    Returns:
        ForwardPassResult with the captured activations on CPU.

    Notes:
        - Out-of-range layer indices (>= ``model.num_layers``) are silently
          skipped. Caller may inspect ``result.layer_acts`` to confirm which
          layers were actually captured.
        - Nothing is moved back to GPU here; the caller decides.
        - On exit, frees CUDA cache (best-effort) but does NOT delete
          ``result.layer_acts`` — the caller owns that data.
    """
    n_layers = model.num_layers
    sorted_layers = sorted(set(int(l) for l in required_layers))
    valid_layers = [l for l in sorted_layers if 0 <= l < n_layers]

    if not valid_layers and not need_logits:
        return ForwardPassResult(layer_acts={}, logits=None, seq_len=int(token_ids.shape[0]))

    max_required = valid_layers[-1] if valid_layers else -1
    do_stop = stop_after_max and (not need_logits) and (0 <= max_required < n_layers - 1)

    layer_acts: dict[int, torch.Tensor] = {}
    logits_saved = None

    with torch.no_grad():
        with model.trace(token_ids) as tracer:
            for layer_idx in valid_layers:
                # (1, seq_len, hidden_dim) → (seq_len, hidden_dim) on CPU
                layer_acts[layer_idx] = (
                    model.layers_output[layer_idx]
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .save()
                )

            if need_logits:
                logits_saved = (
                    model.st.output.logits
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .save()
                )
            elif do_stop:
                tracer.stop()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ForwardPassResult(
        layer_acts=layer_acts,
        logits=logits_saved,
        seq_len=int(token_ids.shape[0]),
    )
