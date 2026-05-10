"""
little_steer.steering

Steered text generation: inject a steering vector into the residual stream
during autoregressive generation.

Main entry point: steered_generate()

How it works:
  At every forward pass during generation (one per new token), nnterp adds
  `alpha * steering_vector` to the output of the chosen transformer layer.
  Positive alpha pushes the model towards the vector direction; negative
  alpha pushes it away.

response_only=True:
  By default the steering hook fires on every forward pass — including the
  prefill of the prompt, where the model is "reading" the user's question.
  When ``response_only=True`` the hook only modifies activations at the
  newly-generated token positions (i.e. the assistant's own response). The
  prompt is left untouched. This is closer in spirit to the K-Steering
  paper's setup, and it makes alpha-vs-effect curves easier to interpret —
  you're measuring "how much does steering the response itself change the
  model's output", not "how much does steering the model's READING of the
  prompt change things".

Performance note:
  nnsight tracing adds per-token overhead compared to native HF generation.
  To keep generation fast, prefer do_sample=False (greedy) and keep
  max_new_tokens small while experimenting.  The overhead is unavoidable
  when using nnsight — it is the price of activation access during generation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .models.model import LittleSteerModel
    from .vectors.steering_vector import SteeringVector


def steered_generate(
    model: "LittleSteerModel",
    messages: list[dict[str, str]] | str,
    steering_vec: "SteeringVector | torch.Tensor | None" = None,
    layer: int | None = None,
    *,
    alpha: float = 0.0,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    add_generation_prompt: bool = True,
    response_only: bool = False,
    prompt_only: bool = False,
    n_steered_tokens: int | None = None,
    n_steered_sentences: int | None = None,
    return_steered_mask: bool = False,
) -> "str | tuple[str, list[bool]]":
    """Generate text with optional activation steering.

    Adds ``alpha * steering_vec`` to the residual stream at ``layer`` on every
    forward pass during generation.

    Args:
        model:           LittleSteerModel instance.
        messages:        Chat messages (list of {"role":…, "content":…} dicts)
                         OR a pre-formatted string.  If a list, the chat template
                         is applied automatically.
        steering_vec:    The steering vector.  Can be:
                         - A SteeringVector object (layer inferred from it).
                         - A raw torch.Tensor of shape (hidden_dim,).
                         - None — runs a plain baseline generation.
        layer:           Which transformer layer to inject at.  Required when
                         steering_vec is a raw Tensor; inferred from SteeringVector
                         automatically (but can be overridden here).
        alpha:           Injection strength.  0 or None steering_vec → no steering.
                         Positive → steer towards behaviour.
                         Negative → steer away from behaviour.
        max_new_tokens:  Generation token budget.
        do_sample:       If False (default), greedy decoding — fastest and most
                         reproducible.  Set True to use temperature/top_p.
        temperature:     Sampling temperature (only used when do_sample=True).
        top_p:           Nucleus probability mass (only used when do_sample=True).
        add_generation_prompt:
                         Append the model's generation prompt after messages
                         (usually True for the standard single-turn use case).
        response_only:   If True, only inject on newly-generated assistant tokens;
                         leave the prompt prefill untouched. Implemented via a
                         forward hook that gates on ``hidden.shape[1] == 1``.
        prompt_only:     If True, only inject during prompt prefill. The altered
                         activations are cached in the KV cache, so future token
                         generations are influenced via attention — a subtler
                         effect than decoding-time steering.
        n_steered_tokens: If set, stop steering after this many generated tokens.
                         Subsequent tokens are generated without the vector.
        n_steered_sentences: If set, stop steering after this many complete
                         sentences (detected by `.`, `!`, `?`, `\\n` in decoded tokens).
        return_steered_mask: If True, returns ``(text, mask)`` where ``mask``
                         is a list of booleans (one per generated token) that is
                         True for tokens generated while the hook was active.
                         Only meaningful when n_steered_tokens or
                         n_steered_sentences are set (otherwise all are True or
                         all are False depending on phase).

    Returns:
        The generated text as a plain string when return_steered_mask=False.
        A (text, mask) tuple when return_steered_mask=True.

    Example:
        sv = vectors.filter(label="II_STATE_ETHICAL_MORAL_CONCERN",
                            method="mean_difference").vectors[0]

        # Baseline — no steering
        baseline = ls.steered_generate(model, messages)

        # Steer towards the behaviour
        towards = ls.steered_generate(model, messages, sv, alpha=30.0)

        # Steer away from the behaviour
        away = ls.steered_generate(model, messages, sv, alpha=-30.0)

        # Multi-layer steering: apply the same vector at two layers
        towards = ls.steered_generate(model, messages, sv.vector, layer=[14, 16], alpha=20.0)
    """
    # -----------------------------------------------------------------------
    # Resolve steering vector and layer(s)
    # -----------------------------------------------------------------------
    raw_vec: torch.Tensor | None = None
    steer_layer: int | list[int] | None = None

    if steering_vec is not None and alpha != 0.0:
        # Import here to avoid a circular dependency at module level
        from .vectors.steering_vector import SteeringVector as _SV

        if isinstance(steering_vec, _SV):
            raw_vec = steering_vec.vector
            steer_layer = layer if layer is not None else steering_vec.layer
        elif isinstance(steering_vec, torch.Tensor):
            if layer is None:
                raise ValueError(
                    "When steering_vec is a raw Tensor you must supply `layer`."
                )
            raw_vec = steering_vec
            steer_layer = layer
        else:
            raise TypeError(
                f"steering_vec must be a SteeringVector or torch.Tensor, "
                f"got {type(steering_vec).__name__!r}."
            )

    # -----------------------------------------------------------------------
    # Format input
    # -----------------------------------------------------------------------
    if isinstance(messages, str):
        formatted = messages
    else:
        formatted = model.format_messages(
            messages, add_generation_prompt=add_generation_prompt
        )

    input_ids = model.tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"]

    # -----------------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------------
    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    st = model.st  # nnterp StandardizedTransformer (nnsight LanguageModel)

    _needs_hook = raw_vec is not None and steer_layer is not None and (
        response_only or prompt_only
        or n_steered_tokens is not None
        or n_steered_sentences is not None
    )

    steered_mask: list[bool] = []

    if _needs_hook:
        layers_list = [steer_layer] if isinstance(steer_layer, int) else list(steer_layer)
        steering_hook, lm_head_hook = _build_phase_hooks(
            model, raw_vec, alpha,
            response_only=response_only,
            prompt_only=prompt_only,
            n_steered_tokens=n_steered_tokens,
            n_steered_sentences=n_steered_sentences,
            steered_mask=steered_mask,
        )
        handles: list = []
        try:
            for li in layers_list:
                handles.append(model.layers[li].register_forward_hook(steering_hook))
            if lm_head_hook is not None:
                handles.append(model.st.lm_head.register_forward_hook(lm_head_hook))
            with st.generate(input_ids, **gen_kwargs) as _gen:
                output_ids = st.generator.output.save()
        finally:
            for h in handles:
                h.remove()
    else:
        with st.generate(input_ids, **gen_kwargs) as _gen:
            if raw_vec is not None and steer_layer is not None:
                st.steer(steer_layer, raw_vec, factor=alpha)
            output_ids = st.generator.output.save()

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[1]:]
    text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    if return_steered_mask:
        n_new = len(new_tokens)
        # Pad or trim mask to match actual number of new tokens
        mask = steered_mask[:n_new]
        while len(mask) < n_new:
            mask.append(False)
        return text, mask
    return text


def _build_phase_hooks(
    model: "LittleSteerModel",
    vector: torch.Tensor,
    alpha: float,
    *,
    response_only: bool = False,
    prompt_only: bool = False,
    n_steered_tokens: int | None = None,
    n_steered_sentences: int | None = None,
    steered_mask: "list[bool] | None" = None,
):
    """Build steering hook(s) for various phase modes.

    Returns:
        (steering_hook, lm_head_hook) — lm_head_hook is None when not needed.

    Phase semantics
    ---------------
    response_only:      Fire only on single-token decode steps (shape[1] == 1).
    prompt_only:        Fire only on multi-token prefill steps (shape[1] > 1).
    n_steered_tokens:   Fire on decode steps; count inside steering_hook, stop at N.
    n_steered_sentences: Fire until N sentence-ending tokens have been generated
                         (tracked via lm_head hook on logit argmax).
    steered_mask:       Optional list to append per-decode-step booleans indicating
                        whether the vector was active for that step.
    """
    cached_vec = vector.detach().clone()

    # Shared mutable state (list cells as mutable closures)
    active = [True]
    decode_count = [0]      # counts decode steps where hook fired (for n_tokens)
    sentence_count = [0]    # counts sentence-ending tokens (for n_sentences)

    def steering_hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        is_decode = hidden.shape[1] == 1

        # Gate on phase
        if prompt_only:
            if is_decode:
                if steered_mask is not None:
                    steered_mask.append(False)
                return output
        else:
            # response_only / n_tokens / n_sentences — skip prefill
            if not is_decode:
                return output

        if steered_mask is not None and is_decode:
            steered_mask.append(active[0])

        if not active[0]:
            return output

        v = cached_vec.to(hidden.device).to(hidden.dtype)
        new_hidden = hidden + alpha * v

        # n_steered_tokens: count decode steps where we actually applied the vector
        if is_decode and n_steered_tokens is not None:
            decode_count[0] += 1
            if decode_count[0] >= n_steered_tokens:
                active[0] = False

        if rest is not None:
            return (new_hidden,) + rest
        return new_hidden

    lm_head_hook = None
    if n_steered_sentences is not None:
        tokenizer = model.tokenizer

        def _lm_head_hook(_module, _inputs, output):
            logits = output[0] if isinstance(output, tuple) else output
            # Only track during decode steps (seq_len == 1)
            # Handle both (batch, seq, vocab) and (batch, vocab) shapes
            if logits.ndim == 3 and logits.shape[1] != 1:
                return output
            if not active[0]:
                return output

            # Get the predicted next-token id from the last position
            flat_logits = logits.view(-1, logits.shape[-1])  # (batch*seq, vocab)
            next_id = int(flat_logits[-1].argmax())
            token_text = tokenizer.decode([next_id])
            if any(c in token_text for c in ".!?\n"):
                sentence_count[0] += 1
                if sentence_count[0] >= n_steered_sentences:
                    active[0] = False

            return output

        lm_head_hook = _lm_head_hook

    return steering_hook, lm_head_hook


def multi_steered_generate(
    model: "LittleSteerModel",
    messages: list[dict[str, str]] | str,
    steering_specs: list[tuple["SteeringVector | torch.Tensor", int | None, float]],
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    add_generation_prompt: bool = True,
    response_only: bool = False,
) -> str:
    """Generate text with multiple steering vectors applied simultaneously.

    Allows composing multiple behaviour directions — e.g. steer towards
    safety concern awareness while steering away from sycophancy.

    Uses forward hooks (not nnsight st.steer) so multiple vectors at the same
    layer accumulate additively — each hook appends its delta independently.

    Args:
        model:            LittleSteerModel instance.
        messages:         Chat messages or pre-formatted string.
        steering_specs:   List of (vector, layer, alpha) tuples. Each vector
                          is applied independently at its specified layer with
                          the given alpha. ``layer`` can be None if vector is
                          a SteeringVector (layer inferred from it).
        max_new_tokens:   Generation token budget.
        do_sample:        Sampling vs greedy.
        temperature:      Sampling temperature.
        top_p:            Nucleus sampling mass.
        add_generation_prompt: Append generation prompt after messages.
        response_only:    If True, only inject on newly-generated tokens
                          (skip prefill), same semantics as steered_generate.

    Returns:
        Generated text string.

    Example:
        output = ls.multi_steered_generate(model, messages, [
            (safety_vec, None, 15.0),     # amplify safety concern
            (sycophancy_vec, None, -10.0), # suppress sycophancy
        ])
    """
    from .vectors.steering_vector import SteeringVector as _SV

    # Resolve all steering specs
    resolved: list[tuple[torch.Tensor, int, float]] = []
    for vec, layer, alpha in steering_specs:
        if alpha == 0.0:
            continue
        if isinstance(vec, _SV):
            raw = vec.vector
            steer_layer = layer if layer is not None else vec.layer
        elif isinstance(vec, torch.Tensor):
            if layer is None:
                raise ValueError(
                    "When steering_vec is a raw Tensor you must supply layer."
                )
            raw = vec
            steer_layer = layer
        else:
            raise TypeError(
                f"steering_vec must be a SteeringVector or torch.Tensor, "
                f"got {type(vec).__name__!r}."
            )
        resolved.append((raw, steer_layer, alpha))

    if not resolved:
        return steered_generate(
            model, messages, max_new_tokens=max_new_tokens,
            do_sample=do_sample, temperature=temperature, top_p=top_p,
            add_generation_prompt=add_generation_prompt,
        )

    # Format input
    if isinstance(messages, str):
        formatted = messages
    else:
        formatted = model.format_messages(
            messages, add_generation_prompt=add_generation_prompt
        )

    input_ids = model.tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False,
    )["input_ids"]

    gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    st = model.st

    def _make_hook(cached_vec: torch.Tensor, a: float):
        """Return a forward hook that adds alpha*vec to the layer output."""
        def _hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None
            is_decode = hidden.shape[1] == 1
            if response_only and not is_decode:
                return output
            v = cached_vec.to(hidden.device).to(hidden.dtype)
            new_hidden = hidden + a * v
            if rest is not None:
                return (new_hidden,) + rest
            return new_hidden
        return _hook

    handles: list = []
    try:
        for raw_vec, steer_layer, alpha in resolved:
            cached = raw_vec.detach().clone()
            handles.append(
                model.layers[steer_layer].register_forward_hook(
                    _make_hook(cached, alpha)
                )
            )
        with st.generate(input_ids, **gen_kwargs) as _gen:
            output_ids = st.generator.output.save()
    finally:
        for h in handles:
            h.remove()

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True)
