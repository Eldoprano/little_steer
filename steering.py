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
) -> str:
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

    Returns:
        The generated text as a plain string (only the newly generated tokens,
        special tokens stripped).

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

    with st.generate(input_ids, **gen_kwargs) as _gen:
        if raw_vec is not None and steer_layer is not None:
            # steer() handles device placement and accepts int or list[int]
            st.steer(steer_layer, raw_vec, factor=alpha)
        output_ids = st.generator.output.save()

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[1]:]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True)
