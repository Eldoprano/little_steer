"""
little_steer.k_steering

Gradient-based activation steering using an MLPProbe or LinearProbeMultilabel
as the differentiable signal.  Implements Algorithms 1 and 2 from the
K-Steering paper (Oozeer et al. 2025, arXiv:2505.24535).

Key difference from steering.py (fixed steering vectors):
  Here the gradient direction is computed fresh from the *current* activations
  at each generation step, so it adapts to the evolving context rather than
  applying a single pre-computed direction.

Two entry points:

  k_steered_generate()         — Algorithm 1: iterative gradient steps.
  projection_removal_generate() — Algorithm 2: one-shot Householder reflection.

Both work by registering a PyTorch forward hook on the target transformer layer
and removing it after generation finishes.  The hook fires at every token
generation step, computes the gradient-based update through the probe, and
patches the activations before they propagate through the rest of the model.

Usage:
    probe = MLPProbeTrainer().train(result, spec="last_token", layer=20)

    # Steer toward stateSafetyConcern, away from neutralFiller
    output = ls.k_steered_generate(
        model, messages, probe,
        layer=20,
        increase_labels=["stateSafetyConcern"],
        decrease_labels=["neutralFiller"],
        alpha=1.0,
        K=3,
    )

    # One-shot reflection: suppress a label without iterating
    output = ls.projection_removal_generate(
        model, messages, probe,
        layer=20,
        decrease_labels=["intendHarmfulCompliance"],
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .models.model import LittleSteerModel
    from .mlp_probe import MLPProbe, LinearProbeMultilabel


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_label_indices(probe, labels: list[str] | None) -> list[int]:
    if not labels:
        return []
    missing = [l for l in labels if l not in probe.label_to_idx]
    if missing:
        raise ValueError(
            f"Labels not found in probe: {missing}. "
            f"Available: {list(probe.label_to_idx)}"
        )
    return [probe.label_to_idx[l] for l in labels]


def _compute_steer_loss(
    logits: torch.Tensor,
    increase_indices: list[int],
    decrease_indices: list[int],
) -> torch.Tensor:
    """Loss that drives activations toward increase_labels, away from decrease_labels.

    Applied over ALL sequence positions at once (mean over seq_len and labels),
    matching the paper's Algorithm 1 steering loss.

    Args:
        logits:           Shape (..., n_labels) — raw logits from the probe.
        increase_indices: Column indices of labels to amplify (loss -= their mean).
        decrease_indices: Column indices of labels to suppress (loss += their mean).
    """
    loss = torch.zeros(1, device=logits.device)
    if increase_indices:
        loss = loss - logits[..., increase_indices].mean()
    if decrease_indices:
        loss = loss + logits[..., decrease_indices].mean()
    return loss


def _k_gradient_steps(
    activations: torch.Tensor,
    probe: "MLPProbe | LinearProbeMultilabel",
    increase_indices: list[int],
    decrease_indices: list[int],
    alpha: float,
    K: int,
    gamma: float,
) -> torch.Tensor:
    """Algorithm 1: K gradient steps through the probe.

    Args:
        activations:      Shape (seq_len, hidden_dim), model dtype.
        probe:            Trained probe (on same device as activations).
        increase_indices: Label indices to amplify.
        decrease_indices: Label indices to suppress.
        alpha:            Base step size.
        K:                Number of gradient steps.
        gamma:            Per-step decay factor (effective step = alpha * gamma^k).

    Returns:
        Updated activations in the original dtype, detached.
    """
    a = activations.detach().float()
    probe_float = probe.float()

    for step in range(K):
        a = a.detach().requires_grad_(True)
        logits = probe_float(a)     # (seq_len, n_labels)
        loss = _compute_steer_loss(logits, increase_indices, decrease_indices)
        loss.backward()
        with torch.no_grad():
            a = a - alpha * (gamma ** step) * a.grad

    return a.detach().to(activations.dtype)


def _projection_removal(
    activations: torch.Tensor,
    probe: "MLPProbe | LinearProbeMultilabel",
    decrease_indices: list[int],
) -> torch.Tensor:
    """Algorithm 2: one-shot Householder reflection.

    Removes the component of the activations that points toward the unwanted
    behavior direction (as identified by the probe gradient).

    Reflection formula: a' = a - 2 * (a·g / ‖g‖²) * g
    where g = ∂loss/∂a,  loss = mean of logits for labels to suppress.

    Treats the full (seq_len, hidden_dim) activation as a single flat vector
    for the dot-product terms, so the reflection is applied globally.

    Args:
        activations:      Shape (seq_len, hidden_dim), model dtype.
        probe:            Trained probe (on same device as activations).
        decrease_indices: Label indices to suppress.

    Returns:
        Reflected activations in the original dtype, detached.
    """
    a = activations.detach().float().requires_grad_(True)
    probe_float = probe.float()

    logits = probe_float(a)
    loss = _compute_steer_loss(logits, [], decrease_indices)
    loss.backward()

    g = a.grad.detach()

    with torch.no_grad():
        a_flat = a.detach().reshape(-1)
        g_flat = g.reshape(-1)
        scale = (a_flat @ g_flat) / (g_flat @ g_flat + 1e-12)
        a_new = a.detach() - 2.0 * scale * g

    return a_new.to(activations.dtype)


def _get_primary_device(model: "LittleSteerModel") -> torch.device:
    """Get the device of the first model parameter."""
    return next(model.st.parameters()).device


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def k_steered_generate(
    model: "LittleSteerModel",
    messages: list[dict[str, str]] | str,
    probe: "MLPProbe | LinearProbeMultilabel",
    *,
    layer: int,
    increase_labels: list[str] | None = None,
    decrease_labels: list[str] | None = None,
    alpha: float = 1.0,
    K: int = 3,
    gamma: float = 0.9,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    add_generation_prompt: bool = True,
) -> str:
    """Generate text with gradient-based activation steering (Algorithm 1).

    At each generation step, hooks into the residual stream at ``layer``,
    runs K gradient steps through ``probe`` to nudge activations toward
    ``increase_labels`` and away from ``decrease_labels``, then continues
    generation with the patched activations.

    Unlike static steering vectors, the gradient direction is computed from
    the current activations at each token, so it adapts to the evolving context.

    Args:
        model:            LittleSteerModel instance.
        messages:         Chat messages or pre-formatted string.
        probe:            Trained MLPProbe or LinearProbeMultilabel.
        layer:            Transformer layer to intervene at.
        increase_labels:  Labels to amplify (probe columns to push up).
        decrease_labels:  Labels to suppress (probe columns to push down).
        alpha:            Gradient step size.
        K:                Number of gradient steps per token.
        gamma:            Per-step step-size decay (step k uses alpha * gamma^k).
        max_new_tokens:   Generation token budget.
        do_sample:        Sampling vs greedy decoding.
        temperature:      Sampling temperature (only when do_sample=True).
        top_p:            Nucleus sampling mass (only when do_sample=True).
        add_generation_prompt: Append generation prompt after messages.

    Returns:
        Generated text string (newly generated tokens only).

    Example:
        probe = MLPProbeTrainer().train(result, spec="last_token", layer=20)
        output = ls.k_steered_generate(
            model, messages, probe,
            layer=20,
            increase_labels=["stateSafetyConcern"],
            decrease_labels=["neutralFiller"],
            alpha=1.0, K=3,
        )
    """
    if not increase_labels and not decrease_labels:
        raise ValueError("Provide at least one of increase_labels or decrease_labels.")

    inc_idx = _resolve_label_indices(probe, increase_labels)
    dec_idx = _resolve_label_indices(probe, decrease_labels)

    dev = _get_primary_device(model)
    probe_on_device = probe.to(dev)

    def hook_fn(
        module: torch.nn.Module,
        input: tuple,
        output: tuple | torch.Tensor,
    ) -> tuple | torch.Tensor:
        is_tuple = isinstance(output, tuple)
        acts = output[0] if is_tuple else output   # (batch, seq_len, hidden_dim)

        with torch.enable_grad():
            updated = _k_gradient_steps(
                acts.squeeze(0),         # (seq_len, hidden_dim)
                probe_on_device,
                inc_idx,
                dec_idx,
                alpha,
                K,
                gamma,
            ).unsqueeze(0)              # (1, seq_len, hidden_dim)

        if is_tuple:
            return (updated,) + output[1:]
        return updated

    layer_module = model.layers[layer]
    handle = layer_module.register_forward_hook(hook_fn)

    formatted = (
        messages if isinstance(messages, str)
        else model.format_messages(messages, add_generation_prompt=add_generation_prompt)
    )
    input_ids = model.tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(dev)

    gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    try:
        with torch.no_grad():
            output_ids = model.st._model.generate(input_ids, **gen_kwargs)
    finally:
        handle.remove()
        probe.to("cpu")

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True)


def projection_removal_generate(
    model: "LittleSteerModel",
    messages: list[dict[str, str]] | str,
    probe: "MLPProbe | LinearProbeMultilabel",
    *,
    layer: int,
    decrease_labels: list[str],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.9,
    add_generation_prompt: bool = True,
) -> str:
    """Generate text with one-shot Householder reflection (Algorithm 2).

    At each generation step, reflects the activations at ``layer`` through
    the hyperplane perpendicular to the probe gradient for ``decrease_labels``.
    This removes the component of the activations that points toward the
    unwanted behavior direction — a single-step ablation of the behavior.

    Cheaper than k_steered_generate (one forward+backward per token instead
    of K).  Most useful when you only want to suppress a behavior.

    Args:
        model:            LittleSteerModel instance.
        messages:         Chat messages or pre-formatted string.
        probe:            Trained MLPProbe or LinearProbeMultilabel.
        layer:            Transformer layer to intervene at.
        decrease_labels:  Labels whose component to remove from activations.
        max_new_tokens:   Generation token budget.
        do_sample:        Sampling vs greedy decoding.
        temperature:      Sampling temperature (only when do_sample=True).
        top_p:            Nucleus sampling mass (only when do_sample=True).
        add_generation_prompt: Append generation prompt after messages.

    Returns:
        Generated text string (newly generated tokens only).

    Example:
        output = ls.projection_removal_generate(
            model, messages, probe,
            layer=20,
            decrease_labels=["intendHarmfulCompliance"],
        )
    """
    if not decrease_labels:
        raise ValueError("Provide at least one decrease_labels.")

    dec_idx = _resolve_label_indices(probe, decrease_labels)

    dev = _get_primary_device(model)
    probe_on_device = probe.to(dev)

    def hook_fn(
        module: torch.nn.Module,
        input: tuple,
        output: tuple | torch.Tensor,
    ) -> tuple | torch.Tensor:
        is_tuple = isinstance(output, tuple)
        acts = output[0] if is_tuple else output   # (batch, seq_len, hidden_dim)

        with torch.enable_grad():
            updated = _projection_removal(
                acts.squeeze(0),
                probe_on_device,
                dec_idx,
            ).unsqueeze(0)

        if is_tuple:
            return (updated,) + output[1:]
        return updated

    layer_module = model.layers[layer]
    handle = layer_module.register_forward_hook(hook_fn)

    formatted = (
        messages if isinstance(messages, str)
        else model.format_messages(messages, add_generation_prompt=add_generation_prompt)
    )
    input_ids = model.tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(dev)

    gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    try:
        with torch.no_grad():
            output_ids = model.st._model.generate(input_ids, **gen_kwargs)
    finally:
        handle.remove()
        probe.to("cpu")

    new_tokens = output_ids[0][input_ids.shape[1]:]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True)
