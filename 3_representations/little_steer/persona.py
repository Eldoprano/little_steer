"""Persona / Assistant-Axis utilities.

Implements helpers from Lu et al. 2026 ("Persona Selection Model"):

* :func:`extract_assistant_axis` — collect activations from role-induced
  prompts vs. a neutral baseline, mean-centre per role, then take the
  first PCA component as the "assistant axis".
* :func:`persona_drift` — project the assistant turns of a multi-turn
  conversation onto the axis to track how the model's persona shifts
  over the course of a dialogue.
* :func:`residual_norm` — average L2 norm of the residual stream at a
  given layer over a calibration dataset. Pair with
  :func:`~little_steer.vectors.transforms.normalize_to_residual_scale`
  to make alpha comparable across layers and models.

The paper's full recipe (275 roles × 5 system prompts × 240 questions,
fully-in-role filtering, etc.) is out of scope. This module gives you the
linear-algebra plumbing — bring your own role/prompt collection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from ._internal.forward import run_forward_pass
from .vectors.steering_vector import SteeringVector, SteeringVectorSet

if TYPE_CHECKING:
    from .models.model import LittleSteerModel
    from thesis_schema import ConversationEntry


def _last_token_activations(
    model: "LittleSteerModel",
    texts: Iterable[str],
    layers: list[int],
    *,
    max_seq_len: int = 4096,
    show_progress: bool = False,
) -> dict[int, list[torch.Tensor]]:
    """Run model on each text, return last-token activations per layer."""
    out: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    iterator = tqdm(list(texts), desc="Activations") if show_progress else texts
    for text in iterator:
        token_ids = model.tokenize(text)["input_ids"][0][:max_seq_len]
        if token_ids.numel() == 0:
            continue
        fwd = run_forward_pass(model, token_ids, layers, need_logits=False)
        for layer_idx in layers:
            la = fwd.layer_acts.get(layer_idx)
            if la is not None:
                out[layer_idx].append(la[-1].float())  # last token
    return out


def extract_assistant_axis(
    model: "LittleSteerModel",
    *,
    role_prompts: dict[str, list[str]],
    neutral_prompts: list[str],
    layers: list[int],
    max_seq_len: int = 4096,
    show_progress: bool = True,
) -> SteeringVectorSet:
    """Build the assistant axis (and per-role mean-centred vectors).

    Following Lu et al. 2026: for each role, take the mean of its activations
    at each layer, subtract the combined mean of the neutral activations,
    and stack the resulting per-role direction vectors. The first PCA
    component across roles is the "assistant axis" itself.

    Args:
        model:           LittleSteerModel.
        role_prompts:    {role_name: [prompts]} — typically system+question
                         pairs that elicit the role.
        neutral_prompts: List of neutral prompts ("You are an LLM"-style or
                         no-system-prompt). Their combined mean is subtracted
                         from every role mean.
        layers:          Which transformer layers to compute the axis on.
                         The paper used "middle layers"; for an N-layer model
                         try ``range(int(N*0.4), int(N*0.7))``.

    Returns:
        SteeringVectorSet with one vector per (role, layer) using
        ``method="role_centered"``, plus one extra vector per layer using
        ``label="ASSISTANT_AXIS"`` and ``method="pca"``.

    Notes:
        Vectors are NOT normalised — the magnitude reflects how far each
        role lives from the neutral centroid. Apply
        :func:`~little_steer.vectors.transforms.normalize_to_residual_scale`
        if you want comparable alphas across layers.
    """
    if not role_prompts:
        raise ValueError("role_prompts must contain at least one role")
    if not neutral_prompts:
        raise ValueError("neutral_prompts must be non-empty")

    sorted_layers = sorted(set(int(l) for l in layers))

    # 1. Neutral baseline: combined mean across all neutral prompts.
    neutral_acts = _last_token_activations(
        model, neutral_prompts, sorted_layers,
        max_seq_len=max_seq_len, show_progress=show_progress,
    )
    neutral_mean: dict[int, torch.Tensor] = {}
    for layer in sorted_layers:
        if neutral_acts[layer]:
            neutral_mean[layer] = torch.stack(neutral_acts[layer]).mean(dim=0)

    # 2. Per-role means; subtract neutral mean to get role-centred vectors.
    vectors: list[SteeringVector] = []
    role_centered: dict[int, dict[str, torch.Tensor]] = {l: {} for l in sorted_layers}

    iterator = (
        tqdm(role_prompts.items(), desc="Roles") if show_progress else role_prompts.items()
    )
    for role_name, prompts in iterator:
        role_acts = _last_token_activations(
            model, prompts, sorted_layers,
            max_seq_len=max_seq_len, show_progress=False,
        )
        for layer in sorted_layers:
            if not role_acts[layer]:
                continue
            role_mean = torch.stack(role_acts[layer]).mean(dim=0)
            centred = role_mean - neutral_mean.get(layer, torch.zeros_like(role_mean))
            role_centered[layer][role_name] = centred
            vectors.append(SteeringVector(
                vector=centred,
                layer=layer,
                label=role_name,
                method="role_centered",
                extraction_spec="last_token",
                metadata={"n_prompts": len(prompts)},
            ))

    # 3. First PCA component across roles → "assistant axis".
    for layer in sorted_layers:
        per_role = list(role_centered[layer].values())
        if len(per_role) < 2:
            continue
        X = torch.stack(per_role).numpy().astype(np.float32)
        pca = PCA(n_components=1)
        pca.fit(X)
        axis = torch.from_numpy(pca.components_[0].copy()).float()
        vectors.append(SteeringVector(
            vector=axis,
            layer=layer,
            label="ASSISTANT_AXIS",
            method="pca",
            extraction_spec="last_token",
            metadata={
                "n_roles": len(per_role),
                "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            },
        ))

    return SteeringVectorSet(vectors)


def persona_drift(
    model: "LittleSteerModel",
    conversation: list[dict[str, str]],
    axis_vector: SteeringVector | torch.Tensor,
    *,
    layer: int | None = None,
    normalize: bool = True,
    max_seq_len: int = 4096,
) -> list[float]:
    """Project each assistant turn onto the assistant axis.

    Args:
        conversation: Multi-turn ``[{"role": ..., "content": ...}, ...]``.
        axis_vector:  SteeringVector (layer auto-inferred) or a raw 1-D
                      tensor (then ``layer`` is required).
        layer:        Override / required when ``axis_vector`` is a tensor.
        normalize:    If True, projection is cosine similarity (in [-1, 1]);
                      if False, raw dot product.

    Returns:
        One float per assistant turn, in conversation order. Decreasing
        values across turns are evidence of persona drift (Lu et al.).

    Notes:
        This implementation rebuilds the formatted prefix at each turn and
        runs a fresh forward pass — it's O(turns) but simple. For long
        conversations consider extracting the activation directly from one
        single forward pass over the full transcript.
    """
    if isinstance(axis_vector, SteeringVector):
        vec = axis_vector.vector
        eff_layer = layer if layer is not None else axis_vector.layer
    else:
        if layer is None:
            raise ValueError("layer is required when axis_vector is a raw Tensor")
        vec = axis_vector
        eff_layer = int(layer)

    vec_norm = vec.float()
    if normalize:
        vec_norm = vec_norm / (vec_norm.norm() + 1e-8)

    drifts: list[float] = []
    for i, msg in enumerate(conversation):
        if msg["role"] != "assistant":
            continue
        # Build prefix up to (and including) this assistant turn.
        prefix = conversation[: i + 1]
        formatted = model.format_messages(prefix, add_generation_prompt=False)
        token_ids = model.tokenize(formatted)["input_ids"][0][:max_seq_len]
        if token_ids.numel() == 0:
            continue
        fwd = run_forward_pass(model, token_ids, [eff_layer], need_logits=False)
        last_act = fwd.layer_acts[eff_layer][-1].float()
        if normalize:
            last_act = last_act / (last_act.norm() + 1e-8)
        drifts.append(float((last_act * vec_norm).sum().item()))

    return drifts


def residual_norm(
    model: "LittleSteerModel",
    dataset: list["ConversationEntry"] | list[str],
    layer: int,
    *,
    max_seq_len: int = 4096,
    sample_size: int | None = None,
    show_progress: bool = True,
) -> float:
    """Average L2 norm of the residual stream at ``layer``.

    Used to put steering vectors on a comparable scale: the paper recommends
    rescaling vectors so ``||vector|| ≈ residual_norm(layer)`` and then using
    alpha values around ~1.0.

    Args:
        dataset: list of ConversationEntry (will be formatted via the chat
                 template) OR list of plain strings.
        layer:   Transformer layer to measure.
        sample_size: If given, randomly subsample the dataset to this size.
    """
    items = list(dataset)
    if sample_size is not None and sample_size < len(items):
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(len(items), size=sample_size, replace=False)
        items = [items[i] for i in idx]

    iterator = tqdm(items, desc=f"L{layer} residual norm") if show_progress else items
    norms: list[float] = []

    for item in iterator:
        if isinstance(item, str):
            text = item
        else:
            text = model.format_messages(item.messages)
        token_ids = model.tokenize(text)["input_ids"][0][:max_seq_len]
        if token_ids.numel() == 0:
            continue
        fwd = run_forward_pass(model, token_ids, [layer], need_logits=False)
        la = fwd.layer_acts.get(layer)
        if la is None:
            continue
        # Average per-token norm
        norms.append(la.float().norm(dim=-1).mean().item())

    return float(np.mean(norms)) if norms else float("nan")
