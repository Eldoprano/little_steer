"""Post-processing transforms for steering vectors.

Three operations that come up over and over in the literature:

* :func:`project_out` — remove a basis (e.g. neutral-text PCs) from a
  vector. Backbone of the confound-removal idea from Anthropic's emotion
  vector paper.
* :func:`normalize_to_residual_scale` — rescale a vector so that the
  injection ``alpha * vector`` lands at a sensible magnitude relative to
  the model's typical residual-stream norm at that layer (Lu et al. 2026
  "Persona Selection" recipe).
* :func:`compose` — weighted sum of multiple vectors, kept on the same
  layer. Useful for multi-attribute interventions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .steering_vector import SteeringVector


def project_out(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the components of ``vector`` that lie inside ``basis``.

    Args:
        vector: 1-D tensor of shape (D,).
        basis:  2-D tensor of shape (k, D). Rows are basis directions —
                they need NOT be orthonormal; we Gram-Schmidt internally.

    Returns:
        1-D tensor of shape (D,), the component of ``vector`` orthogonal to
        ``basis``.
    """
    if vector.dim() != 1:
        raise ValueError(f"vector must be 1-D, got shape {tuple(vector.shape)}")
    if basis.dim() != 2 or basis.shape[1] != vector.shape[0]:
        raise ValueError(
            f"basis shape {tuple(basis.shape)} incompatible with vector dim "
            f"{vector.shape[0]}"
        )

    v = vector.float().clone()
    Q, _ = torch.linalg.qr(basis.float().T)   # Q: (D, k_eff), orthonormal columns
    # Project onto Q, subtract.
    coefs = Q.T @ v                            # (k_eff,)
    return v - Q @ coefs


def normalize_to_residual_scale(
    vector: torch.Tensor,
    target_norm: float,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale ``vector`` so that its L2 norm equals ``target_norm``.

    Pair with :func:`~little_steer.persona.residual_norm` to compute
    ``target_norm`` on a representative dataset:

        norm_at_layer = ls.persona.residual_norm(model, calibration_set, layer=20)
        scaled = normalize_to_residual_scale(vec.vector, target_norm=norm_at_layer)

    With this rescaling, alpha values across layers and across vectors are
    directly comparable because they're expressed in units of "fraction of a
    typical residual-stream vector".
    """
    n = vector.float().norm()
    if n < eps:
        return vector.float()
    return vector.float() * (target_norm / n)


def compose(
    vectors: Iterable["SteeringVector | torch.Tensor"],
    weights: Iterable[float],
) -> torch.Tensor:
    """Weighted sum of vectors. Caller is responsible for layer compatibility.

    All inputs must share hidden dim. Returns the raw 1-D tensor (no metadata).
    """
    from .steering_vector import SteeringVector as _SV

    vec_list = []
    for v in vectors:
        if isinstance(v, _SV):
            vec_list.append(v.vector.float())
        else:
            vec_list.append(v.float())
    weight_list = list(weights)
    if len(vec_list) != len(weight_list):
        raise ValueError(
            f"vectors ({len(vec_list)}) and weights ({len(weight_list)}) "
            "must have the same length"
        )
    if not vec_list:
        raise ValueError("compose() needs at least one vector")
    stacked = torch.stack(vec_list)                       # (n, D)
    w = torch.tensor(weight_list, dtype=stacked.dtype)    # (n,)
    return (w.unsqueeze(-1) * stacked).sum(dim=0)
