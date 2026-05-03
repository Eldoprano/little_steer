"""Tests for the _internal.span_eval helpers.

Pure-math; no model loading required.
"""

import pytest
import torch

from little_steer._internal.span_eval import (
    cosine_against_stack,
    stack_vectors,
)
from little_steer.vectors.steering_vector import SteeringVector


def _mk_vec(layer: int, label: str, method: str, vec: torch.Tensor) -> SteeringVector:
    return SteeringVector(
        vector=vec, layer=layer, label=label, method=method,
        extraction_spec="last",
    )


def test_stack_vectors_groups_by_layer():
    v1 = _mk_vec(10, "a", "pca", torch.tensor([1.0, 0.0]))
    v2 = _mk_vec(10, "b", "pca", torch.tensor([0.0, 1.0]))
    v3 = _mk_vec(20, "a", "pca", torch.tensor([1.0, 1.0]))
    matrices, keys = stack_vectors([v1, v2, v3], normalize=True)
    assert set(matrices.keys()) == {10, 20}
    assert matrices[10].shape == (2, 2)
    assert matrices[20].shape == (1, 2)
    # Keys flat-list, in (layer asc) order.
    assert keys == [
        ("a", "pca", 10),
        ("b", "pca", 10),
        ("a", "pca", 20),
    ]


def test_stack_vectors_normalises_when_requested():
    v = _mk_vec(0, "a", "pca", torch.tensor([3.0, 4.0]))
    matrices, _ = stack_vectors([v], normalize=True)
    norm = matrices[0][0].norm().item()
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_stack_vectors_skips_normalisation_when_disabled():
    v = _mk_vec(0, "a", "pca", torch.tensor([3.0, 4.0]))
    matrices, _ = stack_vectors([v], normalize=False)
    assert matrices[0][0].norm().item() == pytest.approx(5.0, abs=1e-5)


def test_cosine_against_stack_recovers_identity_for_aligned_vectors():
    # Three already-normalised vectors plus three matching activations.
    vecs = torch.eye(4)        # (4, 4) — orthonormal basis
    acts = torch.eye(4)        # (4, 4)
    sims = cosine_against_stack(acts, vecs, normalize_acts=True)
    # sims is (4, 4). Diagonals == 1, off-diagonals == 0.
    diag = sims.diag()
    off = sims - torch.diag(diag)
    assert torch.allclose(diag, torch.ones(4), atol=1e-5)
    assert off.abs().max().item() == pytest.approx(0.0, abs=1e-5)


def test_cosine_against_stack_handles_non_normalised_acts():
    vecs = torch.tensor([[1.0, 0.0]])  # already unit
    acts = torch.tensor([[2.0, 0.0],   # parallel, magnitude 2
                         [0.0, 5.0]])  # orthogonal
    sims = cosine_against_stack(acts, vecs, normalize_acts=True)
    assert sims[0, 0].item() == pytest.approx(1.0, abs=1e-5)
    assert sims[1, 0].item() == pytest.approx(0.0, abs=1e-5)


def test_cosine_against_stack_supports_higher_dims():
    # (2, 3, 4) acts × (5, 4) vecs → (2, 3, 5)
    vecs = torch.randn(5, 4)
    acts = torch.randn(2, 3, 4)
    sims = cosine_against_stack(acts, vecs)
    assert sims.shape == (2, 3, 5)
