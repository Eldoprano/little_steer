"""Tests for vectors.transforms (pure-math, no model needed)."""

import pytest
import torch

from little_steer.vectors.transforms import (
    compose,
    normalize_to_residual_scale,
    project_out,
)
from little_steer.vectors.steering_vector import SteeringVector


def test_project_out_removes_basis_components():
    # Build a vector, project out itself → result is ~zero.
    v = torch.tensor([3.0, 4.0, 0.0])
    basis = v.unsqueeze(0)
    out = project_out(v, basis)
    assert out.norm().item() < 1e-5


def test_project_out_preserves_orthogonal_components():
    # Vector along x; project out y. Result should equal x component.
    v = torch.tensor([3.0, 4.0, 0.0])
    basis = torch.tensor([[0.0, 1.0, 0.0]])
    out = project_out(v, basis)
    assert out[0].item() == pytest.approx(3.0, abs=1e-5)
    assert out[1].item() == pytest.approx(0.0, abs=1e-5)
    assert out[2].item() == pytest.approx(0.0, abs=1e-5)


def test_project_out_handles_non_orthogonal_basis():
    # Basis vectors that are not orthogonal — internal QR should orthonormalize.
    v = torch.tensor([1.0, 1.0, 1.0])
    basis = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],   # not orthogonal to row 0
    ])
    out = project_out(v, basis)
    # The plane spanned by basis is the xy-plane; out should be along z.
    assert out[0].item() == pytest.approx(0.0, abs=1e-5)
    assert out[1].item() == pytest.approx(0.0, abs=1e-5)
    assert out[2].item() == pytest.approx(1.0, abs=1e-5)


def test_project_out_validates_shapes():
    v = torch.tensor([1.0, 2.0])
    bad_basis = torch.tensor([1.0, 2.0, 3.0])  # 1-D
    with pytest.raises(ValueError):
        project_out(v, bad_basis)
    with pytest.raises(ValueError):
        project_out(torch.tensor([[1.0]]), torch.tensor([[1.0]]))  # vector 2-D


def test_normalize_to_residual_scale_changes_magnitude():
    v = torch.tensor([1.0, 0.0, 0.0])
    out = normalize_to_residual_scale(v, target_norm=5.0)
    assert out.norm().item() == pytest.approx(5.0, abs=1e-5)


def test_normalize_to_residual_scale_preserves_direction():
    v = torch.tensor([3.0, 4.0])
    out = normalize_to_residual_scale(v, target_norm=10.0)
    cos = (out / out.norm()) @ (v / v.norm())
    assert cos.item() == pytest.approx(1.0, abs=1e-5)


def test_normalize_to_residual_scale_handles_zero_vector():
    v = torch.zeros(4)
    out = normalize_to_residual_scale(v, target_norm=1.0)
    # No division by zero, returns the zero vector unchanged.
    assert out.norm().item() == pytest.approx(0.0)


def test_compose_weighted_sum():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    out = compose([a, b], [2.0, 3.0])
    assert out[0].item() == pytest.approx(2.0, abs=1e-5)
    assert out[1].item() == pytest.approx(3.0, abs=1e-5)


def test_compose_accepts_steering_vectors():
    sv = SteeringVector(
        vector=torch.tensor([1.0, 2.0]),
        layer=10, label="foo", method="pca", extraction_spec="last",
    )
    out = compose([sv], [0.5])
    assert out[0].item() == pytest.approx(0.5, abs=1e-5)
    assert out[1].item() == pytest.approx(1.0, abs=1e-5)


def test_compose_rejects_mismatched_lengths():
    a = torch.tensor([1.0])
    with pytest.raises(ValueError):
        compose([a, a], [1.0])


def test_compose_rejects_empty_input():
    with pytest.raises(ValueError):
        compose([], [])
