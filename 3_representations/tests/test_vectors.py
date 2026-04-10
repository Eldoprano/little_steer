"""
Tests for steering vector creation methods.
"""

import pytest
import torch
import numpy as np

from little_steer.vectors.methods import (
    MeanDifference,
    MeanCentering,
    PCADirection,
    LinearProbe,
)
from little_steer.vectors.steering_vector import SteeringVector, SteeringVectorSet


def random_acts(n: int, dim: int = 64) -> list[torch.Tensor]:
    """Generate n random (dim,) activation tensors."""
    return [torch.randn(dim) for _ in range(n)]


def separable_acts(n: int, dim: int = 64, offset: float = 5.0):
    """Generate two linearly separable activation sets."""
    target = [torch.randn(dim) + offset for _ in range(n)]
    baseline = [torch.randn(dim) - offset for _ in range(n)]
    return target, baseline


# ── MeanDifference ────────────────────────────────────────────────────────────

def test_mean_difference_shape():
    target, baseline = separable_acts(20)
    vec = MeanDifference.compute(target, baseline)
    assert vec.shape == (64,)


def test_mean_difference_direction():
    """With clearly separated clusters, vector should point from baseline to target."""
    dim = 64
    target = [torch.ones(dim) * 10 for _ in range(10)]
    baseline = [torch.ones(dim) * (-10) for _ in range(10)]
    vec = MeanDifference.compute(target, baseline)
    # Should be approximately 20 * ones(dim) / 1 (unnormalized)
    assert vec.mean().item() > 0


def test_mean_difference_reversed():
    target, baseline = separable_acts(20, offset=5.0)
    vec_fw = MeanDifference.compute(target, baseline)
    vec_bw = MeanDifference.compute(baseline, target)
    assert torch.allclose(vec_fw, -vec_bw, atol=1e-5)


# ── MeanCentering ─────────────────────────────────────────────────────────────

def test_mean_centering_shape():
    target = random_acts(20)
    others = {"cat_a": random_acts(15), "cat_b": random_acts(18)}
    vec = MeanCentering.compute(target, others)
    assert vec.shape == (64,)


def test_mean_centering_no_others_raises():
    target = random_acts(10)
    with pytest.raises(ValueError):
        MeanCentering.compute(target, {})


def test_mean_centering_single_other():
    target = random_acts(10)
    others = {"cat_a": random_acts(10)}
    # Should not raise
    vec = MeanCentering.compute(target, others)
    assert vec.shape == (64,)


# ── PCADirection ──────────────────────────────────────────────────────────────

def test_pca_direction_shape():
    target = random_acts(30)
    components = PCADirection.compute(target, n_components=1)
    assert len(components) == 1
    assert components[0].shape == (64,)


def test_pca_direction_multiple_components():
    target = random_acts(30)
    components = PCADirection.compute(target, n_components=3)
    assert len(components) == 3
    for c in components:
        assert c.shape == (64,)


def test_pca_direction_unit_norm():
    """PCA components from sklearn are unit-normalized."""
    target = random_acts(30)
    components = PCADirection.compute(target, n_components=2)
    for c in components:
        norm = c.norm().item()
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_pca_direction_few_samples():
    """With only 2 samples, still returns 1 component."""
    target = random_acts(2)
    components = PCADirection.compute(target, n_components=1)
    assert len(components) == 1


# ── LinearProbe ───────────────────────────────────────────────────────────────

def test_linear_probe_shape():
    target, baseline = separable_acts(30)
    vec = LinearProbe.compute(target, baseline)
    assert vec.shape == (64,)


def test_linear_probe_normalized():
    target, baseline = separable_acts(30)
    vec = LinearProbe.compute(target, baseline, normalize=True)
    norm = vec.norm().item()
    assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


def test_linear_probe_separable_accuracy():
    """Probe on clearly separable data should achieve near-perfect accuracy."""
    target, baseline = separable_acts(50, offset=10.0)
    vec, accuracy = LinearProbe.compute_with_score(target, baseline)
    assert accuracy > 0.9, f"Expected high accuracy on separable data, got {accuracy}"


def test_linear_probe_direction():
    """Probe vector should roughly point from baseline toward target."""
    dim = 64
    # Make a clear direction: target is in +direction, baseline in -direction
    direction = torch.zeros(dim)
    direction[0] = 1.0  # First dimension only

    target = [direction + torch.randn(dim) * 0.1 for _ in range(50)]
    baseline = [-direction + torch.randn(dim) * 0.1 for _ in range(50)]

    vec = LinearProbe.compute(target, baseline)
    # The first dimension should be positive (pointing toward target)
    assert vec[0].item() > 0


# ── SteeringVectorSet ─────────────────────────────────────────────────────────

def make_vector_set() -> SteeringVectorSet:
    vectors = []
    for method in ["mean_centering", "pca", "linear_probe"]:
        for spec in ["last_token", "whole_sentence"]:
            for layer in [15, 20, 25]:
                vectors.append(SteeringVector(
                    vector=torch.randn(64),
                    layer=layer,
                    label="I_REPHRASE_PROMPT",
                    method=method,
                    extraction_spec=spec,
                    metadata={"n_target_samples": 42},
                ))
    return SteeringVectorSet(vectors)


def test_vector_set_creation():
    vset = make_vector_set()
    assert len(vset) == 3 * 2 * 3  # 18 vectors


def test_vector_set_filter_method():
    vset = make_vector_set()
    pca_only = vset.filter(method="pca")
    assert all(v.method == "pca" for v in pca_only)
    assert len(pca_only) > 0


def test_vector_set_filter_layer():
    vset = make_vector_set()
    layer_20 = vset.filter(layer=20)
    assert all(v.layer == 20 for v in layer_20)


def test_vector_set_filter_combined():
    vset = make_vector_set()
    filtered = vset.filter(method="pca", spec="last_token", layer=20)
    assert len(filtered) == 1
    assert filtered.vectors[0].method == "pca"
    assert filtered.vectors[0].extraction_spec == "last_token"
    assert filtered.vectors[0].layer == 20


def test_vector_set_group_by():
    vset = make_vector_set()
    by_method = vset.group_by("method")
    assert set(by_method.keys()) == {"mean_centering", "pca", "linear_probe"}
    for method, subset in by_method.items():
        assert all(v.method == method for v in subset)


def test_vector_set_save_load(tmp_path):
    vset = make_vector_set()
    path = tmp_path / "test_vectors.pt"
    vset.save(str(path))
    loaded = SteeringVectorSet.load(str(path))
    assert len(loaded) == len(vset)
    # Check first vector approximately matches
    orig_vec = vset.vectors[0].vector
    load_vec = loaded.vectors[0].vector
    assert torch.allclose(orig_vec, load_vec)


def test_vector_normalized():
    v = SteeringVector(
        vector=torch.tensor([3.0, 4.0]),
        layer=0, label="L", method="pca", extraction_spec="s",
    )
    normed = v.normalized()
    assert abs(normed.vector.norm().item() - 1.0) < 1e-5


def test_vector_set_summary_contains_key_info():
    vset = make_vector_set()
    summary = vset.summary()
    assert "18" in summary or "vectors" in summary
    assert "pca" in summary
    assert "mean_centering" in summary
    assert "last_token" in summary
