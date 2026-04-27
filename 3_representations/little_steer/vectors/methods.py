"""
little_steer.vectors.methods

Steering vector creation methods.

All methods take activations as list[Tensor] where each Tensor is (hidden_dim,)
(or (n_tokens, hidden_dim) for "none" aggregation — methods handle the mean).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _stack(acts: list[torch.Tensor]) -> np.ndarray:
    """Stack a list of tensors into a numpy array.
    Handles both (hidden_dim,) and (n_tokens, hidden_dim) by averaging over tokens.
    Always casts to float32 since numpy/sklearn don't support bfloat16.
    """
    tensors = []
    for a in acts:
        if a.dim() == 2:
            tensors.append(a.mean(dim=0).float())
        else:
            tensors.append(a.float())
    return torch.stack(tensors).numpy().astype(np.float32)


class MeanDifference:
    """Steering vector = target mean − baseline mean.

    Classic contrastive pair. Requires a specific baseline category.

    Note on standardization: this method operates in raw activation space.
    High-variance dimensions will dominate the direction. For steering this is
    intentional — the scale of each dimension carries meaning. If you want a
    scale-invariant direction (e.g. for probing / cosine-similarity analysis),
    use LinearProbe instead, which normalizes internally and then projects the
    weight back to raw space.
    """

    @staticmethod
    def compute(
        target: list[torch.Tensor],
        baseline: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            target: Activation tensors for the target category.
            baseline: Activation tensors for the baseline category.

        Returns:
            (hidden_dim,) steering vector.
        """
        target_mean = torch.from_numpy(_stack(target).mean(axis=0))
        baseline_mean = torch.from_numpy(_stack(baseline).mean(axis=0))
        return (target_mean - baseline_mean).float()


class MeanCentering:
    """Steering vector = target mean − centroid of ALL other categories.

    More robust than MeanDifference because it doesn't depend on choosing
    one specific baseline — it uses all non-target categories.
    """

    @staticmethod
    def compute(
        target: list[torch.Tensor],
        others: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            target: Activation tensors for the target category.
            others: Dict {label: [tensors]} for all non-target categories.

        Returns:
            (hidden_dim,) steering vector.
        """
        target_mean = torch.from_numpy(_stack(target).mean(axis=0))
        other_means = []
        for acts in others.values():
            if acts:
                other_means.append(_stack(acts).mean(axis=0))
        if not other_means:
            raise ValueError("No other categories provided for MeanCentering")
        centroid = torch.from_numpy(np.stack(other_means).mean(axis=0))
        return (target_mean - centroid).float()


class PCADirection:
    """Top PCA component(s) of target activations.

    Captures the direction of maximum variance in the target category's
    activation subspace.

    Note on standardization: PCA is sensitive to feature scale. By default
    `center=True` removes the mean, but dimensions are NOT divided by std.
    This means high-variance dimensions dominate the first component, which
    is the conventional choice for RepE steering (the raw scale reflects
    the actual spread in activation space). If you want scale-invariant
    principal directions (e.g. for cross-layer or cross-model comparison),
    pre-standardize the tensors before passing them in.
    """

    @staticmethod
    def compute(
        target: list[torch.Tensor],
        n_components: int = 1,
        center: bool = True,
    ) -> list[torch.Tensor]:
        """
        Args:
            target: Activation tensors for the target category.
            n_components: Number of principal components to return.
            center: Whether to center data before PCA (recommended).

        Returns:
            List of n_components (hidden_dim,) tensors (principal components).
        """
        X = _stack(target)
        if center:
            X = X - X.mean(axis=0, keepdims=True)
        # Clamp n_components to min(n_samples, hidden_dim)
        n_components = min(n_components, X.shape[0], X.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X)
        return [torch.from_numpy(comp.copy()).float() for comp in pca.components_]


class LinearProbe:
    """Train a logistic regression probe; use its weight vector as steering direction.

    The decision boundary of a linear classifier separates target from baseline
    in activation space. The weight vector is the normal to this boundary —
    i.e., the direction most informative about the label. This makes it a
    principled steering direction, especially good for reading/detection.
    """

    @staticmethod
    def compute(
        target: list[torch.Tensor],
        baseline: list[torch.Tensor],
        C: float = 1.0,
        max_iter: int = 1000,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            target: Activation tensors for the target category (label=1).
            baseline: Activation tensors for the baseline category (label=0).
            C: Regularization strength (smaller = stronger regularization).
            max_iter: Max iterations for logistic regression solver.
            normalize: Whether to L2-normalize the output vector.

        Returns:
            (hidden_dim,) steering vector (the classifier weight vector).
        """
        X_target = _stack(target)
        X_baseline = _stack(baseline)

        X = np.vstack([X_target, X_baseline])
        y = np.array([1] * len(X_target) + [0] * len(X_baseline))

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use balanced class weights to handle imbalanced datasets automatically
        clf = LogisticRegression(
            C=C, max_iter=max_iter, solver="lbfgs", class_weight="balanced"
        )
        clf.fit(X_scaled, y)

        # The probe was trained on standardized features: score = w_scaled · (X - μ)/σ + b.
        # During steering we add the vector directly to raw activations, so we must
        # project back: w_raw = w_scaled / σ.  This preserves the decision boundary
        # direction in the original (unscaled) activation space.
        weight_raw = clf.coef_[0] / scaler.scale_
        weight = torch.from_numpy(weight_raw.astype(np.float32))

        if normalize:
            weight = weight / (weight.norm() + 1e-8)

        return weight

    @staticmethod
    def compute_with_score(
        target: list[torch.Tensor],
        baseline: list[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, float]:
        """Return (steering_vector, probe_accuracy).

        Useful for evaluating how separable the categories are at this layer.
        """
        from sklearn.model_selection import cross_val_score

        X_target = _stack(target)
        X_baseline = _stack(baseline)
        X = np.vstack([X_target, X_baseline])
        y = np.array([1] * len(X_target) + [0] * len(X_baseline))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        C = kwargs.get("C", 1.0)
        clf = LogisticRegression(
            C=C, max_iter=kwargs.get("max_iter", 1000),
            solver="lbfgs", class_weight="balanced",
        )

        n_folds = min(5, min(np.sum(y == 0), np.sum(y == 1)), len(y) // 2)
        n_folds = max(2, n_folds)
        cv_scores = cross_val_score(clf, X_scaled, y, cv=n_folds)
        accuracy = float(cv_scores.mean())

        vector = LinearProbe.compute(target, baseline, **kwargs)
        return vector, accuracy
