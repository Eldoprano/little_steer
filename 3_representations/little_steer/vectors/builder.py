"""
little_steer.vectors.builder

SteeringVectorBuilder: Creates SteeringVectors from an ExtractionResult
for all combinations of (spec, method, layer).
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import torch

from ..extraction.result import ExtractionResult
from .methods import _stack
from .steering_vector import SteeringVector, SteeringVectorSet

AVAILABLE_METHODS = Literal["mean_difference", "mean_centering", "pca", "linear_probe"]


class SteeringVectorBuilder:
    """Build steering vectors from an ExtractionResult.

    For each (spec, method, layer) combination, produces a SteeringVector.
    Returns a SteeringVectorSet for flexible filtering and access.

    Stacked numpy arrays are cached in the instance across build() calls so that
    when you call build() for each of N labels in a loop, each (spec, label, layer)
    is stacked only once total rather than once per label call.

    Example:
        builder = SteeringVectorBuilder()
        vectors = builder.build(
            result,
            target_label="I_REPHRASE_PROMPT",
            methods=["mean_centering", "pca", "linear_probe"],
            baseline_label="IV_INTEND_REFUSAL_OR_SAFE_ACTION",
        )
        print(vectors.summary())
    """

    def __init__(self):
        # Cache: (extraction_result_id, spec, label, layer) → stacked np.ndarray
        self._stacked_cache: dict[tuple, np.ndarray | None] = {}
        # Cache: (extraction_result_id, spec, layer) → {label: mean np.ndarray}
        self._means_cache: dict[tuple, dict[str, np.ndarray]] = {}

    def _get_stacked(
        self,
        extraction_result: ExtractionResult,
        spec: str,
        label: str,
        layer: int,
    ) -> np.ndarray | None:
        key = (id(extraction_result), spec, label, layer)
        if key not in self._stacked_cache:
            acts = extraction_result.get(spec, label, layer)
            self._stacked_cache[key] = _stack(acts) if acts else None
        return self._stacked_cache[key]

    def _get_means_for_layer(
        self,
        extraction_result: ExtractionResult,
        spec: str,
        layer: int,
    ) -> dict[str, np.ndarray]:
        """Return {label: mean_vector} for all labels at this (spec, layer), cached."""
        key = (id(extraction_result), spec, layer)
        if key not in self._means_cache:
            means: dict[str, np.ndarray] = {}
            for lbl in extraction_result.labels():
                acts = extraction_result.get(spec, lbl, layer)
                if acts:
                    X = _stack(acts)
                    if len(X) > 0:
                        means[lbl] = X.mean(axis=0)
                    # X is not stored in _stacked_cache — only the mean is kept.
                    # This avoids holding all-labels × all-layers stacked arrays
                    # simultaneously during the first build() call.
            self._means_cache[key] = means
        return self._means_cache[key]

    def clear_cache(self):
        """Clear stacked-array and means caches (call if extraction_result changes)."""
        self._stacked_cache.clear()
        self._means_cache.clear()

    def clear_stacked_cache(self):
        """Free large stacked arrays while keeping computed means.

        Call after each build() when looping over many labels. Means stay
        cached so mean_centering still works for subsequent labels without
        re-reading all activations.
        """
        self._stacked_cache.clear()

    def build(
        self,
        extraction_result: ExtractionResult,
        target_label: str,
        methods: list[str] | None = None,
        baseline_label: str | None = None,
        pca_components: int = 1,
        probe_C: float = 1.0,
        pca_contrastive: bool = False,
    ) -> SteeringVectorSet:
        """Build steering vectors for one target label across all specs and layers.

        Args:
            extraction_result: Output from ActivationExtractor.extract().
            target_label: The category label to create vectors for.
            methods: List of method names to use. Defaults to all four methods
                     (requires baseline_label for mean_difference and linear_probe).
            baseline_label: Comparison category for mean_difference and linear_probe.
                            If None, those two methods are skipped with a warning.
            pca_components: Number of PCA components (default 1 = first component).
            probe_C: Regularization for logistic regression probe.

        Returns:
            SteeringVectorSet containing all produced vectors.
        """
        if methods is None:
            methods = ["mean_centering", "pca", "mean_difference", "linear_probe"]

        # Validate that target and baseline labels exist in the extraction result
        available_labels = extraction_result.labels()
        if target_label not in available_labels:
            raise ValueError(
                f"target_label={target_label!r} not found in extraction result. "
                f"Available labels: {available_labels}"
            )
        if baseline_label is not None and baseline_label not in available_labels:
            raise ValueError(
                f"baseline_label={baseline_label!r} not found in extraction result. "
                f"Available labels: {available_labels}"
            )

        vector_set = SteeringVectorSet()

        for spec_name in extraction_result.specs():
            for layer in extraction_result.layers():
                # Use cache; skip if no target activations at this (spec, layer)
                target_stacked = self._get_stacked(extraction_result, spec_name, target_label, layer)
                if target_stacked is None or len(target_stacked) == 0:
                    continue

                # Fetch (or cache-hit) means for all labels at this (spec, layer)
                layer_means = self._get_means_for_layer(extraction_result, spec_name, layer)

                # Centroid-others for mean_centering via sum trick (O(n) arithmetic)
                mc_centroid_others: dict[str, np.ndarray] = {}
                if "mean_centering" in methods and layer_means:
                    labels_present = list(layer_means.keys())
                    sum_all = sum(layer_means[l] for l in labels_present)
                    n = len(labels_present)
                    mc_centroid_others = {
                        lbl: (sum_all - layer_means[lbl]) / max(n - 1, 1)
                        for lbl in labels_present
                    }

                for method_name in methods:
                    vec = self._build_one(
                        spec_name=spec_name,
                        layer=layer,
                        method_name=method_name,
                        target_label=target_label,
                        extraction_result=extraction_result,
                        baseline_label=baseline_label,
                        pca_components=pca_components,
                        probe_C=probe_C,
                        layer_means=layer_means,
                        mc_centroid_others=mc_centroid_others,
                        pca_contrastive=pca_contrastive,
                    )
                    if vec is not None:
                        vector_set.add(vec)

        print(
            f"✅ Built {len(vector_set)} vectors for label='{target_label}' "
            f"| specs={extraction_result.specs()} "
            f"| methods={methods} "
            f"| layers={extraction_result.layers()}"
        )
        return vector_set

    def build_all_labels(
        self,
        extraction_result: ExtractionResult,
        methods: list[str] | None = None,
        baseline_label: str | None = None,
        **kwargs,
    ) -> dict[str, SteeringVectorSet]:
        """Build steering vectors for ALL labels in the extraction result.

        Returns:
            {label: SteeringVectorSet} mapping.
        """
        result: dict[str, SteeringVectorSet] = {}
        for label in extraction_result.labels():
            result[label] = self.build(
                extraction_result,
                target_label=label,
                methods=methods,
                baseline_label=baseline_label,
                **kwargs,
            )
        return result

    def _build_one(
        self,
        spec_name: str,
        layer: int,
        method_name: str,
        target_label: str,
        extraction_result: ExtractionResult,
        baseline_label: str | None,
        pca_components: int,
        probe_C: float,
        layer_means: dict[str, np.ndarray],
        mc_centroid_others: dict[str, np.ndarray],
        pca_contrastive: bool = False,
    ) -> SteeringVector | None:
        """Build a single steering vector using cached stacked arrays."""
        try:
            target_stacked = self._get_stacked(extraction_result, spec_name, target_label, layer)
            if target_stacked is None or len(target_stacked) == 0:
                return None
            metadata = {"n_target_samples": len(target_stacked)}

            if method_name == "mean_centering":
                centroid = mc_centroid_others.get(target_label)
                target_mean = layer_means.get(target_label)
                if centroid is None or target_mean is None:
                    return None
                vector = torch.from_numpy((target_mean - centroid).astype(np.float32))
                metadata["n_baseline_categories"] = len(mc_centroid_others) - 1

            elif method_name == "mean_difference":
                if baseline_label is None:
                    warnings.warn(
                        "mean_difference requires baseline_label — skipping. "
                        "Pass baseline_label=... to build()."
                    )
                    return None
                baseline_mean = layer_means.get(baseline_label)
                target_mean = layer_means.get(target_label)
                if baseline_mean is None or target_mean is None:
                    return None
                vector = torch.from_numpy((target_mean - baseline_mean).astype(np.float32))
                baseline_stacked = self._get_stacked(extraction_result, spec_name, baseline_label, layer)
                metadata["n_baseline_samples"] = len(baseline_stacked) if baseline_stacked is not None else 0
                metadata["baseline_label"] = baseline_label

            elif method_name == "pca":
                X = target_stacked
                if len(X) == 0:
                    return None
                if pca_contrastive and layer_means:
                    other_means = [m for lbl, m in layer_means.items() if lbl != target_label]
                    if other_means:
                        others_centroid = np.mean(np.vstack(other_means), axis=0)
                        X = X - others_centroid
                from sklearn.decomposition import PCA as _PCA
                n_comp = min(pca_components, X.shape[0], X.shape[1])
                Xc = X - X.mean(axis=0, keepdims=True)
                _pca = _PCA(n_components=n_comp)
                _pca.fit(Xc)
                vector = torch.from_numpy(_pca.components_[0].copy().astype(np.float32))
                metadata["pca_component"] = 0
                metadata["pca_contrastive"] = pca_contrastive

            elif method_name == "linear_probe":
                if baseline_label is None:
                    warnings.warn(
                        "linear_probe requires baseline_label — skipping. "
                        "Pass baseline_label=... to build()."
                    )
                    return None
                baseline_stacked = self._get_stacked(extraction_result, spec_name, baseline_label, layer)
                if baseline_stacked is None or len(baseline_stacked) == 0:
                    return None
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                X = np.vstack([target_stacked, baseline_stacked])
                y = np.array([1] * len(target_stacked) + [0] * len(baseline_stacked))
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                clf = LogisticRegression(C=probe_C, max_iter=1000, solver="lbfgs", class_weight="balanced")
                clf.fit(X_scaled, y)
                accuracy = float(clf.score(X_scaled, y))
                weight_raw = clf.coef_[0] / scaler.scale_
                weight = torch.from_numpy(weight_raw.astype(np.float32))
                vector = weight / (weight.norm() + 1e-8)
                metadata["probe_accuracy"] = accuracy
                metadata["n_baseline_samples"] = len(baseline_stacked)
                metadata["baseline_label"] = baseline_label

            else:
                raise ValueError(f"Unknown method: {method_name!r}")

            return SteeringVector(
                vector=vector,
                layer=layer,
                label=target_label,
                method=method_name,
                extraction_spec=spec_name,
                metadata=metadata,
            )

        except Exception as e:
            warnings.warn(
                f"Failed to build {method_name} vector at "
                f"(spec={spec_name}, layer={layer}): {e}"
            )
            return None
