"""
little_steer.vectors.builder

SteeringVectorBuilder: Creates SteeringVectors from an ExtractionResult
for all combinations of (spec, method, layer).
"""

from __future__ import annotations

import warnings
from typing import Literal

from ..extraction.result import ExtractionResult
from .methods import MeanCentering, MeanDifference, PCADirection, LinearProbe
from .steering_vector import SteeringVector, SteeringVectorSet

AVAILABLE_METHODS = Literal["mean_difference", "mean_centering", "pca", "linear_probe"]


class SteeringVectorBuilder:
    """Build steering vectors from an ExtractionResult.

    For each (spec, method, layer) combination, produces a SteeringVector.
    Returns a SteeringVectorSet for flexible filtering and access.

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

    def build(
        self,
        extraction_result: ExtractionResult,
        target_label: str,
        methods: list[str] | None = None,
        baseline_label: str | None = None,
        pca_components: int = 1,
        probe_C: float = 1.0,
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
            # Pre-compute "others" for mean_centering once per spec
            # instead of recomputing for every layer
            _others_cache: dict[int, dict[str, list]] = {}
            if "mean_centering" in methods:
                for layer in extraction_result.layers():
                    others = {}
                    for lbl in available_labels:
                        if lbl != target_label:
                            acts = extraction_result.get(spec_name, lbl, layer)
                            if acts:
                                others[lbl] = acts
                    if others:
                        _others_cache[layer] = others

            for layer in extraction_result.layers():
                target_acts = extraction_result.get(spec_name, target_label, layer)

                if len(target_acts) == 0:
                    continue

                for method_name in methods:
                    vec = self._build_one(
                        spec_name=spec_name,
                        layer=layer,
                        method_name=method_name,
                        target_label=target_label,
                        target_acts=target_acts,
                        extraction_result=extraction_result,
                        baseline_label=baseline_label,
                        pca_components=pca_components,
                        probe_C=probe_C,
                        _others_cache=_others_cache,
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
        target_acts,
        extraction_result: ExtractionResult,
        baseline_label: str | None,
        pca_components: int,
        probe_C: float,
        _others_cache: dict[int, dict[str, list]] | None = None,
    ) -> SteeringVector | None:
        """Build a single steering vector. Returns None on failure."""
        try:
            metadata = {"n_target_samples": len(target_acts)}

            if method_name == "mean_centering":
                others = (_others_cache or {}).get(layer, {})
                if not others:
                    return None
                vector = MeanCentering.compute(target_acts, others)
                metadata["n_baseline_categories"] = len(others)

            elif method_name == "mean_difference":
                if baseline_label is None:
                    warnings.warn(
                        "mean_difference requires baseline_label — skipping. "
                        "Pass baseline_label=... to build()."
                    )
                    return None
                baseline_acts = extraction_result.get(spec_name, baseline_label, layer)
                if not baseline_acts:
                    return None
                vector = MeanDifference.compute(target_acts, baseline_acts)
                metadata["n_baseline_samples"] = len(baseline_acts)
                metadata["baseline_label"] = baseline_label

            elif method_name == "pca":
                components = PCADirection.compute(target_acts, n_components=pca_components)
                vector = components[0]  # First component
                metadata["pca_component"] = 0

            elif method_name == "linear_probe":
                if baseline_label is None:
                    warnings.warn(
                        "linear_probe requires baseline_label — skipping. "
                        "Pass baseline_label=... to build()."
                    )
                    return None
                baseline_acts = extraction_result.get(spec_name, baseline_label, layer)
                if not baseline_acts:
                    return None
                vector, accuracy = LinearProbe.compute_with_score(
                    target_acts, baseline_acts, C=probe_C
                )
                metadata["probe_accuracy"] = accuracy
                metadata["n_baseline_samples"] = len(baseline_acts)
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
