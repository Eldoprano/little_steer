"""
little_steer.vectors.steering_vector

SteeringVector and SteeringVectorSet data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch


@dataclass
class SteeringVector:
    """A single steering vector for a specific (label, method, spec, layer)."""

    vector: torch.Tensor
    """The steering direction, shape (hidden_dim,)."""

    layer: int
    """Layer index this vector was created from."""

    label: str
    """Target category label (e.g., 'I_REPHRASE_PROMPT')."""

    method: str
    """Creation method: 'mean_difference', 'mean_centering', 'pca', 'linear_probe'."""

    extraction_spec: str
    """Name of the ExtractionSpec used (e.g., 'last_token', 'whole_sentence')."""

    metadata: dict = field(default_factory=dict)
    """Extra info: n_samples, probe_accuracy, explained_variance_ratio, etc."""

    @property
    def hidden_dim(self) -> int:
        return self.vector.shape[0]

    def normalized(self) -> "SteeringVector":
        """Return a copy with the vector L2-normalized."""
        return SteeringVector(
            vector=self.vector / (self.vector.norm() + 1e-8),
            layer=self.layer,
            label=self.label,
            method=self.method,
            extraction_spec=self.extraction_spec,
            metadata=self.metadata,
        )

    def __repr__(self) -> str:
        return (
            f"SteeringVector("
            f"label={self.label!r}, "
            f"method={self.method!r}, "
            f"spec={self.extraction_spec!r}, "
            f"layer={self.layer}, "
            f"dim={self.hidden_dim})"
        )


class SteeringVectorSet:
    """A queryable collection of steering vectors.

    Supports filtering by any combination of method, layer, label, spec.
    Can be grouped, iterated, saved to disk, and loaded back.

    Example:
        # Filter to specific method and spec
        pca_last = vectors.filter(method="pca", spec="last_token")

        # Group by layer
        by_layer = vectors.group_by("layer")
        layer_20_vecs = by_layer[20]

        # Iterate
        for vec in vectors:
            print(vec)

        # Save/load
        vectors.save("steering_vectors.pt")
        loaded = SteeringVectorSet.load("steering_vectors.pt")
    """

    def __init__(self, vectors: list[SteeringVector] | None = None):
        self.vectors: list[SteeringVector] = vectors or []

    def add(self, vector: SteeringVector) -> None:
        self.vectors.append(vector)

    def filter(
        self,
        *,
        method: str | None = None,
        layer: int | None = None,
        label: str | None = None,
        spec: str | None = None,
    ) -> "SteeringVectorSet":
        """Return a new SteeringVectorSet matching ALL given criteria."""
        filtered = self.vectors
        if method is not None:
            filtered = [v for v in filtered if v.method == method]
        if layer is not None:
            filtered = [v for v in filtered if v.layer == layer]
        if label is not None:
            filtered = [v for v in filtered if v.label == label]
        if spec is not None:
            filtered = [v for v in filtered if v.extraction_spec == spec]
        return SteeringVectorSet(filtered)

    def group_by(self, key: str) -> dict:
        """Group vectors by any SteeringVector attribute.

        Args:
            key: Attribute name — 'method', 'layer', 'label', or 'extraction_spec'.

        Returns:
            {value: SteeringVectorSet} for each unique value of the attribute.
        """
        groups: dict = {}
        for v in self.vectors:
            k = getattr(v, key)
            if k not in groups:
                groups[k] = []
            groups[k].append(v)
        return {k: SteeringVectorSet(vs) for k, vs in groups.items()}

    def labels(self) -> list[str]:
        return sorted(set(v.label for v in self.vectors))

    def methods(self) -> list[str]:
        return sorted(set(v.method for v in self.vectors))

    def layers(self) -> list[int]:
        return sorted(set(v.layer for v in self.vectors))

    def specs(self) -> list[str]:
        return sorted(set(v.extraction_spec for v in self.vectors))

    def summary(self) -> str:
        """Human-readable summary of this vector set."""
        if not self.vectors:
            return "SteeringVectorSet (empty)"
        lines = [
            f"SteeringVectorSet: {len(self.vectors)} vectors",
            f"  labels:  {self.labels()}",
            f"  methods: {self.methods()}",
            f"  specs:   {self.specs()}",
            f"  layers:  {self.layers()}",
        ]
        if self.vectors:
            lines.append(f"  hidden_dim: {self.vectors[0].hidden_dim}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.vectors)

    def __iter__(self) -> Iterator[SteeringVector]:
        return iter(self.vectors)

    def __repr__(self) -> str:
        return (
            f"SteeringVectorSet("
            f"n={len(self.vectors)}, "
            f"labels={self.labels()}, "
            f"methods={self.methods()}, "
            f"layers={self.layers()})"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to disk."""
        path = Path(path)
        payload = [
            {
                "vector": v.vector,
                "layer": v.layer,
                "label": v.label,
                "method": v.method,
                "extraction_spec": v.extraction_spec,
                "metadata": v.metadata,
            }
            for v in self.vectors
        ]
        torch.save(payload, path)
        print(f"💾 SteeringVectorSet ({len(self.vectors)} vectors) → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SteeringVectorSet":
        """Load from disk."""
        path = Path(path)
        payload = torch.load(path, weights_only=False, map_location="cpu")
        vectors = [
            SteeringVector(
                vector=item["vector"],
                layer=item["layer"],
                label=item["label"],
                method=item["method"],
                extraction_spec=item["extraction_spec"],
                metadata=item.get("metadata", {}),
            )
            for item in payload
        ]
        loaded = cls(vectors)
        print(f"📂 SteeringVectorSet ({len(vectors)} vectors) ← {path}")
        return loaded
