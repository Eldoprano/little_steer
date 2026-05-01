"""
little_steer.extraction.result

Container for extracted activations, organized by:
  spec_name → label → layer_idx → list[Tensor]
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch


@dataclass
class ExtractionMetadata:
    """Statistics and configuration for an extraction run."""

    plan_name: str
    n_conversations: int = 0
    n_annotations_processed: int = 0
    n_annotations_skipped: int = 0
    model_id: str = ""
    total_time_s: float = 0.0


class ExtractionResult:
    """Structured container for extracted activations.

    Organized as:
        spec_name → label → layer_idx → list of Tensors

    Each tensor has shape (hidden_dim,) if aggregation="mean",
    or (n_tokens, hidden_dim) if aggregation="none".

    Usage:
        # Access activations
        acts = result.get("last_token", "I_REPHRASE_PROMPT", layer=20)
        # acts is a list of (hidden_dim,) tensors

        # Iterate over all combinations
        for spec, label, layer, acts in result.iter_all():
            print(spec, label, layer, len(acts))

        # Rich summary
        print(result.summary())
    """

    def __init__(self, plan_name: str, metadata: ExtractionMetadata | None = None):
        self.plan_name = plan_name
        self.metadata = metadata or ExtractionMetadata(plan_name=plan_name)
        # Main data store: spec → label → layer → [Tensor, ...]
        self._data: dict[str, dict[str, dict[int, list[torch.Tensor]]]] = {}

    def add(
        self,
        spec: str,
        label: str,
        layer: int,
        activation: torch.Tensor,
    ) -> None:
        """Add one activation tensor for a (spec, label, layer) combination."""
        self._data.setdefault(spec, {}).setdefault(label, {}).setdefault(layer, []).append(
            activation.detach().cpu()
        )

    def get(self, spec: str, label: str, layer: int) -> list[torch.Tensor]:
        """Retrieve all activation tensors for a specific (spec, label, layer)."""
        return self._data.get(spec, {}).get(label, {}).get(layer, [])

    def specs(self) -> list[str]:
        """All extraction spec names present in this result."""
        return list(self._data.keys())

    def labels(self) -> list[str]:
        """All label categories present across all specs."""
        all_labels: set[str] = set()
        for spec_data in self._data.values():
            all_labels.update(spec_data.keys())
        return sorted(all_labels)

    def layers(self) -> list[int]:
        """All layer indices present across all specs."""
        all_layers: set[int] = set()
        for spec_data in self._data.values():
            for label_data in spec_data.values():
                all_layers.update(label_data.keys())
        return sorted(all_layers)

    def count(self, spec: str, label: str, layer: int) -> int:
        """Number of activation samples for a specific combination."""
        return len(self.get(spec, label, layer))

    def iter_all(
        self,
    ) -> Iterator[tuple[str, str, int, list[torch.Tensor]]]:
        """Iterate over all (spec, label, layer, activations) tuples."""
        for spec, spec_data in self._data.items():
            for label, label_data in spec_data.items():
                for layer, acts in label_data.items():
                    yield spec, label, layer, acts

    def summary(self) -> str:
        """Rich multi-line summary of what's in this result."""
        from rich.table import Table
        from rich.console import Console
        from io import StringIO

        lines = [f"ExtractionResult '{self.plan_name}'"]
        lines.append(
            f"  Conversations: {self.metadata.n_conversations}, "
            f"Annotations processed: {self.metadata.n_annotations_processed}, "
            f"Skipped: {self.metadata.n_annotations_skipped}"
        )

        for spec_name in self.specs():
            spec_data = self._data[spec_name]
            lines.append(f"\n  📊 Spec: '{spec_name}'")
            for label, label_data in sorted(spec_data.items()):
                counts = {layer: len(acts) for layer, acts in label_data.items()}
                if counts:
                    sample_count = list(counts.values())[0]
                    layer_str = str(sorted(counts.keys()))
                    lines.append(
                        f"     {label}: {sample_count} samples × {len(counts)} layers {layer_str}"
                    )
        return "\n".join(lines)

    def merge_from(self, other: "ExtractionResult") -> None:
        """Merge activations from another ExtractionResult into this one in-place."""
        for spec, label_data in other._data.items():
            for label, layer_data in label_data.items():
                for layer, acts in layer_data.items():
                    for act in acts:
                        self.add(spec, label, layer, act)
        self.metadata.n_conversations += other.metadata.n_conversations
        self.metadata.n_annotations_processed += other.metadata.n_annotations_processed
        self.metadata.n_annotations_skipped += other.metadata.n_annotations_skipped

    def __repr__(self) -> str:
        specs = self.specs()
        labels = self.labels()
        layers = self.layers()
        return (
            f"ExtractionResult(plan={self.plan_name!r}, "
            f"specs={specs}, labels={len(labels)}, layers={layers})"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to disk using torch.save (preserves tensors efficiently)."""
        path = Path(path)
        torch.save({
            "plan_name": self.plan_name,
            "metadata": self.metadata,
            "data": self._data,
        }, path)
        print(f"💾 ExtractionResult saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> ExtractionResult:
        """Load from disk."""
        path = Path(path)
        payload = torch.load(path, weights_only=False, map_location="cpu")
        result = cls(
            plan_name=payload["plan_name"],
            metadata=payload.get("metadata"),
        )
        result._data = payload["data"]
        print(f"📂 ExtractionResult loaded ← {path}")
        return result
