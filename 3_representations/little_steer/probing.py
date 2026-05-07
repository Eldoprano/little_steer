"""
little_steer.probing

Reading steering vector signal from model activations.

All functions here are thin wrappers around :class:`~little_steer.session.Session`,
which owns the model, caches forward passes, and does the actual work. If you
plan to call more than one of these on the same dataset, construct a Session
directly so that forward-pass caching kicks in:

    session = ls.Session(model, entries)
    scores  = session.score(vectors)
    results = session.evaluate(vectors, aggregations=["mean", "first"])

The two patterns produce identical results — the Session form just avoids
re-running the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .extraction.config import TokenSelection
from .data.tokenizer_utils import TokenSpan

if TYPE_CHECKING:
    from .models.model import LittleSteerModel
    from .vectors.steering_vector import SteeringVector, SteeringVectorSet
    from .mlp_probe import MLPProbe, LinearProbeMultilabel
    from thesis_schema import ConversationEntry


# ---------------------------------------------------------------------------
# Result types (re-exported by session.py to avoid circular imports)
# ---------------------------------------------------------------------------


@dataclass
class BehaviorScore:
    """Discrimination score for a (label, layer, method) triple.

    Attributes:
        label, layer, method: identifying tuple.
        mean_present:  Mean cosine similarity on annotated spans WITH this label.
        mean_absent:   Mean cosine similarity on annotated spans WITHOUT it.
        n_present, n_absent: span counts per group.

    Property ``discrimination = mean_present − mean_absent`` is the headline
    figure — higher = vector aligns better with the label.
    """

    label: str
    layer: int
    method: str
    mean_present: float
    mean_absent: float
    n_present: int
    n_absent: int

    @property
    def discrimination(self) -> float:
        return self.mean_present - self.mean_absent

    def __repr__(self) -> str:
        return (
            f"BehaviorScore("
            f"label={self.label!r}, layer={self.layer}, method={self.method!r}, "
            f"present={self.mean_present:.3f} (n={self.n_present}), "
            f"absent={self.mean_absent:.3f} (n={self.n_absent}), "
            f"disc={self.discrimination:+.3f})"
        )


@dataclass
class TokenSimilarities:
    """Per-token cosine similarities at one or more layers.

    Returned by :func:`get_token_similarities` and consumed by
    ``visualization.token_view``.
    """

    tokens: List[str]
    similarities: Dict[int, List[float]]
    token_char_spans: List[Tuple[int, int]]
    token_spans: List[TokenSpan]
    formatted_text: str
    label: str
    layer: int


@dataclass
class ProbeDetectionResult:
    """Per-token detection scores from a probe or vector set.

    ``scores[seq_idx, label_idx]`` is sigmoid prob in [0, 1] (probe path) or
    cosine similarity in [-1, 1] (vector path).
    """

    tokens: List[str]
    token_char_spans: List[Tuple[int, int]]
    scores: np.ndarray
    labels: List[str]
    formatted_text: str
    token_spans: List[TokenSpan]
    layer: int


@dataclass
class EvaluationResult:
    """Full classification metrics for a (label, layer, method, aggregation) row."""

    label: str
    layer: int
    method: str
    aggregation: str
    auroc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    threshold: float
    confusion_matrix: np.ndarray
    mean_present: float
    mean_absent: float
    n_present: int
    n_absent: int

    @property
    def discrimination(self) -> float:
        return self.mean_present - self.mean_absent

    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"label={self.label!r}, layer={self.layer}, method={self.method!r}, "
            f"agg={self.aggregation!r}, auroc={self.auroc:.3f}, f1={self.f1:.3f})"
        )


# ---------------------------------------------------------------------------
# Public functions — Session-backed wrappers
# ---------------------------------------------------------------------------


def probe_text(
    model: "LittleSteerModel",
    text: str,
    vector_set: "SteeringVectorSet",
    *,
    token_selection: TokenSelection | None = None,
    normalize: bool = True,
) -> dict[tuple[str, str, int], float]:
    """Cosine similarity of ``text``'s aggregated activation against every
    vector in ``vector_set``, per layer.

    ``token_selection`` defaults to the last token. Returns a dict keyed by
    ``(label, method, layer)``.
    """
    from .session import Session
    return Session(model).probe_text(
        text, vector_set, token_selection=token_selection, normalize=normalize,
    )


def score_dataset(
    model: "LittleSteerModel",
    dataset: list["ConversationEntry"],
    vector_set: "SteeringVectorSet",
    *,
    token_selection: TokenSelection | None = None,
    max_seq_len: int = 4096,
    normalize: bool = True,
    show_progress: bool = True,
) -> list[BehaviorScore]:
    """Discrimination score per (label, layer, method).

    See :class:`~little_steer.session.Session` for the cache-aware variant.
    """
    from .session import Session
    return Session(model, dataset, max_seq_len=max_seq_len).score(
        vector_set,
        token_selection=token_selection,
        normalize=normalize,
        show_progress=show_progress,
    )


def evaluate_dataset(
    model: "LittleSteerModel",
    dataset: List["ConversationEntry"],
    vector_set: "SteeringVectorSet",
    *,
    aggregation: str | List[str] = "mean",
    layers: Optional[List[int]] = None,
    label_filter: Optional[List[str]] = None,
    max_seq_len: int = 4096,
    normalize: bool = True,
    show_progress: bool = True,
    progress_fn=None,
) -> List[EvaluationResult]:
    """Full classification metrics — AUROC, F1, threshold, confusion matrix.

    ``aggregation`` accepts a single string (back-compat) or a list of strings.
    Passing a list runs all aggregations against the same forward pass — the
    fast path replacing the old "loop and call N times" pattern.
    """
    aggs = [aggregation] if isinstance(aggregation, str) else list(aggregation)
    from .session import Session
    return Session(model, dataset, max_seq_len=max_seq_len).evaluate(
        vector_set,
        aggregations=aggs,
        layers=layers,
        label_filter=label_filter,
        normalize=normalize,
        show_progress=show_progress,
        progress_fn=progress_fn,
    )


def get_token_similarities(
    model: "LittleSteerModel",
    entry: "ConversationEntry",
    vector: "SteeringVector",
    layers: Optional[List[int]] = None,
) -> TokenSimilarities:
    """Per-token cosine similarities at one or more layers."""
    from .session import Session
    out = Session(model, [entry]).token_similarities(entry, vector, layers=layers)
    if out is None:
        # The Session helper returns None when ctx is missing; build a minimal
        # TokenSimilarities so callers that always expect an object don't break.
        return TokenSimilarities(
            tokens=[], similarities={}, token_char_spans=[],
            token_spans=[], formatted_text="",
            label=vector.label, layer=vector.layer,
        )
    return out


def get_probe_predictions(
    model: "LittleSteerModel",
    entry: "ConversationEntry",
    probe: "MLPProbe | LinearProbeMultilabel",
    layer: int,
) -> ProbeDetectionResult:
    """Per-token sigmoid probabilities from a trained probe."""
    from .session import Session
    out = Session(model, [entry]).detect_with_probe(entry, probe, layer)
    if out is None:
        raise RuntimeError(f"Could not run probe predictions on '{entry.id}'")
    return out


def get_multilabel_token_scores(
    model: "LittleSteerModel",
    entry: "ConversationEntry",
    vector_set: "SteeringVectorSet",
    layer: int,
) -> ProbeDetectionResult:
    """Per-token cosine similarities, one column per vector at ``layer``."""
    from .session import Session
    out = Session(model, [entry]).detect_with_vectors(entry, vector_set, layer)
    if out is None:
        raise RuntimeError(f"Could not run vector detection on '{entry.id}'")
    return out


def vector_similarity_matrix(
    vector_set: "SteeringVectorSet",
) -> tuple[np.ndarray, list[str]]:
    """Pairwise cosine similarity between all vectors in a set.

    Diagnostic for label/method/layer redundancy. High off-diagonal values
    suggest the vectors capture overlapping directions.
    """
    vecs = list(vector_set)
    n = len(vecs)
    if n == 0:
        return np.array([]).reshape(0, 0), []

    mat = torch.stack([v.vector.float() for v in vecs])
    mat_norm = F.normalize(mat, dim=-1)
    sim = (mat_norm @ mat_norm.T).numpy()
    labels = [
        f"{v.label}|{v.method}|L{v.layer}|{v.extraction_spec}" for v in vecs
    ]
    return sim, labels
