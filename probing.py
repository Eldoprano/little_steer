"""
little_steer.probing

Functions for reading steering vector signal from model activations.

Two main entry points:

  probe_text(model, text, vector_set)
      Run the model on a string and compute cosine similarity of the (aggregated)
      activations against every vector in the set, per layer.
      Returns a dict {(label, method, layer) → cosine_similarity}.

  score_dataset(model, dataset, vector_set)
      Over a labeled dataset, measure how well each vector discriminates between
      spans *with* its target label vs spans *without* it.
      Returns a list of BehaviorScore objects — sort by .discrimination to find
      the most informative layer for each behaviour.

Typical workflow for layer selection:

    scores = ls.score_dataset(model, dataset, vectors.filter(method="pca"))
    scores.sort(key=lambda s: -s.discrimination)
    best = scores[0]
    print(f"Best layer: {best.layer}  discrimination={best.discrimination:+.3f}")
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .extraction.config import TokenSelection
from .extraction.pooling import apply_aggregation
from .data.tokenizer_utils import TokenPositionMapper, TokenSpan

if TYPE_CHECKING:
    from .models.model import LittleSteerModel
    from .vectors.steering_vector import SteeringVector, SteeringVectorSet
    from .data.schema import ConversationEntry


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class BehaviorScore:
    """Discrimination score for a single (label, layer, method) combination.

    Captures how well a steering vector's direction aligns with positions in
    the dataset that carry its target behaviour vs. positions that don't.

    Attributes:
        label:         The behaviour label the vector was built for.
        layer:         Transformer layer index.
        method:        Extraction method (e.g. 'pca', 'mean_difference').
        mean_present:  Mean cosine similarity on annotated spans WITH this label.
        mean_absent:   Mean cosine similarity on annotated spans WITHOUT this label.
        n_present:     Number of spans in the 'present' group.
        n_absent:      Number of spans in the 'absent' group.
        discrimination: mean_present − mean_absent — the primary sorting key.
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
        """mean_present − mean_absent.  Higher = vector better identifies the label."""
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
    """Per-token cosine similarities against a steering vector at one or more layers.

    Produced by :func:`get_token_similarities`. Contains everything needed to
    render a token-level visualization: the token strings, their character
    spans, per-layer similarity values, and the resolved annotation spans.

    Attributes:
        tokens:          Decoded token strings, length = seq_len.
        similarities:    layer_idx → list of cosine similarities (one per token).
        token_char_spans: (char_start, char_end) per token in ``formatted_text``.
        token_spans:     Resolved annotation spans with token indices and labels.
        formatted_text:  The full chat-template-formatted text.
        label:           Target label of the steering vector used.
        layer:           Vector's native layer (default display layer).
    """

    tokens: List[str]
    similarities: Dict[int, List[float]]
    token_char_spans: List[Tuple[int, int]]
    token_spans: List[TokenSpan]
    formatted_text: str
    label: str
    layer: int


@dataclass
class EvaluationResult:
    """Classification metrics for a single (label, layer, method, aggregation) combination.

    Produced by :func:`evaluate_dataset`. Captures how well a steering vector
    discriminates labelled spans at a given layer using a chosen threshold.

    Attributes:
        label:            The behaviour label the vector was built for.
        layer:            Transformer layer index.
        method:           Extraction method (e.g. 'pca', 'mean_difference').
        aggregation:      Token aggregation used ('mean', 'first', 'last',
                          'sentence_mean', 'sentence_first', 'sentence_last').
        auroc:            Area under the ROC curve (NaN if only one class present).
        f1:               F1 score at the threshold that maximises F1.
        precision:        Precision at the optimal threshold.
        recall:           Recall at the optimal threshold.
        threshold:        Cosine similarity threshold that maximises F1.
        confusion_matrix: 2×2 array [[TN, FP], [FN, TP]].
        mean_present:     Mean similarity on spans WITH the label.
        mean_absent:      Mean similarity on spans WITHOUT the label.
        n_present:        Number of 'present' spans.
        n_absent:         Number of 'absent' spans.
    """

    label: str
    layer: int
    method: str
    aggregation: str
    auroc: float
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
        """mean_present − mean_absent."""
        return self.mean_present - self.mean_absent

    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"label={self.label!r}, layer={self.layer}, method={self.method!r}, "
            f"agg={self.aggregation!r}, auroc={self.auroc:.3f}, f1={self.f1:.3f})"
        )


# ---------------------------------------------------------------------------
# probe_text
# ---------------------------------------------------------------------------


def probe_text(
    model: "LittleSteerModel",
    text: str,
    vector_set: "SteeringVectorSet",
    *,
    token_selection: TokenSelection | None = None,
    normalize: bool = True,
) -> dict[tuple[str, str, int], float]:
    """Compute cosine similarity of model activations with a set of steering vectors.

    Runs a single forward pass on `text` and compares the (aggregated) activation
    at each required layer against every vector in `vector_set`.

    Args:
        model:           LittleSteerModel instance.
        text:            Input text, already formatted (e.g. via model.format_messages()).
        vector_set:      Steering vectors to probe against.
        token_selection: How to aggregate tokens from the full sequence.
                         Defaults to TokenSelection("last") — the last token,
                         which is what the model "sees" next.
        normalize:       If True, L2-normalise both activations and vectors before
                         computing the dot product (gives true cosine similarity).

    Returns:
        Dict mapping (label, method, layer) → cosine similarity in [−1, 1].

    Example:
        text = model.format_messages([{"role": "user", "content": "How do I..."}],
                                     add_generation_prompt=True)
        sims = ls.probe_text(model, text, vectors.filter(method="pca"))
        for (label, method, layer), sim in sorted(sims.items()):
            print(f"  {label} | layer {layer} | {sim:+.3f}")
    """
    if token_selection is None:
        token_selection = TokenSelection("last", aggregation="mean")

    if len(vector_set) == 0:
        return {}

    required_layers = sorted(set(v.layer for v in vector_set))
    max_layer = max(required_layers)
    stop_early = max_layer < model.num_layers - 1

    encoding = model.tokenize(text)
    token_ids = encoding["input_ids"][0]
    seq_len = len(token_ids)

    needs_logits = token_selection.needs_logits
    logits_saved = None

    # Forward pass — collect required layers only
    layer_acts: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        with model.trace(token_ids) as tracer:
            for layer_idx in required_layers:
                layer_acts[layer_idx] = (
                    model.layers_output[layer_idx]
                    .squeeze(0)        # (seq_len, hidden_dim)
                    .detach()
                    .cpu()
                    .save()
                )
            if needs_logits:
                logits_saved = (
                    model.st.output.logits
                    .squeeze(0)        # (seq_len, vocab_size)
                    .detach()
                    .cpu()
                    .save()
                )
            elif stop_early:
                tracer.stop()

    logits: torch.Tensor | None = logits_saved

    # Token selection over the full sequence
    resolved = token_selection.apply(0, seq_len, seq_len)
    if resolved is None:
        return {}
    sel_start, sel_end = resolved
    sel_logits = logits[sel_start:sel_end] if logits is not None else None

    results: dict[tuple[str, str, int], float] = {}
    for vec in vector_set:
        if vec.layer not in layer_acts:
            continue

        act = layer_acts[vec.layer]       # (seq_len, hidden_dim)
        selected = act[sel_start:sel_end]  # (n_tokens, hidden_dim)
        if selected.shape[0] == 0:
            continue

        raw = apply_aggregation(
            selected,
            token_selection.aggregation,
            token_logits=sel_logits,
            confidence_threshold=token_selection.confidence_threshold,
            high_entropy_fraction=token_selection.high_entropy_fraction,
        )
        activation = raw.float() if raw.dim() == 1 else raw[-1].float()  # "none" fallback

        sv = vec.vector.float()
        if normalize:
            activation = F.normalize(activation.unsqueeze(0), dim=-1).squeeze(0)
            sv = F.normalize(sv.unsqueeze(0), dim=-1).squeeze(0)

        sim = (activation * sv.to(activation.device)).sum().item()
        results[(vec.label, vec.method, vec.layer)] = sim

    del layer_acts
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# get_token_similarities
# ---------------------------------------------------------------------------


def get_token_similarities(
    model: "LittleSteerModel",
    entry: "ConversationEntry",
    vector: "SteeringVector",
    layers: Optional[List[int]] = None,
) -> TokenSimilarities:
    """Compute per-token cosine similarity against a steering vector at each layer.

    Runs a single forward pass on the formatted entry and collects the residual
    stream at every requested layer. Returns one similarity value per token per
    layer — the raw signal before any span-level aggregation.

    Useful for:
      - Visualising how vector similarity evolves token-by-token.
      - Understanding which specific sentences or words drive high similarity.
      - Multi-layer similarity heatmaps.

    Args:
        model:   LittleSteerModel instance.
        entry:   Conversation entry (with annotations for span highlighting).
        vector:  Steering vector to probe against.
        layers:  Layer indices to collect. Defaults to ``[vector.layer]``.

    Returns:
        :class:`TokenSimilarities` containing per-token, per-layer similarities
        along with token strings, character spans, and resolved annotation spans.

    Example:
        ts = ls.get_token_similarities(model, entry, vec, layers=[10, 20, 30])
        print(ts.tokens[:5])            # first 5 decoded tokens
        print(ts.similarities[20][:5])  # layer 20, first 5 tokens
    """
    required_layers = sorted(set(layers if layers else [vector.layer]))
    max_layer = max(required_layers)
    stop_early = max_layer < model.num_layers - 1

    formatted_text, message_offsets = model.format_messages_with_offsets(entry.messages)
    token_spans = model._token_mapper.map_annotations_to_tokens(
        entry, formatted_text, message_offsets
    )

    encoding = model.tokenize(formatted_text, return_offsets_mapping=True)
    token_ids = encoding["input_ids"][0]
    offset_mapping = encoding["offset_mapping"][0].tolist()

    # Pre-normalise the vector once
    sv_norm = F.normalize(vector.vector.float().unsqueeze(0), dim=-1).squeeze(0)

    layer_acts: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        with model.trace(token_ids) as tracer:
            for layer_idx in required_layers:
                layer_acts[layer_idx] = (
                    model.layers_output[layer_idx]
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .save()
                )
            if stop_early:
                tracer.stop()

    similarities: Dict[int, List[float]] = {}
    for layer_idx in required_layers:
        acts = layer_acts[layer_idx]                          # (seq_len, hidden_dim)
        acts_norm = F.normalize(acts.float(), dim=-1)         # (seq_len, hidden_dim)
        sims = (acts_norm * sv_norm).sum(dim=-1)              # (seq_len,)
        similarities[layer_idx] = sims.tolist()

    tokens = [model.tokenizer.decode([tid]) for tid in token_ids.tolist()]
    token_char_spans = [(int(s), int(e)) for s, e in offset_mapping]

    del layer_acts
    gc.collect()
    torch.cuda.empty_cache()

    return TokenSimilarities(
        tokens=tokens,
        similarities=similarities,
        token_char_spans=token_char_spans,
        token_spans=token_spans,
        formatted_text=formatted_text,
        label=vector.label,
        layer=vector.layer,
    )


# ---------------------------------------------------------------------------
# evaluate_dataset
# ---------------------------------------------------------------------------


def evaluate_dataset(
    model: "LittleSteerModel",
    dataset: List["ConversationEntry"],
    vector_set: "SteeringVectorSet",
    *,
    aggregation: str = "mean",
    layers: Optional[List[int]] = None,
    label_filter: Optional[List[str]] = None,
    max_seq_len: int = 4096,
    normalize: bool = True,
    show_progress: bool = True,
) -> List[EvaluationResult]:
    """Evaluate steering vectors with full classification metrics.

    Like :func:`score_dataset` but computes per-span scores, then derives
    AUROC, F1 at the optimal threshold, precision, recall, and a confusion
    matrix for each (label, layer, method) combination.

    Aggregation modes:
        ``'mean'`` / ``'sentence_mean'``   — mean over all tokens in the span.
        ``'first'`` / ``'sentence_first'`` — first token of the span.
        ``'last'`` / ``'sentence_last'``   — last token of the span.

    The ``sentence_*`` variants are semantically identical to their plain
    counterparts but are stored under a different ``aggregation`` string in
    the result, which helps when comparing multiple modes on one plot.

    Args:
        model:         LittleSteerModel instance.
        dataset:       List of ConversationEntry objects with annotations.
        vector_set:    Steering vectors to evaluate.
        aggregation:   Token aggregation strategy (see above).
        layers:        If set, only evaluate vectors at these layer indices.
        label_filter:  If set, only evaluate vectors with these labels.
        max_seq_len:   Truncation limit for long sequences.
        normalize:     L2-normalise activations and vectors before similarity.
        show_progress: Show a tqdm progress bar.

    Returns:
        List of :class:`EvaluationResult`, sorted by (label, method, layer).

    Example:
        results = ls.evaluate_dataset(model, dataset, vectors, aggregation="mean")
        # Best AUROC per label
        for r in sorted(results, key=lambda r: -r.auroc):
            print(f"{r.label} L{r.layer}: AUROC={r.auroc:.3f} F1={r.f1:.3f}")
    """
    # Map aggregation string to TokenSelection
    _agg_map = {
        "mean": TokenSelection("all", aggregation="mean"),
        "sentence_mean": TokenSelection("all", aggregation="mean"),
        "first": TokenSelection("first"),
        "sentence_first": TokenSelection("first"),
        "last": TokenSelection("last"),
        "sentence_last": TokenSelection("last"),
    }
    if aggregation not in _agg_map:
        raise ValueError(
            f"Unknown aggregation {aggregation!r}. "
            f"Choose from: {list(_agg_map)}"
        )
    token_selection = _agg_map[aggregation]

    # Apply filters
    from .vectors.steering_vector import SteeringVectorSet
    vecs = [
        v for v in vector_set
        if (label_filter is None or v.label in label_filter)
        and (layers is None or v.layer in layers)
    ]
    if not vecs:
        return []
    filtered_vecs = SteeringVectorSet(vecs)

    required_layers = sorted(set(v.layer for v in filtered_vecs))
    max_layer = max(required_layers)
    stop_early = max_layer < model.num_layers - 1
    needs_logits = token_selection.needs_logits

    # Pre-normalise vectors once
    vec_tensors: dict[tuple, torch.Tensor] = {}
    for v in filtered_vecs:
        sv = v.vector.float()
        if normalize:
            sv = F.normalize(sv.unsqueeze(0), dim=-1).squeeze(0)
        vec_tensors[(v.label, v.method, v.layer)] = sv

    # Accumulators: key → {"present": [float], "absent": [float]}
    acc: dict[tuple, dict[str, list]] = {
        k: {"present": [], "absent": []} for k in vec_tensors
    }

    token_mapper = TokenPositionMapper(model.tokenizer)
    iterator = tqdm(dataset, desc="Evaluating dataset") if show_progress else dataset

    for entry in iterator:
        if not entry.annotations:
            continue

        try:
            formatted_text, message_offsets = model.format_messages_with_offsets(
                entry.messages
            )
            token_spans = token_mapper.map_annotations_to_tokens(
                entry=entry,
                formatted_text=formatted_text,
                message_offsets=message_offsets,
            )
            if not token_spans:
                continue

            encoding = model.tokenize(formatted_text)
            token_ids = encoding["input_ids"][0]
            seq_len = min(len(token_ids), max_seq_len)
            token_ids = token_ids[:seq_len]

            layer_acts: dict[int, torch.Tensor] = {}
            entry_logits_saved = None
            with torch.no_grad():
                with model.trace(token_ids) as tracer:
                    for layer_idx in required_layers:
                        layer_acts[layer_idx] = (
                            model.layers_output[layer_idx]
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .save()
                        )
                    if needs_logits:
                        entry_logits_saved = (
                            model.st.output.logits
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .save()
                        )
                    elif stop_early:
                        tracer.stop()

            entry_logits: torch.Tensor | None = entry_logits_saved

            # Compute one scalar per span per layer
            span_data: list[tuple[set[str], dict[int, torch.Tensor]]] = []
            for ts in token_spans:
                if ts.token_start >= seq_len:
                    continue
                eff_end = min(ts.token_end, seq_len)
                resolved = token_selection.apply(ts.token_start, eff_end, seq_len)
                if resolved is None:
                    continue
                sel_start, sel_end = resolved

                span_acts: dict[int, torch.Tensor] = {}
                for layer_idx in required_layers:
                    if layer_idx not in layer_acts:
                        continue
                    selected = layer_acts[layer_idx][sel_start:sel_end]
                    if selected.shape[0] == 0:
                        continue

                    sel_logits = (
                        entry_logits[sel_start:sel_end]
                        if entry_logits is not None else None
                    )
                    raw = apply_aggregation(
                        selected,
                        token_selection.aggregation,
                        token_logits=sel_logits,
                        confidence_threshold=token_selection.confidence_threshold,
                        high_entropy_fraction=token_selection.high_entropy_fraction,
                    )
                    act = raw.float() if raw.dim() == 1 else raw[-1].float()
                    if normalize:
                        act = F.normalize(act.unsqueeze(0), dim=-1).squeeze(0)
                    span_acts[layer_idx] = act

                if span_acts:
                    span_data.append((set(ts.labels), span_acts))

            for (label, method, layer), sv in vec_tensors.items():
                for span_labels, span_acts in span_data:
                    if layer not in span_acts:
                        continue
                    sim = (span_acts[layer] * sv.to(span_acts[layer].device)).sum().item()
                    bucket = "present" if label in span_labels else "absent"
                    acc[(label, method, layer)][bucket].append(sim)

            del layer_acts

        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            warnings.warn(f"OOM on '{entry.id}', skipping.")
            continue
        except Exception as e:
            warnings.warn(f"Error on '{entry.id}': {e}")
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # Compute metrics
    results: List[EvaluationResult] = []
    for (label, method, layer), buckets in acc.items():
        present = buckets["present"]
        absent = buckets["absent"]
        n_p, n_a = len(present), len(absent)

        if n_p == 0 and n_a == 0:
            continue

        mean_present = float(np.mean(present)) if present else float("nan")
        mean_absent = float(np.mean(absent)) if absent else float("nan")

        y_scores = present + absent
        y_true = [1] * n_p + [0] * n_a

        # AUROC
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y_true, y_scores)) if len(set(y_true)) > 1 else float("nan")
        except Exception:
            auroc = float("nan")

        # Optimal threshold (maximise F1)
        best_f1, best_thr = 0.0, 0.0
        best_prec, best_rec = 0.0, 0.0
        if y_scores:
            from sklearn.metrics import f1_score, precision_recall_fscore_support
            thresholds = np.linspace(min(y_scores), max(y_scores), 50)
            for thr in thresholds:
                preds = [1 if s >= thr else 0 for s in y_scores]
                f1 = f1_score(y_true, preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = float(thr)
                    p, r, _, _ = precision_recall_fscore_support(
                        y_true, preds, average="binary", zero_division=0
                    )
                    best_prec, best_rec = float(p), float(r)

        # Confusion matrix at best threshold
        preds = [1 if s >= best_thr else 0 for s in y_scores]
        from sklearn.metrics import confusion_matrix as sk_cm
        cm = sk_cm(y_true, preds, labels=[0, 1])

        results.append(EvaluationResult(
            label=label,
            layer=layer,
            method=method,
            aggregation=aggregation,
            auroc=auroc,
            f1=best_f1,
            precision=best_prec,
            recall=best_rec,
            threshold=best_thr,
            confusion_matrix=cm,
            mean_present=mean_present,
            mean_absent=mean_absent,
            n_present=n_p,
            n_absent=n_a,
        ))

    results.sort(key=lambda r: (r.label, r.method, r.layer))
    return results


# ---------------------------------------------------------------------------
# score_dataset
# ---------------------------------------------------------------------------


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
    """Score how well each steering vector discriminates labelled spans.

    For every vector in `vector_set`, accumulates:
      - "present" similarities: spans in the dataset annotated WITH that vector's label
      - "absent"  similarities: spans annotated with OTHER labels (but not this one)

    The difference (`BehaviorScore.discrimination = mean_present − mean_absent`) tells
    you how strongly the vector's direction correlates with the target behaviour.  Use
    this to:
      - Pick the best layer for a given (label, method) pair.
      - Compare extraction methods (pca vs mean_difference vs …).
      - Validate that a vector actually carries signal before using it for steering.

    Only annotated spans are used for both groups — uninformative padding and
    off-content tokens are excluded automatically.

    Args:
        model:           LittleSteerModel instance.
        dataset:         List of ConversationEntry objects with annotations.
        vector_set:      Steering vectors to evaluate.
        token_selection: How to aggregate tokens within each span.
                         Defaults to TokenSelection("all", aggregation="mean"),
                         meaning the mean over all tokens in the span — consistent
                         with the 'whole_sentence' extraction spec.
        max_seq_len:     Truncation limit for very long sequences.
        normalize:       L2-normalise activations and vectors before similarity.
        show_progress:   Show a tqdm progress bar over the dataset.

    Returns:
        List of BehaviorScore, one per unique (label, layer, method) in the set,
        sorted by (label, method, layer).

    Example:
        scores = ls.score_dataset(model, dataset, vectors)
        # Sort by discrimination to find the most informative layer
        for s in sorted(scores, key=lambda s: -s.discrimination):
            print(s)
    """
    if token_selection is None:
        token_selection = TokenSelection("all", aggregation="mean")

    if len(vector_set) == 0:
        return []

    required_layers = sorted(set(v.layer for v in vector_set))
    max_layer = max(required_layers)
    stop_early = max_layer < model.num_layers - 1
    needs_logits = token_selection.needs_logits

    # Pre-normalise vectors once
    vec_tensors: dict[tuple[str, str, int], torch.Tensor] = {}
    for v in vector_set:
        sv = v.vector.float()
        if normalize:
            sv = F.normalize(sv.unsqueeze(0), dim=-1).squeeze(0)
        vec_tensors[(v.label, v.method, v.layer)] = sv

    # Accumulators: key → [sum_present, n_present, sum_absent, n_absent]
    acc: dict[tuple[str, str, int], list[float | int]] = {
        k: [0.0, 0, 0.0, 0] for k in vec_tensors
    }

    token_mapper = TokenPositionMapper(model.tokenizer)
    iterator = tqdm(dataset, desc="Scoring dataset") if show_progress else dataset

    for entry in iterator:
        if not entry.annotations:
            continue

        try:
            formatted_text, message_offsets = model.format_messages_with_offsets(
                entry.messages
            )
            token_spans = token_mapper.map_annotations_to_tokens(
                entry=entry,
                formatted_text=formatted_text,
                message_offsets=message_offsets,
            )
            if not token_spans:
                continue

            encoding = model.tokenize(formatted_text)
            token_ids = encoding["input_ids"][0]
            seq_len = min(len(token_ids), max_seq_len)
            token_ids = token_ids[:seq_len]

            # Single forward pass for all required layers
            layer_acts: dict[int, torch.Tensor] = {}
            entry_logits_saved = None
            with torch.no_grad():
                with model.trace(token_ids) as tracer:
                    for layer_idx in required_layers:
                        layer_acts[layer_idx] = (
                            model.layers_output[layer_idx]
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .save()
                        )
                    if needs_logits:
                        entry_logits_saved = (
                            model.st.output.logits
                            .squeeze(0)        # (seq_len, vocab_size)
                            .detach()
                            .cpu()
                            .save()
                        )
                    elif stop_early:
                        tracer.stop()

            entry_logits: torch.Tensor | None = entry_logits_saved

            # Compute one activation vector per span, per layer
            span_data: list[tuple[set[str], dict[int, torch.Tensor]]] = []
            for ts in token_spans:
                if ts.token_start >= seq_len:
                    continue
                eff_end = min(ts.token_end, seq_len)
                resolved = token_selection.apply(ts.token_start, eff_end, seq_len)
                if resolved is None:
                    continue
                sel_start, sel_end = resolved

                span_acts: dict[int, torch.Tensor] = {}
                for layer_idx in required_layers:
                    if layer_idx not in layer_acts:
                        continue
                    selected = layer_acts[layer_idx][sel_start:sel_end]
                    if selected.shape[0] == 0:
                        continue

                    sel_logits = (
                        entry_logits[sel_start:sel_end]
                        if entry_logits is not None else None
                    )
                    raw = apply_aggregation(
                        selected,
                        token_selection.aggregation,
                        token_logits=sel_logits,
                        confidence_threshold=token_selection.confidence_threshold,
                        high_entropy_fraction=token_selection.high_entropy_fraction,
                    )
                    act = raw.float() if raw.dim() == 1 else raw[-1].float()  # "none" fallback

                    if normalize:
                        act = F.normalize(act.unsqueeze(0), dim=-1).squeeze(0)
                    span_acts[layer_idx] = act

                if span_acts:
                    span_data.append((set(ts.labels), span_acts))

            # Accumulate cosine similarities
            for (label, method, layer), sv in vec_tensors.items():
                for span_labels, span_acts in span_data:
                    if layer not in span_acts:
                        continue
                    sim = (span_acts[layer] * sv.to(span_acts[layer].device)).sum().item()
                    if label in span_labels:
                        acc[(label, method, layer)][0] += sim   # sum_present
                        acc[(label, method, layer)][1] += 1     # n_present
                    else:
                        acc[(label, method, layer)][2] += sim   # sum_absent
                        acc[(label, method, layer)][3] += 1     # n_absent

            del layer_acts

        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            warnings.warn(f"OOM on '{entry.id}', skipping.")
            continue
        except Exception as e:
            warnings.warn(f"Error on '{entry.id}': {e}")
            continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    scores: list[BehaviorScore] = []
    for (label, method, layer), (sp, np_, sa, na) in acc.items():
        scores.append(
            BehaviorScore(
                label=label,
                layer=layer,
                method=method,
                mean_present=sp / np_ if np_ > 0 else float("nan"),
                mean_absent=sa / na if na > 0 else float("nan"),
                n_present=int(np_),
                n_absent=int(na),
            )
        )

    scores.sort(key=lambda s: (s.label, s.method, s.layer))
    return scores
