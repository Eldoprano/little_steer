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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .extraction.config import TokenSelection
from .extraction.pooling import apply_aggregation
from .data.tokenizer_utils import TokenPositionMapper

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
