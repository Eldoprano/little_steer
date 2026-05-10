"""Unified evaluation API.

A `Session` owns a model + (optionally) a dataset and caches forward passes
across calls. All read-side operations — discrimination scores, full
classification metrics, per-token similarities, probe detection — go through
a Session. The loose ``probing.score_dataset`` / ``evaluate_dataset`` /
``probe_text`` / ``get_token_similarities`` / ``get_probe_predictions``
functions remain as thin convenience wrappers, but they construct a
temporary Session under the hood, so any optimisation here lifts every
caller automatically.

Design notes:

* The forward-pass cache is keyed by ``(entry_id, layers, need_logits)``.
  Default capacity is 32 entries (LRU). Set ``cache_size=0`` to disable.
* Vector cosine similarities are computed in a single matmul per layer per
  aggregation by stacking vectors via
  :func:`._internal.span_eval.stack_vectors`.
* Multiple aggregations per call (``evaluate(..., aggregations=["mean",
  "first", "last"])``) share one forward pass per entry — N× speedup over
  the old per-aggregation loop.
"""

from __future__ import annotations

import gc
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ._internal.forward import ForwardPassResult, run_forward_pass
from ._internal.span_eval import (
    EntryContext,
    cosine_against_stack,
    prepare_entry,
    select_span_activation,
    stack_vectors,
)
from .extraction.config import TokenSelection

if TYPE_CHECKING:
    from .mlp_probe import LinearProbeMultilabel, MLPProbe
    from .models.model import LittleSteerModel
    from .vectors.steering_vector import SteeringVector, SteeringVectorSet
    from thesis_schema import ConversationEntry


# Reuse the public dataclasses so callers see the same return types.
from .probing import (
    BehaviorScore,
    EvaluationResult,
    ProbeDetectionResult,
    TokenSimilarities,
)


# Aggregation alias map.  ``sentence_*`` modes are kept around because plot code
# distinguishes them by name even though the math is identical.
_AGG_MAP: dict[str, TokenSelection] = {
    "mean":            TokenSelection("all",   aggregation="mean"),
    "sentence_mean":   TokenSelection("all",   aggregation="mean"),
    "first":           TokenSelection("first"),
    "sentence_first":  TokenSelection("first"),
    "last":            TokenSelection("last"),
    "sentence_last":   TokenSelection("last"),
}


def _resolve_aggregation(name: str) -> TokenSelection:
    if name not in _AGG_MAP:
        raise ValueError(
            f"Unknown aggregation {name!r}. Choose from: {list(_AGG_MAP)}"
        )
    return _AGG_MAP[name]


class Session:
    """Cached forward-pass facade for detection / probing.

    Example:
        session = ls.Session(model, entries)
        scores  = session.score(vectors)
        results = session.evaluate(vectors, aggregations=["mean", "first"])
        ts      = session.token_similarities(entries[0], vectors[0],
                                             layers=[10, 20, 30])

    The same forward pass on ``entries[0]`` is reused across all three calls
    above as long as the cache hasn't evicted it.
    """

    def __init__(
        self,
        model: "LittleSteerModel",
        entries: Iterable["ConversationEntry"] | None = None,
        *,
        max_seq_len: int = 4096,
        cache_size: int = 32,
    ):
        self.model = model
        self.entries: list = list(entries) if entries is not None else []
        self.max_seq_len = max_seq_len
        self.cache_size = max(0, int(cache_size))

        self._ctx_cache: "OrderedDict[str, EntryContext]" = OrderedDict()
        self._fwd_cache: "OrderedDict[tuple, ForwardPassResult]" = OrderedDict()

    # ------------------------------------------------------------------
    # Dataset management
    # ------------------------------------------------------------------

    def add(self, entries: Iterable["ConversationEntry"]) -> "Session":
        self.entries.extend(entries)
        return self

    def clear_cache(self) -> None:
        """Drop all cached contexts and forward passes (free CPU memory)."""
        self._ctx_cache.clear()
        self._fwd_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Cache primitives
    # ------------------------------------------------------------------

    def context(
        self,
        entry: "ConversationEntry",
        *,
        need_offset_mapping: bool = False,
    ) -> EntryContext | None:
        """Get the cached EntryContext for ``entry``, building it if missing.

        Returns None when ``entry`` has no annotations or no annotations
        survive token mapping.
        """
        key = entry.id
        cached = self._ctx_cache.get(key)
        # If the cached context is missing offset_mapping but the caller
        # needs it, rebuild.
        if cached is not None and (not need_offset_mapping or cached.offset_mapping is not None):
            self._ctx_cache.move_to_end(key)
            return cached

        ctx = prepare_entry(
            self.model,
            entry,
            max_seq_len=self.max_seq_len,
            need_offset_mapping=need_offset_mapping,
        )
        if ctx is None:
            return None
        self._ctx_cache[key] = ctx
        self._ctx_cache.move_to_end(key)
        # No bound on context cache — it's small (bookkeeping only, not activations).
        return ctx

    def forward(
        self,
        entry: "ConversationEntry",
        layers: Iterable[int],
        *,
        need_logits: bool = False,
    ) -> ForwardPassResult | None:
        """Run (or return cached) forward pass for ``entry`` capturing ``layers``.

        Returns None when the entry has no usable annotations (same semantics
        as ``context``).
        """
        ctx = self.context(entry)
        if ctx is None:
            return None

        layer_set = frozenset(int(l) for l in layers)
        if not layer_set and not need_logits:
            return ForwardPassResult(layer_acts={}, logits=None, seq_len=ctx.seq_len)

        cache_key = (entry.id, layer_set, bool(need_logits))
        if self.cache_size > 0:
            cached = self._fwd_cache.get(cache_key)
            if cached is not None:
                self._fwd_cache.move_to_end(cache_key)
                return cached

        try:
            result = run_forward_pass(
                self.model,
                ctx.token_ids,
                layer_set,
                need_logits=need_logits,
            )
        except torch.cuda.OutOfMemoryError:
            warnings.warn(
                f"OOM on '{entry.id}' (seq_len={ctx.seq_len}). Skipping. "
                f"Try reducing Session(max_seq_len=...)."
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None
        except Exception as e:
            warnings.warn(f"Error on '{entry.id}': {e}")
            return None

        if self.cache_size > 0:
            self._fwd_cache[cache_key] = result
            while len(self._fwd_cache) > self.cache_size:
                self._fwd_cache.popitem(last=False)

        return result

    # ------------------------------------------------------------------
    # Detection / probing
    # ------------------------------------------------------------------

    def evaluate(
        self,
        vectors: "SteeringVectorSet",
        *,
        aggregations: Iterable[str] = ("mean",),
        layers: Iterable[int] | None = None,
        label_filter: Iterable[str] | None = None,
        normalize: bool = True,
        show_progress: bool = True,
        progress_fn=None,
    ) -> list[EvaluationResult]:
        """Compute AUROC / F1 / precision / recall / confusion-matrix per
        (label, layer, method, aggregation).

        Aggregations share one forward pass per entry, so requesting
        ``("mean", "first", "last")`` is ~3× cheaper than calling this
        method three times.
        """
        from .vectors.steering_vector import SteeringVectorSet

        agg_list = list(aggregations)
        for a in agg_list:
            _resolve_aggregation(a)  # validate

        layer_filter = set(int(l) for l in layers) if layers is not None else None
        label_filter_set = set(label_filter) if label_filter is not None else None

        filtered = [
            v for v in vectors
            if (label_filter_set is None or v.label in label_filter_set)
            and (layer_filter is None or v.layer in layer_filter)
        ]
        if not filtered:
            return []
        filtered_set = SteeringVectorSet(filtered)

        required_layers = sorted({v.layer for v in filtered_set})
        vec_matrices, vec_keys = stack_vectors(filtered_set, normalize=normalize)
        # Pre-compute per-layer (label, method) ordering by row index so we can
        # accumulate efficiently.
        keys_by_layer: dict[int, list[tuple[str, str, int]]] = {}
        for k in vec_keys:
            keys_by_layer.setdefault(k[2], []).append(k)

        # Accumulators: agg → (label, method, layer) → {"present": [], "absent": []}
        acc: dict[str, dict[tuple[str, str, int], dict[str, list[float]]]] = {
            a: {k: {"present": [], "absent": []} for k in vec_keys}
            for a in agg_list
        }

        any_needs_logits = any(_resolve_aggregation(a).needs_logits for a in agg_list)

        if progress_fn is not None:
            iterator = progress_fn(self.entries)
        elif show_progress:
            iterator = tqdm(self.entries, desc="Evaluating dataset")
        else:
            iterator = self.entries

        for _eval_i, entry in enumerate(iterator):
            ctx = self.context(entry)
            if ctx is None:
                continue
            fwd = self.forward(entry, required_layers, need_logits=any_needs_logits)
            if fwd is None:
                continue

            if _eval_i % 50 == 49:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            for agg_name in agg_list:
                ts_agg = _resolve_aggregation(agg_name)
                # For each span: aggregate activation per layer → matmul against vec_matrix.
                per_layer_acts: dict[int, list[torch.Tensor]] = {l: [] for l in required_layers}
                per_span_label_sets: list[set[str]] = []

                for span in ctx.token_spans:
                    acts_for_span: dict[int, torch.Tensor] = {}
                    for layer_idx in required_layers:
                        layer_act = fwd.layer_acts.get(layer_idx)
                        if layer_act is None:
                            continue
                        agg_act = select_span_activation(
                            layer_act,
                            span,
                            ts_agg,
                            seq_len=ctx.seq_len,
                            span_logits=fwd.logits,
                        )
                        if agg_act is None:
                            continue
                        acts_for_span[layer_idx] = agg_act

                    if not acts_for_span:
                        continue

                    for layer_idx, act in acts_for_span.items():
                        per_layer_acts[layer_idx].append(act)
                    per_span_label_sets.append(set(span.labels))

                if not per_span_label_sets:
                    continue

                # One matmul per layer
                for layer_idx, acts_list in per_layer_acts.items():
                    if not acts_list:
                        continue
                    mat = torch.stack(acts_list)                              # (n_spans_at_layer, D)
                    sims = cosine_against_stack(mat, vec_matrices[layer_idx],
                                                normalize_acts=normalize)    # (n_spans, n_at_layer)
                    layer_keys = keys_by_layer[layer_idx]
                    sims_np = sims.detach().cpu().numpy()
                    # acts_list is built in order of per_span_label_sets;
                    # but acts_list might be shorter than per_span_label_sets
                    # if a span had no activations at this specific layer.
                    # We rebuild the alignment by tracking which spans
                    # contributed to this layer.
                    for span_idx_at_layer, sim_row in enumerate(sims_np):
                        # The i-th row of sims_np corresponds to the i-th item
                        # in acts_list, which is also the i-th span that had
                        # an activation at this layer. We don't currently
                        # track that mapping; the simpler invariant is "every
                        # span produces one activation at every layer or none
                        # for the whole entry", which holds because layer
                        # availability comes from forward.layer_acts (set
                        # globally) not from select_span_activation. So
                        # span_idx_at_layer == span_idx.
                        span_labels = per_span_label_sets[span_idx_at_layer]
                        for j, key in enumerate(layer_keys):
                            label = key[0]
                            bucket = "present" if label in span_labels else "absent"
                            acc[agg_name][key][bucket].append(float(sim_row[j]))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return _compile_eval_results(acc)

    def score(
        self,
        vectors: "SteeringVectorSet",
        *,
        token_selection: TokenSelection | None = None,
        normalize: bool = True,
        show_progress: bool = True,
    ) -> list[BehaviorScore]:
        """Compute BehaviorScore (mean_present − mean_absent) per
        (label, layer, method).

        Equivalent to ``evaluate`` but cheaper: only accumulates means and
        counts, not full per-sample arrays.
        """
        ts = token_selection or TokenSelection("all", aggregation="mean")
        from .vectors.steering_vector import SteeringVectorSet

        vec_list = list(vectors)
        if not vec_list:
            return []

        required_layers = sorted({v.layer for v in vec_list})
        vec_matrices, vec_keys = stack_vectors(SteeringVectorSet(vec_list),
                                               normalize=normalize)
        keys_by_layer: dict[int, list[tuple[str, str, int]]] = {}
        for k in vec_keys:
            keys_by_layer.setdefault(k[2], []).append(k)

        acc = {k: [0.0, 0, 0.0, 0] for k in vec_keys}  # sum_p, n_p, sum_a, n_a

        iterator = tqdm(self.entries, desc="Scoring dataset") if show_progress else self.entries

        for _score_i, entry in enumerate(iterator):
            ctx = self.context(entry)
            if ctx is None:
                continue
            fwd = self.forward(entry, required_layers, need_logits=ts.needs_logits)
            if fwd is None:
                continue

            if _score_i % 50 == 49:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            per_layer_acts: dict[int, list[torch.Tensor]] = {l: [] for l in required_layers}
            per_span_label_sets: list[set[str]] = []

            for span in ctx.token_spans:
                acts_for_span: dict[int, torch.Tensor] = {}
                for layer_idx in required_layers:
                    layer_act = fwd.layer_acts.get(layer_idx)
                    if layer_act is None:
                        continue
                    agg_act = select_span_activation(
                        layer_act, span, ts,
                        seq_len=ctx.seq_len,
                        span_logits=fwd.logits,
                    )
                    if agg_act is not None:
                        acts_for_span[layer_idx] = agg_act
                if not acts_for_span:
                    continue
                for layer_idx, act in acts_for_span.items():
                    per_layer_acts[layer_idx].append(act)
                per_span_label_sets.append(set(span.labels))

            if not per_span_label_sets:
                continue

            for layer_idx, acts_list in per_layer_acts.items():
                if not acts_list:
                    continue
                mat = torch.stack(acts_list)
                sims = cosine_against_stack(mat, vec_matrices[layer_idx],
                                            normalize_acts=normalize)
                sims_np = sims.detach().cpu().numpy()
                layer_keys = keys_by_layer[layer_idx]
                for span_idx, sim_row in enumerate(sims_np):
                    span_labels = per_span_label_sets[span_idx]
                    for j, key in enumerate(layer_keys):
                        bucket = "present" if key[0] in span_labels else "absent"
                        if bucket == "present":
                            acc[key][0] += float(sim_row[j])
                            acc[key][1] += 1
                        else:
                            acc[key][2] += float(sim_row[j])
                            acc[key][3] += 1

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scores: list[BehaviorScore] = []
        for (label, method, layer), (sp, np_, sa, na) in acc.items():
            scores.append(BehaviorScore(
                label=label, layer=layer, method=method,
                mean_present=sp / np_ if np_ > 0 else float("nan"),
                mean_absent=sa / na if na > 0 else float("nan"),
                n_present=int(np_), n_absent=int(na),
            ))
        scores.sort(key=lambda s: (s.label, s.method, s.layer))
        return scores

    def probe_text(
        self,
        text: str,
        vectors: "SteeringVectorSet",
        *,
        token_selection: TokenSelection | None = None,
        normalize: bool = True,
    ) -> dict[tuple[str, str, int], float]:
        """Cosine similarity of a raw text's (aggregated) activation against
        every vector in the set, per layer.

        Default ``token_selection`` is the last token — what the model "sees"
        next. This call is NOT cached (no entry id), so prefer
        ``evaluate``/``score`` for repeated dataset queries.
        """
        ts = token_selection or TokenSelection("last", aggregation="mean")
        vec_list = list(vectors)
        if not vec_list:
            return {}

        required_layers = sorted({v.layer for v in vec_list})

        encoding = self.model.tokenize(text)
        token_ids = encoding["input_ids"][0][: self.max_seq_len]
        seq_len = int(token_ids.shape[0])

        fwd = run_forward_pass(
            self.model, token_ids, required_layers, need_logits=ts.needs_logits,
        )

        from .vectors.steering_vector import SteeringVectorSet
        vec_matrices, vec_keys = stack_vectors(SteeringVectorSet(vec_list),
                                               normalize=normalize)
        keys_by_layer: dict[int, list[tuple[str, str, int]]] = {}
        for k in vec_keys:
            keys_by_layer.setdefault(k[2], []).append(k)

        # Treat the whole sequence as a single "span" via TokenSelection.apply
        from .data.tokenizer_utils import TokenSpan
        whole_span = TokenSpan(token_start=0, token_end=seq_len, labels=[])

        out: dict[tuple[str, str, int], float] = {}
        for layer_idx in required_layers:
            layer_act = fwd.layer_acts.get(layer_idx)
            if layer_act is None:
                continue
            agg_act = select_span_activation(
                layer_act, whole_span, ts,
                seq_len=seq_len, span_logits=fwd.logits,
            )
            if agg_act is None:
                continue
            sims = cosine_against_stack(
                agg_act.unsqueeze(0), vec_matrices[layer_idx],
                normalize_acts=normalize,
            ).squeeze(0).detach().cpu().numpy()
            for j, key in enumerate(keys_by_layer[layer_idx]):
                out[key] = float(sims[j])
        return out

    def token_similarities(
        self,
        entry: "ConversationEntry",
        vector: "SteeringVector",
        *,
        layers: list[int] | None = None,
    ) -> TokenSimilarities | None:
        """Per-token cosine similarities at one or more layers."""
        ctx = self.context(entry, need_offset_mapping=True)
        if ctx is None:
            return None

        required_layers = sorted(set(layers if layers else [vector.layer]))
        fwd = self.forward(entry, required_layers, need_logits=False)
        if fwd is None:
            return None

        sv_norm = F.normalize(vector.vector.float().unsqueeze(0), dim=-1).squeeze(0)
        similarities: dict[int, list[float]] = {}
        for layer_idx in required_layers:
            acts = fwd.layer_acts.get(layer_idx)
            if acts is None:
                continue
            sims = cosine_against_stack(
                acts, sv_norm.unsqueeze(0), normalize_acts=True,
            ).squeeze(-1)
            similarities[layer_idx] = sims.tolist()

        tokens = [self.model.tokenizer.decode([tid]) for tid in ctx.token_ids.tolist()]
        return TokenSimilarities(
            tokens=tokens,
            similarities=similarities,
            token_char_spans=ctx.offset_mapping or [],
            token_spans=ctx.token_spans,
            formatted_text=ctx.formatted_text,
            label=vector.label,
            layer=vector.layer,
        )

    def detect_with_probe(
        self,
        entry: "ConversationEntry",
        probe: "MLPProbe | LinearProbeMultilabel",
        layer: int,
    ) -> ProbeDetectionResult | None:
        """Per-token sigmoid probabilities for each label in ``probe``."""
        ctx = self.context(entry, need_offset_mapping=True)
        if ctx is None:
            return None
        fwd = self.forward(entry, [layer])
        if fwd is None or layer not in fwd.layer_acts:
            return None

        acts = fwd.layer_acts[layer]
        probe_cpu = probe.to("cpu")
        with torch.no_grad():
            logits = probe_cpu(acts.float())
            probs = torch.sigmoid(logits).numpy().astype(np.float32)

        tokens = [self.model.tokenizer.decode([tid]) for tid in ctx.token_ids.tolist()]
        return ProbeDetectionResult(
            tokens=tokens,
            token_char_spans=ctx.offset_mapping or [],
            scores=probs,
            labels=probe.labels,
            formatted_text=ctx.formatted_text,
            token_spans=ctx.token_spans,
            layer=layer,
        )

    def detect_with_vectors(
        self,
        entry: "ConversationEntry",
        vectors: "SteeringVectorSet",
        layer: int,
    ) -> ProbeDetectionResult | None:
        """Per-token cosine similarities, one column per vector at ``layer``."""
        vecs_at_layer = [v for v in vectors if v.layer == layer]
        if not vecs_at_layer:
            raise ValueError(
                f"No vectors at layer {layer}. "
                f"Available: {sorted({v.layer for v in vectors})}"
            )

        ctx = self.context(entry, need_offset_mapping=True)
        if ctx is None:
            return None
        fwd = self.forward(entry, [layer])
        if fwd is None or layer not in fwd.layer_acts:
            return None

        acts = fwd.layer_acts[layer]
        from .vectors.steering_vector import SteeringVectorSet
        vec_matrix, vec_keys = stack_vectors(
            SteeringVectorSet(vecs_at_layer), normalize=True,
        )
        scores = cosine_against_stack(
            acts, vec_matrix[layer], normalize_acts=True,
        ).numpy().astype(np.float32)
        labels = [k[0] for k in vec_keys]

        tokens = [self.model.tokenizer.decode([tid]) for tid in ctx.token_ids.tolist()]
        return ProbeDetectionResult(
            tokens=tokens,
            token_char_spans=ctx.offset_mapping or [],
            scores=scores,
            labels=labels,
            formatted_text=ctx.formatted_text,
            token_spans=ctx.token_spans,
            layer=layer,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compile_eval_results(
    acc: dict[str, dict[tuple[str, str, int], dict[str, list[float]]]],
) -> list[EvaluationResult]:
    """Turn accumulated present/absent similarity buckets into EvaluationResults."""
    from sklearn.metrics import (
        confusion_matrix as sk_cm,
        precision_recall_curve,
        roc_auc_score,
    )

    results: list[EvaluationResult] = []
    for agg_name, per_key in acc.items():
        for (label, method, layer), buckets in per_key.items():
            present, absent = buckets["present"], buckets["absent"]
            n_p, n_a = len(present), len(absent)
            if n_p == 0 and n_a == 0:
                continue

            mean_p = float(np.mean(present)) if present else float("nan")
            mean_a = float(np.mean(absent)) if absent else float("nan")

            if n_p == 0 or n_a == 0:
                results.append(EvaluationResult(
                    label=label, layer=layer, method=method, aggregation=agg_name,
                    auroc=float("nan"), auprc=float("nan"), f1=0.0, precision=0.0, recall=0.0,
                    threshold=0.0, confusion_matrix=np.zeros((2, 2), dtype=int),
                    mean_present=mean_p, mean_absent=mean_a,
                    n_present=n_p, n_absent=n_a,
                ))
                continue

            y_scores = present + absent
            y_true = [1] * n_p + [0] * n_a
            try:
                auroc = float(roc_auc_score(y_true, y_scores))
            except Exception:
                auroc = float("nan")

            from sklearn.metrics import average_precision_score
            try:
                auprc = float(average_precision_score(y_true, y_scores))
            except Exception:
                auprc = float("nan")

            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            
            # Add a small epsilon to avoid division by zero
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-15)
            best_idx = int(np.argmax(f1_scores))
            
            best_f1 = float(f1_scores[best_idx])
            best_p = float(precision[best_idx])
            best_r = float(recall[best_idx])
            
            # precision_recall_curve returns thresholds array of length N-1
            # so the last precision/recall value (which is 1.0, 0.0) doesn't have a threshold
            if best_idx < len(thresholds):
                best_thr = float(thresholds[best_idx])
            else:
                best_thr = float(max(y_scores)) if y_scores else 0.0

            preds = [1 if s >= best_thr else 0 for s in y_scores]
            cm = sk_cm(y_true, preds, labels=[0, 1])

            results.append(EvaluationResult(
                label=label, layer=layer, method=method, aggregation=agg_name,
                auroc=auroc, auprc=auprc, f1=best_f1, precision=best_p, recall=best_r,
                threshold=best_thr, confusion_matrix=cm,
                mean_present=mean_p, mean_absent=mean_a,
                n_present=n_p, n_absent=n_a,
            ))
    results.sort(key=lambda r: (r.label, r.method, r.layer, r.aggregation))
    return results
