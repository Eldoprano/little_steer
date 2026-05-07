"""
little_steer.extraction.extractor

ActivationExtractor: the main extraction engine.

Uses nnterp's StandardizedTransformer for model-agnostic layer access,
following nnsight best practices:
  - Layers accessed in forward-pass order (ascending layer index)
  - tracer.stop() used when max required layer < last layer (saves computation)
  - .detach().cpu().save() immediately to avoid VRAM buildup
  - gc.collect() + torch.cuda.empty_cache() between conversations
"""

from __future__ import annotations

import gc
import time
import warnings
from typing import TYPE_CHECKING

import torch
from tqdm.auto import tqdm

from thesis_schema import ConversationEntry
from ..data.tokenizer_utils import TokenPositionMapper
from .config import ExtractionPlan
from .pooling import apply_aggregation
from .result import ExtractionMetadata, ExtractionResult

if TYPE_CHECKING:
    from ..models.model import LittleSteerModel
    from ..models.vram import BatchConfig, VRAMManager


class ActivationExtractor:
    """Extract activations from conversations according to an ExtractionPlan.

    For each conversation:
      1. Format messages → apply chat template → tokenize
      2. Find where each message's content is in the formatted string
      3. Map annotation char positions → token positions (via offset_mapping)
      4. Run one model forward pass with nnterp (using tracer.stop() for efficiency)
      5. For each annotation × each ExtractionSpec: apply TokenSelection and aggregate

    OOM protection:
      - Allows truncation to a max_seq_len
      - Catches CUDA OOM, clears cache, and skips the conversation with a warning

    Example:
        model = LittleSteerModel("Qwen/Qwen3-8B")
        plan = ExtractionPlan("my_plan", specs={
            "last_token": ExtractionSpec(TokenSelection("last"), layers=[20, 25]),
            "whole_sentence": ExtractionSpec(TokenSelection("all"), layers=[20, 25]),
        })
        extractor = ActivationExtractor(model)
        result = extractor.extract(dataset, plan)
    """

    def __init__(
        self,
        model: "LittleSteerModel",
        vram_manager: "VRAMManager | None" = None,
        max_seq_len: int = 4096,
    ):
        self.model = model
        self.max_seq_len = max_seq_len
        self._token_mapper = TokenPositionMapper(model.tokenizer)

        if vram_manager is not None:
            saved = vram_manager.load_config(model.model_id)
            if saved is not None:
                self.max_seq_len = saved.max_seq_len
                print(f"📋 Loaded VRAM config: max_seq_len={self.max_seq_len}")

    def extract(
        self,
        dataset: list[ConversationEntry],
        plan: ExtractionPlan,
        show_progress: bool = True,
        progress_fn=None,
    ) -> ExtractionResult:
        """Extract activations for all conversations in the dataset.

        Args:
            dataset: List of ConversationEntry objects.
            plan: ExtractionPlan defining what/how to extract.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            ExtractionResult with activations organized by spec→label→layer.
        """
        t0 = time.time()
        metadata = ExtractionMetadata(
            plan_name=plan.name,
            model_id=self.model.model_id,
        )
        result = ExtractionResult(plan_name=plan.name, metadata=metadata)

        # Determine which layers to extract (sorted for forward-pass order)
        if plan.needs_all_layers():
            required_layers = list(range(self.model.num_layers))
        else:
            required_layers = sorted(plan.all_required_layers())

        if not required_layers:
            warnings.warn("No layers specified in plan — nothing to extract.")
            return result

        # Warn about any out-of-range layer indices before starting extraction
        invalid_layers = [l for l in required_layers if l >= self.model.num_layers]
        if invalid_layers:
            warnings.warn(
                f"⚠️  The following layer indices exceed the model's layer count "
                f"({self.model.num_layers} layers, valid: 0–{self.model.num_layers - 1}): "
                f"{invalid_layers}. These layers will be silently skipped. "
                f"Check your ExtractionPlan specs."
            )

        valid_required_layers = [l for l in required_layers if l < self.model.num_layers]
        max_required_layer = max(valid_required_layers) if valid_required_layers else -1
        stop_after_layer = 0 <= max_required_layer < self.model.num_layers - 1

        # Check if any spec needs token logits (probability-aware pooling).
        # When True, we run the full forward pass (no tracer.stop()) to get
        # the LM head output, and extract logits alongside activations.
        plan_needs_logits = any(
            s.token_selection.needs_logits for s in plan.specs.values()
        )

        if progress_fn is not None:
            iterator = progress_fn(dataset)
        elif show_progress:
            iterator = tqdm(dataset, desc=f"Extracting '{plan.name}'")
        else:
            iterator = dataset

        for entry in iterator:
            metadata.n_conversations += 1
            success = self._process_conversation(
                entry=entry,
                plan=plan,
                result=result,
                required_layers=required_layers,
                stop_after_layer=stop_after_layer,
                plan_needs_logits=plan_needs_logits,
                metadata=metadata,
            )
            if not success:
                # OOM or other error — continue with next conversation
                continue
            if metadata.n_conversations % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()
        metadata.total_time_s = time.time() - t0
        print(
            f"\n✅ Extraction complete: {metadata.n_annotations_processed} annotations "
            f"from {metadata.n_conversations} conversations "
            f"in {metadata.total_time_s:.1f}s"
        )
        if metadata.n_annotations_skipped > 0:
            print(f"   ⚠️  {metadata.n_annotations_skipped} annotations skipped "
                  f"(empty window after token selection, or mapping failure)")
        return result

    def _process_conversation(
        self,
        entry: ConversationEntry,
        plan: ExtractionPlan,
        result: ExtractionResult,
        required_layers: list[int],
        stop_after_layer: bool,
        plan_needs_logits: bool,
        metadata: ExtractionMetadata,
    ) -> bool:
        """Process a single conversation. Returns False on OOM/fatal error."""
        try:
            # 1. Format messages & find message offsets
            formatted_text, message_offsets = self.model.format_messages_with_offsets(
                entry.messages
            )

            # 2. Filter annotations by label (if plan has a filter)
            annotations = entry.annotations
            if plan.label_filter:
                annotations = [
                    a for a in annotations
                    if any(lbl in plan.label_filter for lbl in a.labels)
                ]

            if not annotations:
                return True

            # 3. Map annotations to token spans via offset_mapping
            token_spans = self._token_mapper.map_annotations_to_tokens(
                entry=entry,
                formatted_text=formatted_text,
                message_offsets=message_offsets,
            )
            # Re-apply label filter (map_annotations_to_tokens uses entry.annotations)
            if plan.label_filter:
                token_spans = [
                    ts for ts in token_spans
                    if any(lbl in plan.label_filter for lbl in ts.labels)
                ]

            if not token_spans:
                return True

            # Warn if annotations target messages beyond the first assistant turn.
            # Multi-turn extraction is not yet supported: most chat templates strip
            # <think> blocks from earlier turns, making extracted activations
            # unrealistic for those positions.
            # TODO: implement per-turn partial formatting for proper multi-turn support.
            first_response_idx = next(
                (i for i, m in enumerate(entry.messages) if m["role"] == "assistant"),
                len(entry.messages) - 1,
            )
            late_spans = [
                ts for ts in token_spans
                if ts.original_span is not None
                and ts.original_span.message_idx > first_response_idx
            ]
            if late_spans:
                warnings.warn(
                    f"'{entry.id}': {len(late_spans)} annotation(s) target messages "
                    f"after the first assistant response (message_idx > {first_response_idx}). "
                    f"Multi-turn extraction is not yet supported — reasoning from past turns "
                    f"is stripped by most chat templates, so activations may be unrealistic. "
                    f"Consider limiting annotations to the first interaction."
                )

            # 4. Tokenize (truncate to max_seq_len)
            encoding = self.model.tokenize(formatted_text, return_offsets_mapping=False)
            token_ids = encoding["input_ids"][0]  # (seq_len,)

            # Truncate to the minimum of max_seq_len and the last annotated token.
            # For causal LMs, activations at position i don't depend on later tokens,
            # so running the forward pass only up to the last annotation is equivalent.
            last_annotated_token = max(ts.token_end for ts in token_spans)
            seq_len = min(len(token_ids), self.max_seq_len, last_annotated_token)
            token_ids = token_ids[:seq_len]

            # 5. Run model forward pass ONCE — collect required layers
            layer_activations: dict[int, torch.Tensor] = {}
            logits_saved = None

            with torch.no_grad():
                with self.model.trace(token_ids) as tracer:
                    # Access layers in forward-pass order (CRITICAL for nnsight)
                    for layer_idx in required_layers:
                        if layer_idx >= self.model.num_layers:
                            break  # required_layers is sorted; all remaining are also out of range
                        # (1, seq_len, hidden_dim) → squeeze batch dim
                        layer_activations[layer_idx] = (
                            self.model.layers_output[layer_idx]
                            .squeeze(0)          # → (seq_len, hidden_dim)
                            .detach()
                            .cpu()
                            .save()
                        )

                    if plan_needs_logits:
                        # Probability-aware pooling needs the LM head output.
                        # This requires a full forward pass — no tracer.stop().
                        # model.st is the StandardizedTransformer; its output is
                        # the HF CausalLM output with a .logits attribute.
                        logits_saved = (
                            self.model.st.output.logits
                            .squeeze(0)          # → (seq_len, vocab_size)
                            .detach()
                            .cpu()
                            .save()
                        )
                    elif stop_after_layer:
                        # Early stopping — skip layers after max_required_layer.
                        # nnsight executes this after all layer saves registered above,
                        # so the forward pass halts immediately after max_required_layer.
                        tracer.stop()

            logits: torch.Tensor | None = logits_saved  # None or (seq_len, vocab_size)

            # 6. For each annotation × spec → slice & store
            for token_span in token_spans:
                # Clamp token span to actual (possibly truncated) seq_len
                if token_span.token_start >= seq_len:
                    metadata.n_annotations_skipped += 1
                    continue

                effective_end = min(token_span.token_end, seq_len)
                effective_start = token_span.token_start

                for spec_name, spec in plan.specs.items():
                    spec_layers = (
                        spec.layers if spec.layers is not None else required_layers
                    )

                    for layer_idx in spec_layers:
                        if layer_idx not in layer_activations:
                            continue

                        layer_act = layer_activations[layer_idx]  # (seq_len, hidden_dim)

                        # Apply TokenSelection (bleed shapes, strategy picks)
                        resolved = spec.token_selection.apply(
                            span_token_start=effective_start,
                            span_token_end=effective_end,
                            sequence_length=seq_len,
                        )

                        if resolved is None:
                            # Empty window after bleed — skip this annotation
                            metadata.n_annotations_skipped += 1
                            continue

                        sel_start, sel_end = resolved
                        selected = layer_act[sel_start:sel_end, :]  # (n_tokens, hidden_dim)

                        if selected.shape[0] == 0:
                            metadata.n_annotations_skipped += 1
                            continue

                        # Aggregate (probability-aware strategies slice logits to match)
                        ts = spec.token_selection
                        sel_logits = (
                            logits[sel_start:sel_end] if logits is not None else None
                        )
                        activation = apply_aggregation(
                            selected,
                            ts.aggregation,
                            token_logits=sel_logits,
                            confidence_threshold=ts.confidence_threshold,
                            high_entropy_fraction=ts.high_entropy_fraction,
                        )

                        # Store for each label
                        for label in token_span.labels:
                            result.add(spec_name, label, layer_idx, activation)
                            metadata.n_annotations_processed += 1

            # Clean up layer activations to free memory
            del layer_activations

        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            warnings.warn(
                f"OOM on conversation '{entry.id}' (seq_len={seq_len}). "
                "Skipping. Consider reducing max_seq_len."
            )
            return False

        except Exception as e:
            warnings.warn(f"Error processing conversation '{entry.id}': {e}")
            return False

        return True
