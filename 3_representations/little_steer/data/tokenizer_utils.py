"""
little_steer.data.tokenizer_utils

Utilities for converting character-based span positions to token positions
for any HuggingFace tokenizer.

Key challenge: after applying a chat template, each message's content lands
at a specific offset in the final formatted string. We need to track where
each message ends up so we can map char positions → token positions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from thesis_schema import AnnotatedSpan, ConversationEntry


@dataclass
class TokenSpan:
    """A span expressed as token indices in the full formatted sequence."""

    token_start: int
    """First token index of this span (inclusive)."""

    token_end: int
    """One-past-last token index (exclusive)."""

    labels: list[str]
    """Category labels from the  source AnnotatedSpan."""

    score: float = 0.0

    original_span: AnnotatedSpan | None = None
    """Reference back to the source annotation for debugging."""

    @property
    def length(self) -> int:
        return self.token_end - self.token_start

    def __repr__(self) -> str:
        return (
            f"TokenSpan([{self.token_start}:{self.token_end}], "
            f"len={self.length}, labels={self.labels})"
        )


class TokenPositionMapper:
    """Maps character positions in ConversationEntry to token positions.

    Works with any HuggingFace tokenizer by leveraging offset_mapping
    (return_offsets_mapping=True), which gives the char start/end for each
    token in the formatted string — no fragile regex or manual counting.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def format_messages(
        self,
        messages: list[dict[str, str]],
        chat_template: str | None = None,
        add_generation_prompt: bool = False,
    ) -> str:
        """Apply the tokenizer's chat template to messages.

        The "reasoning" role is mapped to model-specific format before
        applying the template (e.g., wrapping in <think> tags for Qwen/DeepSeek).
        """
        mapped = _map_reasoning_role(messages, self.tokenizer)
        kwargs: dict = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        return self.tokenizer.apply_chat_template(mapped, **kwargs)

    def find_message_offsets(
        self,
        messages: list[dict[str, str]],
        formatted_text: str,
        chat_template: str | None = None,
    ) -> dict[int, int]:
        """Find the character offset of each message's content in the formatted string.

        Returns offsets keyed by original message index (including 'reasoning' messages).

        Strategy: incrementally format messages 0..i and use rfind on raw content.
        rfind always finds the LAST occurrence of the content in the partial string,
        which is guaranteed to be the freshly-added message — robust to small-model
        repetitions and independent of how think tokens are formatted (\\n vs space).

        Returns:
            {original_message_idx: char_offset_in_formatted_text}
        """
        offsets: dict[int, int] = {}
        mapped = _map_reasoning_role(messages, self.tokenizer)
        search_map = _build_search_map(messages, mapped)
        # search_map: {mapped_idx: [(orig_idx, raw_content_to_search), ...]}

        kwargs: dict = {"tokenize": False, "add_generation_prompt": False}
        if chat_template is not None:
            kwargs["chat_template"] = chat_template

        for mapped_i, mapped_msg in enumerate(mapped):
            partial = self.tokenizer.apply_chat_template(mapped[: mapped_i + 1], **kwargs)

            for orig_i, search_content in search_map.get(mapped_i, []):
                idx = partial.rfind(search_content)
                if idx == -1:
                    if messages[orig_i]["role"] == "reasoning":
                        warnings.warn(
                            f"Reasoning content for message {orig_i} was not found in the "
                            f"formatted text. The chat template may have stripped it "
                            f"(e.g. DeepSeek-R1 strips <think> blocks from all turns). "
                            f"Annotations targeting this message will be skipped."
                        )
                else:
                    offsets[orig_i] = idx

        return offsets

    def map_annotations_to_tokens(
        self,
        entry: ConversationEntry,
        formatted_text: str,
        message_offsets: dict[int, int],
        chat_template: str | None = None,
    ) -> list[TokenSpan]:
        """Convert all annotations in a ConversationEntry to TokenSpans.

        Uses offset_mapping from the tokenizer for precise char→token mapping.

        Args:
            entry: The conversation entry with annotations.
            formatted_text: The already-formatted string (from format_messages).
            message_offsets: Mapping from message_idx → char offset in formatted_text.
            chat_template: Optional override chat template.

        Returns:
            List of TokenSpan objects (skips any that can't be mapped).
        """
        # Tokenize with offset mapping
        encoding = self.tokenizer(
            formatted_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        offset_mapping = encoding["offset_mapping"][0].tolist()
        # offset_mapping[i] = (char_start, char_end) for token i

        token_spans: list[TokenSpan] = []

        for ann in entry.annotations:
            # Resolve char positions to absolute positions in formatted_text
            abs_start, abs_end = self._resolve_annotation_to_absolute(
                ann, entry, message_offsets
            )
            if abs_start is None or abs_end is None:
                continue

            # Map chars → tokens via offset_mapping
            token_start = self._char_to_token_start(offset_mapping, abs_start)
            token_end = self._char_to_token_end(offset_mapping, abs_end)

            if token_start is None or token_end is None or token_start >= token_end:
                continue

            token_spans.append(
                TokenSpan(
                    token_start=token_start,
                    token_end=token_end,
                    labels=ann.labels,
                    score=ann.score,
                    original_span=ann,
                )
            )

        return token_spans

    def _resolve_annotation_to_absolute(
        self,
        ann: AnnotatedSpan,
        entry: ConversationEntry,
        message_offsets: dict[int, int],
    ) -> tuple[int | None, int | None]:
        """Convert annotation char positions → absolute positions in formatted_text."""
        if ann.message_idx == -1:
            # Absolute position: computed over raw content concatenation.
            # We need to map from "content-only concat" to "formatted_text".
            # Approximation: use message offsets to find relative char within
            # the closest message that contains this absolute position.
            cumulative = 0
            for i, msg in enumerate(entry.messages):
                content_len = len(msg["content"])
                if ann.char_start < cumulative + content_len:
                    # This annotation (partially) lives in message i
                    if i in message_offsets:
                        in_msg_start = ann.char_start - cumulative
                        in_msg_end = ann.char_end - cumulative
                        offset = message_offsets[i]
                        return offset + in_msg_start, offset + in_msg_end
                    break
                cumulative += content_len
            return None, None
        else:
            if ann.message_idx not in message_offsets:
                return None, None
            offset = message_offsets[ann.message_idx]
            return offset + ann.char_start, offset + ann.char_end

    def _char_to_token_start(
        self, offset_mapping: list[tuple[int, int]], char_pos: int
    ) -> int | None:
        """Find the first token whose char range covers char_pos."""
        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start <= char_pos < tok_end:
                return i
            if tok_start > char_pos:
                # char_pos falls between tokens (e.g., whitespace) — return next
                return i
        return None

    def _char_to_token_end(
        self, offset_mapping: list[tuple[int, int]], char_pos: int
    ) -> int | None:
        """Find the first token that starts at or after char_pos (exclusive end)."""
        for i, (tok_start, _tok_end) in enumerate(offset_mapping):
            if tok_start >= char_pos:
                return i
        return len(offset_mapping)  # span extends to end of sequence


def _map_reasoning_role(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict[str, str]]:
    """Map the 'reasoning' role to model-specific format.

    For models with thinking support (Qwen3, DeepSeek-R1, Phi-4, Magistral ...):
      - If the chat template has a native 'reasoning_content' field (e.g. Qwen3),
        passes it as a separate key so the template formats it exactly as trained.
      - Otherwise, manually wraps reasoning in <think>...</think> inside the
        assistant message content.

    For models without thinking support:
      - Omits the reasoning message entirely.
    """
    template = getattr(tokenizer, "chat_template", "") or ""
    has_native_reasoning_field = "reasoning_content" in template

    result: list[dict[str, str]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "reasoning":
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                if _model_supports_thinking(tokenizer):
                    reasoning_text = msg["content"]
                    assistant_text = messages[i + 1]["content"]
                    if has_native_reasoning_field:
                        # Let the template handle formatting (e.g. Qwen3's native
                        # reasoning_content support gives the exact trained format).
                        result.append({
                            "role": "assistant",
                            "content": assistant_text,
                            "reasoning_content": reasoning_text,
                        })
                    else:
                        # Manual wrapping for models without the native field.
                        result.append({
                            "role": "assistant",
                            "content": f"<think>\n{reasoning_text}\n</think>\n\n{assistant_text}",
                        })
                    i += 2
                    continue
                else:
                    i += 1  # omit reasoning entirely
                    continue
            else:
                i += 1  # standalone reasoning with no following assistant — omit
                continue
        else:
            result.append(msg)
        i += 1

    return result


def _build_search_map(
    messages: list[dict[str, str]],
    mapped: list[dict[str, str]],
) -> dict[int, list[tuple[int, str]]]:
    """Build a lookup from mapped message index → original messages + their raw content.

    For each mapped message, returns the list of (original_idx, raw_content) pairs
    where raw_content is the actual text we search for in the partial formatted string.
    Using the raw original content (not the merged think block) makes the search
    independent of think-token whitespace conventions across models.

    Returns:
        {mapped_idx: [(orig_idx, raw_content_to_search_for), ...]}
    """
    result: dict[int, list[tuple[int, str]]] = {}
    orig_i = 0
    mapped_i = 0

    while orig_i < len(messages):
        msg = messages[orig_i]
        if msg["role"] == "reasoning":
            if (
                orig_i + 1 < len(messages)
                and messages[orig_i + 1]["role"] == "assistant"
                and mapped_i < len(mapped)
            ):
                # Both original messages contribute to one mapped message.
                # Search for each one's raw content independently.
                result[mapped_i] = [
                    (orig_i, msg["content"]),                        # reasoning text
                    (orig_i + 1, messages[orig_i + 1]["content"]),  # assistant text
                ]
                orig_i += 2
                mapped_i += 1
                continue
            else:
                orig_i += 1  # omitted — skip without advancing mapped_i
                continue
        else:
            result[mapped_i] = [(orig_i, msg["content"])]
            orig_i += 1
            mapped_i += 1

    return result


def _model_supports_thinking(tokenizer: PreTrainedTokenizerBase) -> bool:
    """Heuristic: check if the tokenizer/model supports <think> tokens."""
    # Check for known thinking model identifiers
    model_name = getattr(tokenizer, "name_or_path", "") or ""
    model_name_lower = model_name.lower()

    thinking_indicators = ["qwen3", "deepseek-r1", "qwq", "deepseek-reasoner", "phi-4", "ministral", "magistral", "gemma-4"]
    if any(ind in model_name_lower for ind in thinking_indicators):
        return True

    # Check if <think> is in the vocabulary
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
        if "<think>" in vocab or "▁<think>" in vocab:
            return True

    # Check chat template for thinking keywords
    template = getattr(tokenizer, "chat_template", "") or ""
    if "<think>" in template or "enable_thinking" in template:
        return True

    return False
