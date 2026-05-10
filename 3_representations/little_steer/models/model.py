"""
little_steer.models.model

LittleSteerModel: thin wrapper around nnterp's StandardizedTransformer.

Adds:
  - Custom chat template support (for models without HF templates)
  - Reasoning role → model-specific format mapping
  - Convenience methods for formatted text + message offset tracking
  - Integration with VRAMManager
"""

from __future__ import annotations

from typing import Any
import os

import torch
from nnterp import StandardizedTransformer
from nnterp.rename_utils import RenameConfig
from rich.console import Console

from ..data.tokenizer_utils import TokenPositionMapper, _map_reasoning_role

console = Console()


class LittleSteerModel:
    """Model wrapper using nnterp's StandardizedTransformer for model-agnostic access.

    Provides a unified API for:
      - Formatting messages for any model (with chat template)
      - Tracking where each message's content lands in the formatted string
      - Delegating layer output access to nnterp's standardized interface

    Example:
        model = LittleSteerModel("Qwen/Qwen3-8B")
        with model.trace("Hello world") as tr:
            out = model.layers_output[20].save()
            tr.stop()
    """

    def __init__(
        self,
        model_id: str,
        custom_chat_template: str | None = None,
        device_map: str = "auto",
        torch_dtype: Any = "auto",
        use_pretrained_loading: bool = False,
        attn_rename: str | list[str] | None = None,
        mlp_rename: str | list[str] | None = None,
        **hf_kwargs: Any,
    ):
        """
        Args:
            model_id: HuggingFace model ID or local path.
            custom_chat_template: Jinja2 chat template string. Use this if the
                model's tokenizer doesn't have one, or to override it.
            device_map: HF device_map argument (default 'auto').
            torch_dtype: HF dtype argument (default 'auto' = bf16 on GPU).
            use_pretrained_loading: If True, load weights via from_pretrained
                instead of the default meta-device initialization. Use this for
                models whose config is incompatible with nnsight's meta-init
                (e.g. multimodal architectures like Qwen3.5 where vocab_size
                lives in a nested text_config).
            attn_rename: Name(s) of the attention module (e.g. 'linear_attn' for Qwen3.5).
            mlp_rename: Name(s) of the MLP module.
            **hf_kwargs: Extra kwargs forwarded to StandardizedTransformer.
        """
        self.model_id = model_id
        self._custom_chat_template = custom_chat_template

        if not use_pretrained_loading and torch_dtype == "auto":
            # nnsight's meta-init path (from_config) passes torch_dtype directly to
            # HF transformers which calls getattr(torch, dtype) — "auto" is not a
            # torch attribute. Substitute a concrete dtype so the init doesn't crash.
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        if use_pretrained_loading:
            # dispatch=True tells nnsight to use from_pretrained directly,
            # skipping the from_config meta-init that fails on multimodal configs.
            hf_kwargs["dispatch"] = True
            # from_pretrained needs trust_remote_code for custom architectures;
            # unlike from_config it does not set this automatically.
            hf_kwargs.setdefault("trust_remote_code", True)

        # Setup custom renaming if provided
        rename_config = None
        if attn_rename or mlp_rename:
            rename_config = RenameConfig(attn_name=attn_rename, mlp_name=mlp_rename)

        def load_with_impl(impl: str):
            kwargs = hf_kwargs.copy()
            kwargs["attn_implementation"] = impl
            
            # Automatically handle disk offloading if device_map is used
            if device_map is not None and "offload_folder" not in kwargs:
                offload_base = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                offload_dir = os.path.join(offload_base, "offload", model_id.replace("/", "--"))
                os.makedirs(offload_dir, exist_ok=True)
                kwargs["offload_folder"] = offload_dir

            return StandardizedTransformer(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                rename_config=rename_config,
                **kwargs,
            )

        requested_impl = hf_kwargs.get("attn_implementation")
        
        # Determine best default implementation if not specified
        if requested_impl is None:
            # Chain of fallbacks from fastest to most compatible
            impls = ["flash_attention_2", "sdpa", "eager"]
            if not torch.cuda.is_available():
                impls = ["sdpa", "eager"]
            
            self.st = None
            for i, impl in enumerate(impls):
                try:
                    # Only print the "trying" message for the first attempt
                    if i == 0:
                        console.print(f"[bold]Loading {model_id} via nnterp[/bold] (trying {impl})...")
                    
                    self.st = load_with_impl(impl)
                    if i > 0:
                        console.print(f"[green]Loaded with {impl}[/green] (degraded).")
                    break
                except (ImportError, ValueError, RuntimeError) as e:
                    first_line = str(e).splitlines()[0]
                    if i < len(impls) - 1:
                        console.print(
                            f"[yellow]{impl} failed; trying {impls[i+1]}[/yellow] "
                            f"({first_line})"
                        )
                    else:
                        console.print("[red]All attention implementations failed.[/red]")
                        raise e
        else:
            console.print(f"[bold]Loading {model_id} via nnterp[/bold] ({requested_impl})...")
            self.st = load_with_impl(requested_impl)

        self._token_mapper = TokenPositionMapper(self.st.tokenizer)
        console.print(
            f"[green]{model_id} loaded[/green] — "
            f"{self.num_layers} layers, hidden_size={self.hidden_size}"
        )

    # ------------------------------------------------------------------
    # Model properties (delegated to nnterp)
    # ------------------------------------------------------------------

    @property
    def num_layers(self) -> int:
        return self.st.num_layers

    @property
    def hidden_size(self) -> int:
        return self.st.hidden_size

    @property
    def num_heads(self) -> int:
        return self.st.num_heads

    @property
    def vocab_size(self) -> int:
        return self.st.vocab_size

    @property
    def tokenizer(self):
        return self.st.tokenizer

    @property
    def device(self):
        return self.st.device

    # ------------------------------------------------------------------
    # nnterp tracing (delegated)
    # ------------------------------------------------------------------

    def trace(self, *args, **kwargs):
        """Context manager for nnterp tracing. Use like:

            with model.trace(token_ids) as tracer:
                act = model.layers_output[20].save()
                tracer.stop()
        """
        return self.st.trace(*args, **kwargs)

    @property
    def layers_output(self):
        """Standardized layer output accessor (works across all architectures)."""
        return self.st.layers_output

    @property
    def layers(self):
        """Direct access to model.layers for caching."""
        return self.st.layers

    # ------------------------------------------------------------------
    # Message formatting
    # ------------------------------------------------------------------

    def format_messages(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> str:
        """Apply chat template to messages, mapping the 'reasoning' role.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            add_generation_prompt: Whether to add a trailing generation prompt.

        Returns:
            The formatted string ready for tokenization.
        """
        return self._token_mapper.format_messages(
            messages,
            chat_template=self._custom_chat_template,
            add_generation_prompt=add_generation_prompt,
        )

    def format_messages_with_offsets(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
    ) -> tuple[str, dict[int, int]]:
        """Format messages and find where each message's content lands.

        Returns:
            (formatted_text, {message_idx: char_offset_in_formatted_text})
        """
        formatted_text = self.format_messages(messages, add_generation_prompt)
        offsets = self._token_mapper.find_message_offsets(
            messages,
            formatted_text,
            chat_template=self._custom_chat_template,
        )
        return formatted_text, offsets

    def tokenize(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        add_special_tokens: bool = False,
    ) -> dict:
        """Tokenize text and return the encoding dict."""
        return self.tokenizer(
            text,
            return_offsets_mapping=return_offsets_mapping,
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Apply chat template directly (bypasses reasoning role mapping).

        Use format_messages() for the full pipeline including role mapping.
        """
        if self._custom_chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                chat_template=self._custom_chat_template,
                tokenize=False,
                **kwargs,
            )
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)

    def __repr__(self) -> str:
        return (
            f"LittleSteerModel(id={self.model_id!r}, "
            f"layers={self.num_layers}, "
            f"hidden_size={self.hidden_size})"
        )
