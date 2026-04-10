#!/usr/bin/env python3
"""
generate_responses.py — Robust LLM response generation pipeline.

Outputs data in little_steer's ConversationEntry/JSONL format, ready for
annotation and activation extraction.

CLI usage:
    python generate_responses.py --config config.yaml
    python generate_responses.py --config config.yaml --models qwen3-8b --datasets harmful
    python generate_responses.py --config config.yaml --show-sample

Notebook usage:
    from generate_responses import ResponseGenerator
    gen = ResponseGenerator.from_config("config.yaml")
    gen.run(models=["qwen3-8b"])
    gen.run()  # all models & datasets
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import requests
import signal
import socket
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import os
import sys

# ── Ultra-Silence vLLM/Ray/HF ────────────────────────────────────────────────
os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["RAY_loglevel"] = "error"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
import warnings
from contextlib import contextmanager

# Standard logging silencers
for logger_name in ["vllm", "ray", "transformers", "datasets"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

@contextmanager
def silence_all(enabled: bool = True):
    """Aggressive low-level silence that catches subprocess output."""
    if not enabled:
        yield
        return

    # Save the original file descriptors
    null_fd = os.open(os.devnull, os.O_RDWR)
    old_stdout = os.dup(sys.stdout.fileno())
    old_stderr = os.dup(sys.stderr.fileno())
    
    try:
        # Redirect stdout and stderr at the FD level
        os.dup2(null_fd, sys.stdout.fileno())
        os.dup2(null_fd, sys.stderr.fileno())
        yield
    finally:
        # Restore original FDs
        os.dup2(old_stdout, sys.stdout.fileno())
        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(null_fd)
# ─────────────────────────────────────────────────────────────────────────────

# ── Graceful shutdown ─────────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_sigint(sig, frame) -> None:
    global _shutdown_requested
    if _shutdown_requested:
        print("\nForced exit.", flush=True)
        sys.exit(1)
    _shutdown_requested = True
    print("\n[Ctrl+C] Finishing current chunk then stopping cleanly...", flush=True)


try:
    import torch as _torch
    def _cuda_empty_cache():
        _torch.cuda.empty_cache()
    def _cuda_is_available():
        return _torch.cuda.is_available()
except ImportError:
    def _cuda_empty_cache(): pass
    def _cuda_is_available(): return False
import yaml
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from thesis_schema import ConversationEntry  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config models (Pydantic)
# ─────────────────────────────────────────────────────────────────────────────

QuantizationOption = Literal["4bit", "8bit", "auto", "none"] | None
# "auto" = try float16 first; fall back to bitsandbytes if OOM during load
# "none" = explicitly disable quantization (overrides the default; use for GGUF models)


class ModelDefaults(BaseModel):
    """Default generation parameters — inherited by all models unless overridden."""
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int | None = None
    min_p: float | None = None
    max_new_tokens: int = 2300
    quantization: QuantizationOption = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    # vLLM-specific
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None   # cap context window to save KV-cache memory
    enforce_eager: bool = False        # disable CUDA graphs (slower but less memory)


class ModelConfig(BaseModel):
    """Per-model config. Any omitted field falls back to ModelDefaults."""
    name: str
    model_id: str
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_new_tokens: int | None = None
    quantization: QuantizationOption = None
    system_prompt: str | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    # vLLM-specific
    gpu_memory_utilization: float | None = None
    max_model_len: int | None = None
    enforce_eager: bool | None = None
    tokenizer: str | None = None  # override tokenizer (e.g. use base model's tokenizer)
    hf_config_path: str | None = None  # HF repo to fetch config.json from (needed for GGUF repos without config.json)
    skip_datasets: list[str] = Field(default_factory=list)  # dataset names to skip for this model
    mistral_format: bool = False  # use mistral tokenizer/config/load format (for Mistral reasoning models)
    # OpenAI-compatible backend (e.g. LMStudio, Ollama, any OpenAI-compatible server)
    backend: Literal["vllm", "openai"] = "vllm"
    openai_base_url: str | None = None  # e.g. "http://localhost:1234/v1" for LMStudio
    openai_api_key: str | None = None   # most local servers accept any non-empty string

    def resolve(self, defaults: ModelDefaults) -> ResolvedModelConfig:
        def pick(val: Any, default: Any) -> Any:
            return val if val is not None else default

        return ResolvedModelConfig(
            name=self.name,
            model_id=self.model_id,
            temperature=pick(self.temperature, defaults.temperature),
            top_p=pick(self.top_p, defaults.top_p),
            top_k=pick(self.top_k, defaults.top_k),
            min_p=pick(self.min_p, defaults.min_p),
            max_new_tokens=pick(self.max_new_tokens, defaults.max_new_tokens),
            quantization=pick(self.quantization, defaults.quantization),
            system_prompt=self.system_prompt,
            presence_penalty=pick(self.presence_penalty, defaults.presence_penalty),
            repetition_penalty=pick(self.repetition_penalty, defaults.repetition_penalty),
            gpu_memory_utilization=pick(self.gpu_memory_utilization, defaults.gpu_memory_utilization),
            max_model_len=pick(self.max_model_len, defaults.max_model_len),
            enforce_eager=pick(self.enforce_eager, defaults.enforce_eager),
            tokenizer=self.tokenizer,
            hf_config_path=self.hf_config_path,
            skip_datasets=self.skip_datasets,
            mistral_format=self.mistral_format,
            backend=self.backend,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
        )


class ResolvedModelConfig(BaseModel):
    """Fully-resolved model config after merging with defaults."""
    name: str
    model_id: str
    temperature: float
    top_p: float
    top_k: int | None
    min_p: float | None
    max_new_tokens: int
    quantization: QuantizationOption
    system_prompt: str | None
    presence_penalty: float | None
    repetition_penalty: float | None
    # vLLM-specific
    gpu_memory_utilization: float
    max_model_len: int | None
    enforce_eager: bool
    tokenizer: str | None = None  # override tokenizer (e.g. use base model's tokenizer)
    hf_config_path: str | None = None  # HF repo to fetch config.json from (needed for GGUF repos without config.json)
    skip_datasets: list[str] = Field(default_factory=list)
    mistral_format: bool = False
    # OpenAI-compatible backend
    backend: Literal["vllm", "openai"] = "vllm"
    openai_base_url: str | None = None
    openai_api_key: str | None = None


class DatasetConfig(BaseModel):
    """Dataset source config."""
    name: str
    source: Literal["hf", "local"]
    path: str
    subset: str | None = None
    split: str = "train"
    prompt_field: str = "prompt"
    prompt_index: int | None = None
    filter_max_list_len: int | None = None
    max_samples: int = 500
    extra_fields: list[str] | Literal["*"] | None = None
    # extra_fields controls what dataset row metadata to save alongside the prompt:
    #   null / omitted — save nothing extra
    #   "*"            — save all fields from the row except the prompt field
    #   ["f1", "f2"]   — save only the listed fields


class GenerationConfig(BaseModel):
    # How many prompts to process before writing results to disk.
    # vLLM handles its own internal batching — this only controls checkpoint frequency.
    write_chunk_size: int = 20
    max_retries: int = 3
    run_mode: Literal["sequential", "interleaved"] = "sequential"


class GeneratorConfig(BaseModel):
    output_dir: str = "data"
    hf_token_file: str | None = None
    defaults: ModelDefaults = Field(default_factory=ModelDefaults)
    models: list[ModelConfig]
    datasets: list[DatasetConfig]
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> GeneratorConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# ─────────────────────────────────────────────────────────────────────────────
# Loaded model bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    """vLLM LLM instance with metadata."""
    llm: Any                               # vllm.LLM
    thinking_supported: bool = False       # True if tokenizer supports enable_thinking=True
    actual_quantization: str | None = None
    actual_max_model_len: int | None = None  # KV-cache budget vLLM actually allocated
    actual_enforce_eager: bool = False


@dataclass
class LoadedOpenAIModel:
    """OpenAI-compatible client (e.g. LMStudio, Ollama)."""
    client: Any   # openai.OpenAI
    model_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _map_quantization(q: QuantizationOption) -> str | None:
    """Map our quantization option to vLLM's quantization string."""
    if q in (None, "auto", "none"):
        return None
    if q in ("4bit", "8bit"):
        return "bitsandbytes"
    return None


def _detect_thinking_support(llm: Any) -> bool:
    """Check whether the model's tokenizer supports enable_thinking=True."""
    try:
        tokenizer = llm.get_tokenizer()
        test_msgs = [{"role": "user", "content": "test"}]
        tokenizer.apply_chat_template(
            test_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        return True
    except Exception:
        return False


def _try_load_vllm(
    model_id: str,
    quantization: str | None,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    enforce_eager: bool,
    tokenizer: str | None = None,
    hf_config_path: str | None = None,
    mistral_format: bool = False,
    verbose: bool = False,
) -> Any | None:
    """Attempt to load a vLLM model. Returns None on any failure."""
    from vllm import LLM

    # BitsAndBytes quantization in vLLM requires CUDA graphs disabled.
    actual_enforce_eager = enforce_eager or (quantization == "bitsandbytes")

    kwargs: dict[str, Any] = {
        "model": model_id,
        "dtype": "auto",
        "gpu_memory_utilization": gpu_memory_utilization,
        "trust_remote_code": True,
        "max_num_seqs": 256,
        "enforce_eager": actual_enforce_eager,
        "disable_log_stats": True,
    }
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len
    if quantization is not None:
        kwargs["quantization"] = quantization
        kwargs["load_format"] = "bitsandbytes"
    if tokenizer is not None:
        kwargs["tokenizer"] = tokenizer
    if hf_config_path is not None:
        kwargs["hf_config_path"] = hf_config_path
    if mistral_format:
        kwargs["tokenizer_mode"] = "mistral"  # use mistral_common for tokenization

    try:
        with silence_all(enabled=not verbose):
            return LLM(**kwargs)
    except Exception as e:
        # Keep it clean: only the first sentence of the error, or a brief OOM message.
        full_msg = str(e)
        if "OutOfMemoryError" in full_msg or "out of memory" in full_msg.lower():
            msg = "CUDA out of memory — model too large for available VRAM."
        elif "Engine core initialization failed" in full_msg:
            msg = "Engine core initialization failed (likely OOM or incompatible architecture)."
        else:
            msg = full_msg.split(".")[0].split("\n")[0]
        
        tqdm.write(f"  Load failed: {msg}")
        gc.collect()
        _cuda_empty_cache()
        return None


def _extract_llm_metadata(llm: Any) -> dict[str, Any]:
    """Pull the settings vLLM actually used so we can log them."""
    try:
        import vllm
        vllm_version = vllm.__version__
    except Exception:
        vllm_version = "unknown"

    actual_max_model_len: int | None = None
    try:
        actual_max_model_len = llm.llm_engine.model_config.max_model_len
    except Exception:
        pass

    actual_enforce_eager: bool = False
    try:
        actual_enforce_eager = llm.llm_engine.model_config.enforce_eager
    except Exception:
        pass

    return {
        "vllm_version": vllm_version,
        "actual_max_model_len": actual_max_model_len,
        "actual_enforce_eager": actual_enforce_eager,
    }


def load_vllm_model(
    model_cfg: ResolvedModelConfig,
    hf_token: str | None,
    verbose: bool = False,
) -> LoadedModel:
    """Load a model using vLLM.

    quantization behaviour:
      null   — load as float16
      "4bit" — bitsandbytes 4-bit
      "8bit" — bitsandbytes 8-bit
      "auto" — try float16 first; fall back to bitsandbytes if it doesn't fit
    """
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    tqdm.write(f"  Loading model: {model_cfg.model_id}")

    if model_cfg.quantization != "auto":
        quant = _map_quantization(model_cfg.quantization)
        llm = _try_load_vllm(
            model_cfg.model_id, quant,
            model_cfg.gpu_memory_utilization, model_cfg.max_model_len, model_cfg.enforce_eager,
            tokenizer=model_cfg.tokenizer, hf_config_path=model_cfg.hf_config_path,
            mistral_format=model_cfg.mistral_format, verbose=verbose,
        )
        if llm is None:
            raise RuntimeError(f"Failed to load {model_cfg.model_id} with {quant or 'float16'}")
        meta = _extract_llm_metadata(llm)
        thinking = _detect_thinking_support(llm)
        return LoadedModel(llm, thinking, quant or "float16",
                           meta["actual_max_model_len"], meta["actual_enforce_eager"])

    # Auto mode: try float16 first, then bitsandbytes
    for quant in [None, "bitsandbytes"]:
        label = quant or "float16"
        tqdm.write(f"  Auto → trying {label}")
        llm = _try_load_vllm(
            model_cfg.model_id, quant,
            model_cfg.gpu_memory_utilization, model_cfg.max_model_len, model_cfg.enforce_eager,
            tokenizer=model_cfg.tokenizer, hf_config_path=model_cfg.hf_config_path,
            mistral_format=model_cfg.mistral_format, verbose=verbose,
        )
        if llm is not None:
            meta = _extract_llm_metadata(llm)
            thinking = _detect_thinking_support(llm)
            return LoadedModel(llm, thinking, label,
                               meta["actual_max_model_len"], meta["actual_enforce_eager"])

    raise RuntimeError(
        f"Could not load {model_cfg.model_id}: failed even with bitsandbytes quantization."
    )


def load_openai_model(model_cfg: ResolvedModelConfig) -> LoadedOpenAIModel:
    """Create an OpenAI-compatible client (e.g. for LMStudio or Ollama).

    No model weights are loaded locally — inference is handled by the remote server.
    """
    from openai import OpenAI

    base_url = model_cfg.openai_base_url or "http://localhost:1234/v1"
    api_key = model_cfg.openai_api_key or "lm-studio"

    client = OpenAI(base_url=base_url, api_key=api_key)
    tqdm.write(f"  Connected to OpenAI-compatible API at: {base_url}")
    tqdm.write(f"  Model: {model_cfg.model_id}")
    return LoadedOpenAIModel(client=client, model_id=model_cfg.model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _extract_prompt_text(value: Any, index: int | None = None) -> str:
    """Extract a string from a prompt field value.

    Handles:
    1. Direct strings
    2. Lists of strings (takes index)
    3. Lists of dicts (takes index, then extracts 'value' or 'content')
    """
    if isinstance(value, list):
        idx = index if index is not None else 0
        if 0 <= idx < len(value):
            item = value[idx]
            if isinstance(item, dict):
                # Common keys in HF datasets (LlamaFactory, ShareGPT, etc.)
                return str(item.get("value") or item.get("content") or item)
            return str(item)
        return ""
    if isinstance(value, dict):
        return str(value.get("value") or value.get("content") or value)
    return str(value)


def _extract_extra(row: dict, prompt_field: str, extra_fields: list[str] | Literal["*"] | None) -> dict:
    """Extract extra dataset metadata from a row dict."""
    if extra_fields is None:
        return {}
    if extra_fields == "*":
        return {k: v for k, v in row.items() if k != prompt_field}
    return {k: row[k] for k in extra_fields if k in row}


def load_prompts(
    dataset_cfg: DatasetConfig,
    hf_token: str | None,
) -> list[tuple[str, dict]]:
    """Load prompts from HuggingFace Hub or a local JSONL/CSV file.

    Returns a list of (prompt_text, extra_metadata) tuples.
    extra_metadata contains the additional dataset row fields configured via
    extra_fields (empty dict if extra_fields is null).
    """
    if dataset_cfg.source == "hf":
        from datasets import load_dataset as hf_load
        kwargs: dict[str, Any] = {"token": hf_token}
        if dataset_cfg.subset:
            kwargs["name"] = dataset_cfg.subset
        tqdm.write(f"  Downloading: {dataset_cfg.path} (split={dataset_cfg.split})")
        ds = hf_load(dataset_cfg.path, split=dataset_cfg.split, **kwargs)

        # Apply filter if set
        if dataset_cfg.filter_max_list_len is not None:
            ds = ds.filter(lambda row: (dataset_cfg.prompt_field in row) and
                           isinstance(row[dataset_cfg.prompt_field], list) and
                           len(row[dataset_cfg.prompt_field]) <= dataset_cfg.filter_max_list_len)

        results = [
            (
                _extract_prompt_text(row[dataset_cfg.prompt_field], dataset_cfg.prompt_index),
                _extract_extra(dict(row), dataset_cfg.prompt_field, dataset_cfg.extra_fields),
            )
            for row in ds
        ]
        results = results[:dataset_cfg.max_samples]
        tqdm.write(f"    Loaded {len(results)} prompts.")
        return results

    if dataset_cfg.source == "local":
        path = Path(dataset_cfg.path)
        if path.suffix == ".jsonl":
            results = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        val = row.get(dataset_cfg.prompt_field)
                        if dataset_cfg.filter_max_list_len is not None:
                            if not (isinstance(val, list) and len(val) <= dataset_cfg.filter_max_list_len):
                                continue
                        results.append((
                            _extract_prompt_text(val, dataset_cfg.prompt_index),
                            _extract_extra(row, dataset_cfg.prompt_field, dataset_cfg.extra_fields),
                        ))
            results = results[:dataset_cfg.max_samples]
            tqdm.write(f"    Loaded {len(results)} prompts.")
            return results
        if path.suffix == ".csv":
            import csv
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))

            results = []
            for row in rows:
                val = row.get(dataset_cfg.prompt_field)
                if dataset_cfg.filter_max_list_len is not None:
                    if not (isinstance(val, list) and len(val) <= dataset_cfg.filter_max_list_len):
                        continue
                results.append((
                    _extract_prompt_text(val, dataset_cfg.prompt_index),
                    _extract_extra(row, dataset_cfg.prompt_field, dataset_cfg.extra_fields),
                ))
            results = results[:dataset_cfg.max_samples]
            tqdm.write(f"    Loaded {len(results)} prompts.")
            return results
        raise ValueError(f"Unsupported local file format: {path.suffix} (use .jsonl or .csv)")

    raise ValueError(f"Unknown dataset source: {dataset_cfg.source!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_prompt_id(dataset_name: str, prompt: str) -> str:
    """Stable 16-char MD5 ID for a (dataset, prompt) pair.

    Consistent across models and machines — use this to match responses
    for the same prompt across different runs.
    """
    return hashlib.md5(f"{dataset_name}::{prompt}".encode()).hexdigest()[:16]


def _make_messages(prompt: str, model_cfg: ResolvedModelConfig) -> list[dict]:
    """Build a chat messages list for a single prompt."""
    messages = []
    if model_cfg.system_prompt:
        messages.append({"role": "system", "content": model_cfg.system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _parse_response(raw_text: str) -> tuple[str | None, str]:
    """Split raw generation text into (reasoning, final_response).

    Handles four formats:
      1. <think>...</think> final answer       (DeepSeek-R1)
      2. ...thinking...\n</think>\n final      (Qwen3 — opening tag is in input, not output)
      3. [THINK]...[/THINK] final answer       (Mistral reasoning models)
      4. ...thinking...\n[/THINK]\n final      (Mistral with opening tag in input)

    Returns:
        reasoning: The thinking block content, or None if no closing tag found.
        response: Everything after the closing tag, or the full text if no thinking.
    """
    # Case 1: explicit <think>...</think> wrapper
    match = re.search(r"<think>(.*?)</think>(.*)", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Case 2: </think> present but no opening tag
    match = re.search(r"(.*?)</think>(.*)", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Case 3: explicit [THINK]...[/THINK] wrapper (Mistral)
    match = re.search(r"\[THINK\](.*?)\[/THINK\](.*)", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    # Case 4: [/THINK] present but no opening tag (Mistral with primed input)
    match = re.search(r"(.*?)\[/THINK\](.*)", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return None, raw_text.strip()


def _finish_reason(n_new_tokens: int, max_new_tokens: int, raw_text: str) -> str:
    """Classify why generation stopped.

    Returns one of:
        "eos"                  — model produced an EOS token naturally
        "max_length"           — hit the max_new_tokens limit (response may be cut)
        "incomplete_thinking"  — thinking block opened but never closed
    """
    if n_new_tokens >= max_new_tokens:
        return "max_length"
    if "<think>" in raw_text and "</think>" not in raw_text:
        return "incomplete_thinking"
    if "[THINK]" in raw_text and "[/THINK]" not in raw_text:
        return "incomplete_thinking"
    return "eos"


def _generate_batch_vllm(
    loaded: LoadedModel,
    prompts: list[str],
    model_cfg: ResolvedModelConfig,
    max_retries: int = 3,
    pbar: tqdm | None = None,
) -> list[dict]:
    """Generate responses for a list of prompts using vLLM.

    vLLM handles its own internal batching with PagedAttention —
    all prompts are passed at once and scheduled continuously.

    Returns a list of dicts (one per prompt) with keys:
        reasoning     — str | None   (content of <think>...</think>)
        response      — str          (text after </think>, or full text)
        finish_reason — str          ("eos", "max_length", "incomplete_thinking", "failed")
        n_new_tokens  — int
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        # vLLM uses -1 for "disabled", not None
        top_k=model_cfg.top_k if model_cfg.top_k is not None else -1,
        min_p=model_cfg.min_p if model_cfg.min_p is not None else 0.0,
        max_tokens=model_cfg.max_new_tokens,
        presence_penalty=model_cfg.presence_penalty if model_cfg.presence_penalty is not None else 0.0,
        repetition_penalty=model_cfg.repetition_penalty if model_cfg.repetition_penalty is not None else 1.0,
    )
    chat_template_kwargs = {"enable_thinking": True} if loaded.thinking_supported else {}
    conversations = [_make_messages(p, model_cfg) for p in prompts]

    results: list[dict | None] = [None] * len(prompts)
    retry_indices = list(range(len(prompts)))

    for attempt in range(max_retries + 1):
        if not retry_indices:
            break
        if attempt > 0:
            tqdm.write(f"  Retry {attempt}/{max_retries} for {len(retry_indices)} incomplete response(s)")

        batch_conversations = [conversations[i] for i in retry_indices]
        outputs = loaded.llm.chat(
            batch_conversations, sampling_params,
            chat_template_kwargs=chat_template_kwargs,
        )

        still_retrying = []
        for local_i, global_i in enumerate(retry_indices):
            out = outputs[local_i].outputs[0]
            raw_text = out.text
            n_new = len(out.token_ids)

            # Map vLLM finish_reason ("stop" | "length") to our internal format
            if out.finish_reason == "length":
                reason = "max_length"
            elif ("<think>" in raw_text and "</think>" not in raw_text) or \
                 ("[THINK]" in raw_text and "[/THINK]" not in raw_text):
                reason = "incomplete_thinking"
            else:
                reason = "eos"

            is_complete = reason != "incomplete_thinking" or attempt == max_retries
            if is_complete:
                reasoning, response = _parse_response(raw_text)
                results[global_i] = {
                    "reasoning": reasoning,
                    "response": response,
                    "finish_reason": reason,
                    "n_new_tokens": n_new,
                }
            else:
                still_retrying.append(global_i)

        retry_indices = still_retrying

    # Safety net
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "reasoning": None,
                "response": "",
                "finish_reason": "failed",
                "n_new_tokens": 0,
            }

    return results  # type: ignore[return-value]


def _generate_batch_openai(
    loaded: LoadedOpenAIModel,
    prompts: list[str],
    model_cfg: ResolvedModelConfig,
    max_retries: int = 3,
    pbar: tqdm | None = None,
    on_complete: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """Generate responses using an OpenAI-compatible API (e.g. LMStudio, Ollama).

    Calls are parallelised with a thread pool since the OpenAI SDK is synchronous.

    LMStudio (and some other servers) return the model's reasoning chain in a
    dedicated ``message.reasoning`` field rather than embedding it inside
    ``message.content``.  We read that field first; if it is absent we fall
    back to parsing <think>...</think> tags from the content as before.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def call_one(prompt: str) -> dict:
        messages = _make_messages(prompt, model_cfg)
        kwargs: dict[str, Any] = {
            "model": loaded.model_id,
            "messages": messages,
            "temperature": model_cfg.temperature,
            "top_p": model_cfg.top_p,
            "max_tokens": model_cfg.max_new_tokens,
        }
        if model_cfg.presence_penalty is not None:
            kwargs["presence_penalty"] = model_cfg.presence_penalty

        for attempt in range(max_retries + 1):
            try:
                resp = loaded.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                raw_text = msg.content or ""
                n_new = resp.usage.completion_tokens if resp.usage else 0
                api_finish = resp.choices[0].finish_reason

                if api_finish == "length":
                    reason = "max_length"
                elif ("<think>" in raw_text and "</think>" not in raw_text) or \
                     ("[THINK]" in raw_text and "[/THINK]" not in raw_text):
                    reason = "incomplete_thinking"
                else:
                    reason = "eos"

                if reason == "incomplete_thinking" and attempt < max_retries:
                    continue  # retry this prompt

                # ── Reasoning extraction ────────────────────────────────────
                # Prefer the dedicated reasoning field (LMStudio uses `reasoning_content` or `reasoning`)
                # over parsing think-tags from content.
                api_reasoning: str | None = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                if api_reasoning:
                    # Reasoning came as a separate field; content is already clean.
                    reasoning: str | None = api_reasoning.strip() or None
                    response = raw_text.strip()
                else:
                    # Fall back to parsing <think>/<THINK> tags from content.
                    reasoning, response = _parse_response(raw_text)

                if pbar is not None:
                    pbar.update(1)

                return {
                    "reasoning": reasoning,
                    "response": response,
                    "finish_reason": reason,
                    "n_new_tokens": n_new,
                }
            except Exception as e:
                if attempt == max_retries:
                    tqdm.write(f"  OpenAI API call failed after {max_retries + 1} attempts: {e}")
                    if pbar is not None:
                        pbar.update(1)
                    return {"reasoning": None, "response": "", "finish_reason": "failed", "n_new_tokens": 0}
                # Transient errors (e.g. LMStudio briefly unloading mid-run): back off before retrying
                import time
                wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, ...
                tqdm.write(f"  API error (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait}s: {e}")
                time.sleep(wait)

        if pbar is not None:
            pbar.update(1)
        return {"reasoning": None, "response": "", "finish_reason": "failed", "n_new_tokens": 0}

    max_workers = min(8, max(1, len(prompts)))
    results_map: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_one, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(futures):
            idx = futures[fut]
            result = fut.result()
            results_map[idx] = result
            if on_complete is not None:
                on_complete(idx, result)
            if _shutdown_requested:
                # Cancel futures that haven't started yet; running ones will finish.
                for f in futures:
                    if not f.done():
                        f.cancel()
                break
    return [results_map.get(i) for i in range(len(prompts))]


def generate_batch(
    loaded: LoadedModel | LoadedOpenAIModel,
    prompts: list[str],
    model_cfg: ResolvedModelConfig,
    max_retries: int = 3,
    pbar: tqdm | None = None,
    on_complete: Callable[[int, dict], None] | None = None,
) -> list[dict]:
    """Dispatch to the appropriate backend (vLLM or OpenAI-compatible)."""
    if isinstance(loaded, LoadedOpenAIModel):
        return _generate_batch_openai(loaded, prompts, model_cfg, max_retries, pbar, on_complete)
    return _generate_batch_vllm(loaded, prompts, model_cfg, max_retries, pbar)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_existing_entries(output_path: Path) -> list[ConversationEntry]:
    """Load all ConversationEntry objects from a JSONL file."""
    if not output_path.exists():
        return []
    entries = []
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(ConversationEntry.model_validate_json(line))
                except Exception:
                    pass
    return entries


def rewrite_entries(output_path: Path, entries: list[ConversationEntry]) -> None:
    """Atomically overwrite a JSONL file with the given entries."""
    tmp = output_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(e.model_dump_json() + "\n")
    tmp.replace(output_path)


def get_existing_ids(output_path: Path) -> set[str]:
    """Return IDs of entries already written to a JSONL output file."""
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


MAX_REASONING_FIX_ATTEMPTS = 3


def _entry_needs_fix(data: dict) -> bool:
    """True if an entry is missing reasoning OR has an empty assistant response."""
    messages = data.get("messages", [])
    missing_reasoning = not any(m.get("role") == "reasoning" for m in messages)
    empty_response = not any(m.get("role") == "assistant" and m.get("content") for m in messages)
    return missing_reasoning or empty_response


def has_entries_missing_reasoning(output_path: Path) -> bool:
    """Quick scan: True if any entry needs fixing (missing reasoning or empty response)."""
    if not output_path.exists():
        return False
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if _entry_needs_fix(data):
                    attempts = data.get("metadata", {}).get("reasoning_fix_attempts", 0)
                    if attempts < MAX_REASONING_FIX_ATTEMPTS:
                        return True
            except (json.JSONDecodeError, AttributeError):
                pass
    return False


def append_entry(entry: ConversationEntry, output_path: Path) -> None:
    """Append one ConversationEntry as a JSONL line (streaming, no buffering)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
        f.flush()
        os.fsync(f.fileno())


# ─────────────────────────────────────────────────────────────────────────────
# ResponseGenerator
# ─────────────────────────────────────────────────────────────────────────────

class ResponseGenerator:
    """Main pipeline. Load from a config file, then call run().

    Example:
        gen = ResponseGenerator.from_config("config.yaml")
        gen.run()                              # all models & datasets
        gen.run(models=["qwen3-8b"])           # one model, all datasets
        gen.run(datasets=["harmful"])          # all models, one dataset
    """

    def __init__(self, config: GeneratorConfig, config_path: Path | None = None):
        self.config = config
        self._config_path = config_path
        self._output_dir = (
            (config_path.parent / config.output_dir).resolve()
            if config_path is not None
            else Path(config.output_dir)
        )
        self._hf_token = self._load_hf_token()
        self._prompt_cache: dict[str, list[tuple[str, dict]]] = {}

    def _notify_ntfy(self, title: str, message: str, topic: str = "eldoprano_master"):
        """Send a notification via ntfy.sh."""
        try:
            requests.post(
                f"https://ntfy.sh/{topic}",
                data=message.encode("utf-8"),
                headers={"Title": title.encode("utf-8").decode("latin-1", errors="replace")}
            )
        except Exception as e:
            tqdm.write(f"  [NOTIFICATION FAILED] {e}")

    def _get_prompts(self, dataset_cfg: DatasetConfig) -> list[tuple[str, dict]]:
        """Get prompts for a dataset (memoized in memory)."""
        if dataset_cfg.name not in self._prompt_cache:
            prompts = load_prompts(dataset_cfg, self._hf_token)
            self._prompt_cache[dataset_cfg.name] = prompts
            tqdm.write(f"    Loaded {len(prompts)} prompts.")
        return self._prompt_cache[dataset_cfg.name]


    @classmethod
    def from_config(cls, path: str | Path) -> ResponseGenerator:
        path = Path(path)
        return cls(GeneratorConfig.from_yaml(path), config_path=path)

    # ── HF token ─────────────────────────────────────────────────────────────

    def _load_hf_token(self) -> str | None:
        # Check environment variable first (Standard for Kaggle/Cloud)
        env_token = os.environ.get("HF_TOKEN")
        if env_token:
            return env_token

        if not self.config.hf_token_file:
            return None
        
        p = Path(self.config.hf_token_file)
        if self._config_path is not None:
            p = (self._config_path.parent / self.config.hf_token_file).resolve()
        
        if not p.exists():
            return None
            
        with open(p) as f:
            data = json.load(f)
        return data.get("hf_token") or data.get("token")

    # ── public API ───────────────────────────────────────────────────────────

    def run(
        self,
        models: list[str] | None = None,
        datasets: list[str] | None = None,
        verbose: bool = False,
    ) -> None:
        """Run the generation pipeline."""
        run_id = str(uuid.uuid4())
        machine_id = socket.gethostname()

        model_cfgs = [
            m.resolve(self.config.defaults) for m in self.config.models
            if models is None or m.name in models
        ]
        dataset_cfgs = [
            d for d in self.config.datasets
            if datasets is None or d.name in datasets
        ]

        if not model_cfgs:
            print("No models matched — nothing to do.")
            return
        if not dataset_cfgs:
            print("No datasets matched — nothing to do.")
            return

        print(f"\nRun ID     : {run_id}")
        print(f"Machine    : {machine_id}")
        print(f"Models     : {[m.name for m in model_cfgs]}")
        print(f"Datasets   : {[d.name for d in dataset_cfgs]}")
        print(f"Mode       : {self.config.generation.run_mode}")
        print(f"Chunk size : {self.config.generation.write_chunk_size}")
        print(f"Output     : {self._output_dir}\n")

        failed: list[str] = []
        model_pbar = tqdm(model_cfgs, desc="models", unit="model", position=0)
        for model_cfg in model_pbar:
            if _shutdown_requested:
                break
            model_pbar.set_description(f"model: {model_cfg.name}")

            # ── Check if any work is remaining before loading the model ──
            active_datasets = [d for d in dataset_cfgs if d.name not in model_cfg.skip_datasets]
            has_work = False
            for d_cfg in active_datasets:
                output_path = self._output_dir / f"{model_cfg.name}_{d_cfg.name}.jsonl"
                all_items = self._get_prompts(d_cfg)
                existing_ids = get_existing_ids(output_path)
                remaining = [
                    p for p, _ in all_items
                    if make_prompt_id(d_cfg.name, p) not in existing_ids
                ]
                if remaining:
                    has_work = True
                    break
                # Also check if any existing entries are missing a reasoning step
                if has_entries_missing_reasoning(output_path):
                    has_work = True
                    break

            if not has_work:
                tqdm.write(f"  [SKIP] {model_cfg.name} is already complete for all datasets.")
                continue

            try:
                self._run_for_model(model_cfg, active_datasets, run_id, machine_id, verbose=verbose)
            except Exception as e:
                tqdm.write(f"\n[SKIP] {model_cfg.name} failed: {type(e).__name__}: {e}")
                tqdm.write(f"       Continuing with next model.\n")
                failed.append(model_cfg.name)
                gc.collect()
                if _cuda_is_available():
                    _cuda_empty_cache()

        if _shutdown_requested:
            print("\nStopped. Progress saved — resume by re-running the same command.")
        if failed:
            print(f"\nModels skipped due to errors: {failed}")

        # ── Send notification ────────────────────────────────────────────────
        title = f"GPU Free — Response Generation Finished ({machine_id})"
        status = "STOPPED" if _shutdown_requested else ("PARTIAL" if failed else "SUCCESS")
        
        m_list = [m.name for m in model_cfgs]
        d_list = [d.name for d in dataset_cfgs]
        
        msg = (
            f"Finished generating responses on machine {machine_id}.\n"
            f"Status: {status}\n\n"
            f"Models: {', '.join(m_list)}\n"
            f"Datasets: {', '.join(d_list)}\n"
        )
        if failed:
            msg += f"\nFailed models: {', '.join(failed)}"
        if _shutdown_requested:
            msg += "\n\nNote: Shutdown was requested via Ctrl+C."

        self._notify_ntfy(title, msg)

    # ── per-model pipeline ───────────────────────────────────────────────────

    def _run_for_model(
        self,
        model_cfg: ResolvedModelConfig,
        dataset_cfgs: list[DatasetConfig],
        run_id: str,
        machine_id: str,
        verbose: bool = False,
    ) -> None:
        tqdm.write(f"\n{'='*60}")
        tqdm.write(f"Model : {model_cfg.name}  ({model_cfg.model_id})")
        tqdm.write(f"Backend: {model_cfg.backend}")
        if model_cfg.backend == "vllm":
            tqdm.write(f"Quant : {model_cfg.quantization or 'float16'}")
            tqdm.write(f"GPU mem utilization: {model_cfg.gpu_memory_utilization}")
        tqdm.write(f"{'='*60}")

        if model_cfg.backend == "openai":
            loaded = load_openai_model(model_cfg)
        else:
            loaded = load_vllm_model(model_cfg, self._hf_token, verbose=verbose)
            tqdm.write(f"  Actual quantization: {loaded.actual_quantization or 'float16'}")
        tqdm.write(f"  [SUCCESS] {model_cfg.name} is ready for inference.")

        chunk_size = self.config.generation.write_chunk_size
        tqdm.write(f"  Write chunk size: {chunk_size}\n")

        try:
            if self.config.generation.run_mode == "interleaved":
                self._run_interleaved(loaded, model_cfg, dataset_cfgs, chunk_size, run_id, machine_id)
            else:
                self._run_sequential(loaded, model_cfg, dataset_cfgs, chunk_size, run_id, machine_id)
        finally:
            if isinstance(loaded, LoadedModel):
                del loaded.llm
                gc.collect()
                if _cuda_is_available():
                    _cuda_empty_cache()

    # ── sequential mode ──────────────────────────────────────────────────────

    def _run_sequential(
        self, loaded, model_cfg, dataset_cfgs, chunk_size, run_id, machine_id
    ) -> None:
        for d_cfg in dataset_cfgs:
            self._process_dataset(loaded, model_cfg, d_cfg, chunk_size, run_id, machine_id)

    def _fix_missing_reasoning(
        self, loaded, model_cfg, output_path: Path, chunk_size: int, run_id: str, machine_id: str
    ) -> None:
        """Re-generate any existing entries that have no reasoning step or empty response."""
        entries = load_existing_entries(output_path)
        missing_indices = [
            i for i, e in enumerate(entries)
            if (
                not any(m["role"] == "reasoning" for m in e.messages) or
                not any(m["role"] == "assistant" and m.get("content") for m in e.messages)
            )
            and e.metadata.get("reasoning_fix_attempts", 0) < MAX_REASONING_FIX_ATTEMPTS
        ]
        if not missing_indices:
            return

        tqdm.write(f"  Re-generating {len(missing_indices)} existing entries without reasoning or with empty response...")
        prompts = [
            next((m["content"] for m in entries[i].messages if m["role"] == "user"), "")
            for i in missing_indices
        ]

        pbar = tqdm(total=len(prompts), desc="fix-reasoning", unit="prompt", position=1, leave=False)
        generated_at = datetime.now(timezone.utc).isoformat()

        for start in range(0, len(prompts), chunk_size):
            if _shutdown_requested:
                break
            batch_prompts = prompts[start : start + chunk_size]
            results = generate_batch(loaded, batch_prompts, model_cfg,
                                     max_retries=self.config.generation.max_retries, pbar=pbar)
            for offset, result in enumerate(results):
                idx = missing_indices[start + offset]
                old = entries[idx]
                messages: list[dict] = []
                if model_cfg.system_prompt:
                    messages.append({"role": "system", "content": model_cfg.system_prompt})
                messages.append({"role": "user", "content": prompts[start + offset]})
                if result["reasoning"] is not None:
                    messages.append({"role": "reasoning", "content": result["reasoning"]})
                messages.append({"role": "assistant", "content": result["response"]})
                entries[idx] = ConversationEntry(
                    id=old.id,
                    messages=messages,
                    annotations=old.annotations,
                    model=old.model,
                    judge=old.judge,
                    metadata={
                        **old.metadata,
                        "generated_at": generated_at,
                        "run_id": run_id,
                        "machine_id": machine_id,
                        "finish_reason": result["finish_reason"],
                        "n_new_tokens": result["n_new_tokens"],
                        "has_reasoning": result["reasoning"] is not None,
                        "system_prompt": model_cfg.system_prompt,
                        "reasoning_fix_attempts": old.metadata.get("reasoning_fix_attempts", 0) + 1,
                    },
                )
            # Save after each chunk so progress isn't lost on crash/interrupt
            rewrite_entries(output_path, entries)

        pbar.close()
        tqdm.write(f"  Done fixing reasoning entries.")

    def _process_dataset(
        self, loaded, model_cfg, d_cfg, chunk_size, run_id, machine_id
    ) -> None:
        output_path = self._output_dir / f"{model_cfg.name}_{d_cfg.name}.jsonl"

        # Re-generate any existing entries that are missing a reasoning step
        if output_path.exists():
            self._fix_missing_reasoning(loaded, model_cfg, output_path, chunk_size, run_id, machine_id)

        all_items = self._get_prompts(d_cfg)
        existing_ids = get_existing_ids(output_path)
        remaining = [
            (p, meta) for p, meta in all_items
            if make_prompt_id(d_cfg.name, p) not in existing_ids
        ]

        if not remaining:
            return

        tqdm.write(f"Dataset : {d_cfg.name}  →  {output_path}")
        tqdm.write(f"  {len(remaining)}/{len(all_items)} prompts remaining")

        pbar = tqdm(
            total=len(remaining),
            desc=f"{model_cfg.name}/{d_cfg.name}",
            unit="prompt",
            position=1,
            leave=True,
        )
        for start in range(0, len(remaining), chunk_size):
            if _shutdown_requested:
                break
            self._process_batch(
                loaded, model_cfg, d_cfg,
                remaining[start : start + chunk_size],
                output_path, run_id, machine_id, pbar,
            )
        pbar.close()

    # ── interleaved mode ─────────────────────────────────────────────────────

    def _run_interleaved(
        self, loaded, model_cfg, dataset_cfgs, chunk_size, run_id, machine_id
    ) -> None:
        """Round-robin across datasets: one chunk per dataset per cycle."""
        queues: dict[str, list[str]] = {}
        output_paths: dict[str, Path] = {}
        d_lookup = {d.name: d for d in dataset_cfgs}

        for d_cfg in dataset_cfgs:
            out = self._output_dir / f"{model_cfg.name}_{d_cfg.name}.jsonl"
            output_paths[d_cfg.name] = out
            if out.exists():
                self._fix_missing_reasoning(loaded, model_cfg, out, chunk_size, run_id, machine_id)
            all_items = self._get_prompts(d_cfg)
            existing = get_existing_ids(out)
            remaining = [
                (p, meta) for p, meta in all_items
                if make_prompt_id(d_cfg.name, p) not in existing
            ]
            queues[d_cfg.name] = remaining
            if remaining:
                tqdm.write(f"  {d_cfg.name}: {len(remaining)}/{len(all_items)} remaining")

        total = sum(len(q) for q in queues.values())

        pbar = tqdm(
            total=total,
            desc=f"{model_cfg.name} [interleaved]",
            unit="prompt",
            position=1,
            leave=True,
        )
        while any(queues.values()):
            if _shutdown_requested:
                break
            for d_name in list(queues.keys()):
                if _shutdown_requested:
                    break
                q = queues[d_name]
                if not q:
                    continue
                batch, queues[d_name] = q[:chunk_size], q[chunk_size:]
                pbar.set_postfix({"dataset": d_name})
                self._process_batch(
                    loaded, model_cfg, d_lookup[d_name],
                    batch, output_paths[d_name], run_id, machine_id, pbar,
                )
        pbar.close()

    # ── batch processing ─────────────────────────────────────────────────────

    def _process_batch(
        self,
        loaded: LoadedModel,
        model_cfg: ResolvedModelConfig,
        d_cfg: DatasetConfig,
        batch_items: list[tuple[str, dict]],
        output_path: Path,
        run_id: str,
        machine_id: str,
        pbar: tqdm,
    ) -> None:
        pbar.set_description(f"{model_cfg.name} | {d_cfg.name}")
        batch_prompts = [p for p, _ in batch_items]
        generated_at = datetime.now(timezone.utc).isoformat()
        finish_counts: dict[str, int] = {}

        def build_and_save(idx: int, result: dict) -> None:
            """Build a ConversationEntry for batch_items[idx] and append it to disk."""
            if result is None:
                return
            prompt, row_meta = batch_items[idx]
            prompt_id = make_prompt_id(d_cfg.name, prompt)
            finish_counts[result["finish_reason"]] = (
                finish_counts.get(result["finish_reason"], 0) + 1
            )
            messages: list[dict[str, str]] = []
            if model_cfg.system_prompt:
                messages.append({"role": "system", "content": model_cfg.system_prompt})
            messages.append({"role": "user", "content": prompt})
            if result["reasoning"] is not None:
                messages.append({"role": "reasoning", "content": result["reasoning"]})
            messages.append({"role": "assistant", "content": result["response"]})
            entry = ConversationEntry(
                id=prompt_id,
                messages=messages,
                annotations=[],
                model=model_cfg.model_id,
                judge="",  # populated later during labeling
                metadata={
                    # run context
                    "run_id": run_id,
                    "machine_id": machine_id,
                    "generated_at": generated_at,
                    # model
                    "model_name": model_cfg.name,
                    "backend": model_cfg.backend,
                    "quantization": getattr(loaded, "actual_quantization", None),
                    "max_new_tokens": model_cfg.max_new_tokens,
                    "temperature": model_cfg.temperature,
                    "top_p": model_cfg.top_p,
                    "top_k": model_cfg.top_k,
                    "min_p": model_cfg.min_p,
                    "presence_penalty": model_cfg.presence_penalty,
                    "repetition_penalty": model_cfg.repetition_penalty,
                    "system_prompt": model_cfg.system_prompt,
                    # vLLM engine settings (None for OpenAI-compatible backends)
                    "gpu_memory_utilization": model_cfg.gpu_memory_utilization if model_cfg.backend == "vllm" else None,
                    "actual_max_model_len": getattr(loaded, "actual_max_model_len", None),
                    "actual_enforce_eager": getattr(loaded, "actual_enforce_eager", None),
                    # generation outcome
                    "finish_reason": result["finish_reason"],
                    "n_new_tokens": result["n_new_tokens"],
                    "has_reasoning": result["reasoning"] is not None,
                    # dataset
                    "dataset_name": d_cfg.name,
                    "dataset_source": d_cfg.source,
                    "dataset_path": d_cfg.path,
                    "prompt_id": prompt_id,
                    "chunk_size_used": len(batch_items),
                    # extra fields from the dataset row (category, harm label, etc.)
                    "dataset_row": row_meta,
                },
            )
            append_entry(entry, output_path)

        if isinstance(loaded, LoadedOpenAIModel):
            # OpenAI backend: save each entry as its request completes.
            # on_complete is called inside generate_batch per finished future.
            generate_batch(
                loaded, batch_prompts, model_cfg,
                max_retries=self.config.generation.max_retries,
                pbar=pbar,
                on_complete=build_and_save,
            )
        else:
            # vLLM backend: wait for the full batch, then save all results.
            results = generate_batch(
                loaded, batch_prompts, model_cfg,
                max_retries=self.config.generation.max_retries,
                pbar=pbar,
            )
            for idx, result in enumerate(results):
                build_and_save(idx, result)
            pbar.update(len(batch_items))

        pbar.set_postfix(finish_counts)


# ─────────────────────────────────────────────────────────────────────────────
# Debug helpers
# ─────────────────────────────────────────────────────────────────────────────

def _show_samples(
    gen: ResponseGenerator,
    models: list[str] | None,
    datasets: list[str] | None,
) -> None:
    """Print the formatted prompt for the first item of each model+dataset pair.

    Loads each tokenizer (no model weights) and shows exactly what gets sent
    to the model. Use this to verify chat templates and thinking mode.
    """
    from transformers import AutoTokenizer

    model_cfgs = [
        m.resolve(gen.config.defaults) for m in gen.config.models
        if models is None or m.name in models
    ]
    dataset_cfgs = [
        d for d in gen.config.datasets
        if datasets is None or d.name in datasets
    ]

    for model_cfg in model_cfgs:
        print(f"\n{'='*60}")
        print(f"Model: {model_cfg.name}  ({model_cfg.model_id})")
        print(f"{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.model_id, token=gen._hf_token, trust_remote_code=True
        )

        _test = [{"role": "user", "content": "test"}]
        try:
            tokenizer.apply_chat_template(_test, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=True)
            thinking_supported = True
        except TypeError:
            thinking_supported = False

        print(f"Thinking mode supported: {thinking_supported}")

        for d_cfg in dataset_cfgs:
            items = load_prompts(d_cfg, gen._hf_token)
            if not items:
                continue
            prompt, _ = items[0]
            messages = _make_messages(prompt, model_cfg)

            if thinking_supported:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
                )
            else:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            print(f"\n  Dataset: {d_cfg.name}")
            print(f"  Raw prompt (first 120 chars): {prompt[:120]!r}")
            print(f"  Formatted input:\n{'-'*40}")
            print(formatted)
            print(f"{'-'*40}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LLM responses and save as little_steer JSONL."
    )
    parser.add_argument("--config", default="./config.yaml", help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", metavar="NAME", default=None,
                        help="Model names to run (default: all)")
    parser.add_argument("--datasets", nargs="+", metavar="NAME", default=None,
                        help="Dataset names to run (default: all)")
    parser.add_argument("--show-sample", action="store_true",
                        help="Print the formatted prompt for the first item of each "
                             "model+dataset pair, then exit. Use this to verify the "
                             "chat template is being applied correctly.")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full vLLM/Ray logs and tracebacks.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    gen = ResponseGenerator.from_config(args.config)

    if args.show_sample:
        _show_samples(gen, args.models, args.datasets)
        return

    gen.run(models=args.models, datasets=args.datasets, verbose=args.verbose)


if __name__ == "__main__":
    main()
