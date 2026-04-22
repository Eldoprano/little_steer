#!/usr/bin/env python3
"""
generate_responses.py — LLM response generation pipeline.

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
import logging
import os
import re
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# ── Env vars (must be set before vLLM / transformers import) ─────────────────
os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

for _name in ("vllm", "ray", "transformers", "datasets"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import requests  # noqa: E402
import yaml  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.progress import (  # noqa: E402
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

try:
    import torch  # noqa: E402
    def _cuda_empty(): torch.cuda.empty_cache()
    def _cuda_available(): return torch.cuda.is_available()
except ImportError:
    def _cuda_empty(): pass
    def _cuda_available(): return False

from thesis_schema import ConversationEntry  # noqa: E402

_console = Console(highlight=False)


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────────────────────

_shutdown_requested = False


def _handle_sigint(sig, frame) -> None:
    """First Ctrl+C → stop after the current chunk. Second Ctrl+C → hard exit."""
    global _shutdown_requested
    if _shutdown_requested:
        _console.print("\n[red]Forced exit.[/red]")
        os._exit(1)
    _shutdown_requested = True
    _console.print(
        "\n[yellow][Ctrl+C] Finishing current chunk, then stopping. "
        "Press Ctrl+C again to force quit.[/yellow]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config models
# ─────────────────────────────────────────────────────────────────────────────

QuantizationOption = Literal["4bit", "8bit", "auto", "none"] | None


class ModelDefaults(BaseModel):
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
    max_model_len: int | None = None
    enforce_eager: bool = False


class ModelConfig(BaseModel):
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
    tokenizer: str | None = None
    hf_config_path: str | None = None
    skip_datasets: list[str] = Field(default_factory=list)
    mistral_format: bool = False
    think_prefix: bool = False
    backend: Literal["vllm", "openai"] = "vllm"
    openai_base_url: str | None = None
    openai_api_key: str | None = None

    def resolve(self, d: ModelDefaults) -> ResolvedModelConfig:
        def pick(val, default):
            return val if val is not None else default

        return ResolvedModelConfig(
            name=self.name,
            model_id=self.model_id,
            temperature=pick(self.temperature, d.temperature),
            top_p=pick(self.top_p, d.top_p),
            top_k=pick(self.top_k, d.top_k),
            min_p=pick(self.min_p, d.min_p),
            max_new_tokens=pick(self.max_new_tokens, d.max_new_tokens),
            quantization=pick(self.quantization, d.quantization),
            system_prompt=self.system_prompt,
            presence_penalty=pick(self.presence_penalty, d.presence_penalty),
            repetition_penalty=pick(self.repetition_penalty, d.repetition_penalty),
            gpu_memory_utilization=pick(self.gpu_memory_utilization, d.gpu_memory_utilization),
            max_model_len=pick(self.max_model_len, d.max_model_len),
            enforce_eager=pick(self.enforce_eager, d.enforce_eager),
            tokenizer=self.tokenizer,
            hf_config_path=self.hf_config_path,
            skip_datasets=self.skip_datasets,
            mistral_format=self.mistral_format,
            think_prefix=self.think_prefix,
            backend=self.backend,
            openai_base_url=self.openai_base_url,
            openai_api_key=self.openai_api_key,
        )


class ResolvedModelConfig(BaseModel):
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
    gpu_memory_utilization: float
    max_model_len: int | None
    enforce_eager: bool
    tokenizer: str | None = None
    hf_config_path: str | None = None
    skip_datasets: list[str] = Field(default_factory=list)
    mistral_format: bool = False
    think_prefix: bool = False
    backend: Literal["vllm", "openai"] = "vllm"
    openai_base_url: str | None = None
    openai_api_key: str | None = None


class DatasetConfig(BaseModel):
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
    safety: Literal["safe", "unsafe", "mixed"] | None = None
    safety_field: str | None = None
    unsafe_value: str | None = None


class GenerationConfig(BaseModel):
    write_chunk_size: int = 8
    max_retries: int = 1
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
            return cls.model_validate(yaml.safe_load(f))


# ─────────────────────────────────────────────────────────────────────────────
# Loaded-model bundles
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadedModel:
    llm: Any
    thinking_supported: bool = False
    actual_quantization: str | None = None
    actual_max_model_len: int | None = None
    actual_enforce_eager: bool = False


@dataclass
class LoadedOpenAIModel:
    client: Any
    model_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _map_quantization(q: QuantizationOption) -> str | None:
    if q in (None, "auto", "none"):
        return None
    if q in ("4bit", "8bit"):
        return "bitsandbytes"
    return None


def _detect_thinking_support(llm: Any) -> bool:
    try:
        tok = llm.get_tokenizer()
        tok.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return True
    except Exception:
        return False


def _try_load_vllm(model_cfg: ResolvedModelConfig, quant: str | None) -> Any | None:
    """Attempt a single vLLM load. Returns None on failure (prints concise error)."""
    from vllm import LLM

    # bitsandbytes requires eager mode
    eager = model_cfg.enforce_eager or (quant == "bitsandbytes")

    kwargs: dict[str, Any] = {
        "model": model_cfg.model_id,
        "dtype": "auto",
        "gpu_memory_utilization": model_cfg.gpu_memory_utilization,
        "trust_remote_code": True,
        # Cap concurrency — reasoning models generate long, so >32 thrashes KV cache.
        "max_num_seqs": 32,
        "enforce_eager": eager,
        "disable_log_stats": True,
    }
    if model_cfg.max_model_len is not None:
        kwargs["max_model_len"] = model_cfg.max_model_len
    if quant is not None:
        kwargs["quantization"] = quant
        kwargs["load_format"] = "bitsandbytes"
    if model_cfg.tokenizer is not None:
        kwargs["tokenizer"] = model_cfg.tokenizer
    if model_cfg.hf_config_path is not None:
        kwargs["hf_config_path"] = model_cfg.hf_config_path
    if model_cfg.mistral_format:
        kwargs["tokenizer_mode"] = "mistral"

    try:
        return LLM(**kwargs)
    except Exception as e:
        msg = str(e)
        if "out of memory" in msg.lower() or "OutOfMemoryError" in msg:
            _console.print("  [red]Load failed:[/red] CUDA out of memory")
        else:
            _console.print(f"  [red]Load failed:[/red] {msg.splitlines()[0][:200]}")
        gc.collect()
        _cuda_empty()
        return None


def load_vllm_model(model_cfg: ResolvedModelConfig, hf_token: str | None) -> LoadedModel:
    """Load a vLLM model. Handles quantization='auto' by trying float16 → bnb."""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    _console.print(f"  Loading: {model_cfg.model_id}")

    candidates = (
        [(None,), ("bitsandbytes",)]
        if model_cfg.quantization == "auto"
        else [(_map_quantization(model_cfg.quantization),)]
    )

    for (quant,) in candidates:
        label = quant or "float16"
        if model_cfg.quantization == "auto":
            _console.print(f"  Trying {label}…")
        llm = _try_load_vllm(model_cfg, quant)
        if llm is not None:
            thinking = _detect_thinking_support(llm)
            actual_max = None
            actual_eager = False
            try:
                actual_max = llm.llm_engine.model_config.max_model_len
                actual_eager = llm.llm_engine.model_config.enforce_eager
            except Exception:
                pass
            return LoadedModel(llm, thinking, label, actual_max, actual_eager)

    raise RuntimeError(f"Could not load {model_cfg.model_id}")


def load_openai_model(model_cfg: ResolvedModelConfig) -> LoadedOpenAIModel:
    from openai import OpenAI
    base_url = model_cfg.openai_base_url or "http://localhost:1234/v1"
    api_key = model_cfg.openai_api_key or "lm-studio"
    _console.print(f"  Connecting to {base_url}  (model={model_cfg.model_id})")
    return LoadedOpenAIModel(client=OpenAI(base_url=base_url, api_key=api_key),
                             model_id=model_cfg.model_id)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _extract_prompt_text(value: Any, index: int | None = None) -> str:
    if isinstance(value, list):
        idx = index if index is not None else 0
        if 0 <= idx < len(value):
            item = value[idx]
            if isinstance(item, dict):
                return str(item.get("value") or item.get("content") or item)
            return str(item)
        return ""
    if isinstance(value, dict):
        return str(value.get("value") or value.get("content") or value)
    return str(value)


def _extract_extra(row: dict, prompt_field: str,
                   extra_fields: list[str] | Literal["*"] | None) -> dict:
    if extra_fields is None:
        return {}
    if extra_fields == "*":
        return {k: v for k, v in row.items() if k != prompt_field}
    return {k: row[k] for k in extra_fields if k in row}


def load_prompts(dataset_cfg: DatasetConfig,
                 hf_token: str | None) -> list[tuple[str, dict]]:
    """Return list of (prompt_text, row_metadata) tuples."""
    if dataset_cfg.source == "hf":
        from datasets import load_dataset as hf_load
        kwargs: dict[str, Any] = {"token": hf_token}
        if dataset_cfg.subset:
            kwargs["name"] = dataset_cfg.subset
        _console.print(f"  Downloading {dataset_cfg.path} (split={dataset_cfg.split})")
        ds = hf_load(dataset_cfg.path, split=dataset_cfg.split, **kwargs)

        if dataset_cfg.filter_max_list_len is not None:
            ds = ds.filter(lambda row: (dataset_cfg.prompt_field in row)
                           and isinstance(row[dataset_cfg.prompt_field], list)
                           and len(row[dataset_cfg.prompt_field]) <= dataset_cfg.filter_max_list_len)

        items = [
            (_extract_prompt_text(row[dataset_cfg.prompt_field], dataset_cfg.prompt_index),
             _extract_extra(dict(row), dataset_cfg.prompt_field, dataset_cfg.extra_fields))
            for row in ds
        ][:dataset_cfg.max_samples]
        return items

    if dataset_cfg.source == "local":
        path = Path(dataset_cfg.path)
        items: list[tuple[str, dict]] = []
        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    val = row.get(dataset_cfg.prompt_field)
                    if dataset_cfg.filter_max_list_len is not None:
                        if not (isinstance(val, list)
                                and len(val) <= dataset_cfg.filter_max_list_len):
                            continue
                    items.append((
                        _extract_prompt_text(val, dataset_cfg.prompt_index),
                        _extract_extra(row, dataset_cfg.prompt_field, dataset_cfg.extra_fields),
                    ))
        elif path.suffix == ".csv":
            import csv
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    val = row.get(dataset_cfg.prompt_field)
                    if dataset_cfg.filter_max_list_len is not None:
                        if not (isinstance(val, list)
                                and len(val) <= dataset_cfg.filter_max_list_len):
                            continue
                    items.append((
                        _extract_prompt_text(val, dataset_cfg.prompt_index),
                        _extract_extra(row, dataset_cfg.prompt_field, dataset_cfg.extra_fields),
                    ))
        else:
            raise ValueError(f"Unsupported local file: {path.suffix}")
        return items[:dataset_cfg.max_samples]

    raise ValueError(f"Unknown dataset source: {dataset_cfg.source!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_prompt_id(dataset_name: str, prompt: str) -> str:
    """Stable 16-char MD5 for (dataset, prompt)."""
    return hashlib.md5(f"{dataset_name}::{prompt}".encode()).hexdigest()[:16]


def make_entry_id(dataset_name: str, prompt: str, model_id: str) -> str:
    """Stable 16-char MD5 for one generated sample."""
    return hashlib.md5(f"{dataset_name}::{prompt}::{model_id}".encode()).hexdigest()[:16]


def make_generation_hash(reasoning: str | None) -> str:
    """Stable 16-char MD5 of the generated reasoning content.

    Changes whenever the reasoning text changes (e.g. after regeneration).
    Empty string if no reasoning was produced.
    """
    return hashlib.md5((reasoning or "").encode("utf-8")).hexdigest()[:16]


def _make_messages(prompt: str, model_cfg: ResolvedModelConfig) -> list[dict]:
    msgs: list[dict] = []
    if model_cfg.system_prompt:
        msgs.append({"role": "system", "content": model_cfg.system_prompt})
    msgs.append({"role": "user", "content": prompt})
    return msgs


_PARSE_RES = [
    re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL),
    re.compile(r"(.*?)</think>(.*)", re.DOTALL),
    re.compile(r"\[THINK\](.*?)\[/THINK\](.*)", re.DOTALL),
    re.compile(r"(.*?)\[/THINK\](.*)", re.DOTALL),
]


def _parse_response(raw_text: str) -> tuple[str | None, str]:
    """Split raw text into (reasoning, final_response)."""
    for pat in _PARSE_RES:
        m = pat.search(raw_text)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None, raw_text.strip()


def _classify_finish(vllm_finish: str, raw_text: str) -> str:
    """Map (vLLM finish_reason, generated text) → our canonical finish_reason."""
    if vllm_finish == "length":
        return "max_length"
    if ("<think>" in raw_text and "</think>" not in raw_text) or \
       ("[THINK]" in raw_text and "[/THINK]" not in raw_text):
        return "incomplete_thinking"
    return "eos"


def _has_think_artifact(text: str) -> bool:
    s = text.lstrip()
    return s.startswith("[/THINK]") or s.startswith("</think>")


def _is_approved(finish_reason: str, reasoning: str | None, response: str) -> bool:
    """Cheap approval check (same criteria as fix_quality.py).

    We deliberately do NOT run the expensive n-gram repetition detector here —
    that's the job of `fix_quality.py --tag`, which can run offline over the
    whole dataset without slowing generation.
    """
    if finish_reason in ("max_length", "failed"):
        return False
    if not response:
        return False
    if _has_think_artifact(response):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Batched generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch_vllm(
    loaded: LoadedModel,
    prompts: list[str],
    model_cfg: ResolvedModelConfig,
    max_retries: int = 1,
) -> list[dict]:
    """Generate for a batch of prompts using vLLM's native batching.

    All prompts go through a single `llm.chat()` call so vLLM's PagedAttention
    scheduler runs them in parallel. Prompts whose reasoning never closed are
    retried in a smaller follow-up batch (up to `max_retries` times).

    Returns one dict per prompt with keys: reasoning, response, finish_reason,
    n_new_tokens.
    """
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        top_k=model_cfg.top_k if model_cfg.top_k is not None else -1,
        min_p=model_cfg.min_p if model_cfg.min_p is not None else 0.0,
        max_tokens=model_cfg.max_new_tokens,
        presence_penalty=model_cfg.presence_penalty
            if model_cfg.presence_penalty is not None else 0.0,
        repetition_penalty=model_cfg.repetition_penalty
            if model_cfg.repetition_penalty is not None else 1.0,
    )
    chat_template_kwargs = {"enable_thinking": True} if loaded.thinking_supported else {}

    conversations = [_make_messages(p, model_cfg) for p in prompts]

    # Optional global think-prefix (Mistral uses [THINK], others <think>).
    extra_chat: dict[str, Any] = {}
    if model_cfg.think_prefix:
        prefix = "[THINK]" if model_cfg.mistral_format else "<think>"
        for conv in conversations:
            conv.append({"role": "assistant", "content": prefix})
        if model_cfg.mistral_format:
            extra_chat = {"add_generation_prompt": False,
                          "continue_final_message": True}

    results: list[dict | None] = [None] * len(prompts)
    pending = list(range(len(prompts)))

    for attempt in range(max_retries + 1):
        if not pending:
            break
        batch = [conversations[i] for i in pending]
        outs = loaded.llm.chat(
            batch, sampling_params,
            chat_template_kwargs=chat_template_kwargs,
            use_tqdm=False,  # we render our own Rich progress
            **extra_chat,
        )
        still_pending: list[int] = []
        is_last = attempt == max_retries
        for j, gi in enumerate(pending):
            out = outs[j].outputs[0]
            reason = _classify_finish(out.finish_reason, out.text)
            if reason == "incomplete_thinking" and not is_last:
                still_pending.append(gi)
                continue
            reasoning, response = _parse_response(out.text)
            results[gi] = {
                "reasoning": reasoning,
                "response": response,
                "finish_reason": reason,
                "n_new_tokens": len(out.token_ids),
            }
        pending = still_pending

    # Safety net for any slot that somehow stayed None.
    for i, r in enumerate(results):
        if r is None:
            results[i] = {"reasoning": None, "response": "",
                          "finish_reason": "failed", "n_new_tokens": 0}
    return results  # type: ignore[return-value]


def generate_batch_openai(
    loaded: LoadedOpenAIModel,
    prompts: list[str],
    model_cfg: ResolvedModelConfig,
    max_retries: int = 1,
    max_workers: int = 8,
    on_complete: Any = None,
) -> list[dict]:
    """Parallel OpenAI-compatible calls. Returns results in prompt order."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def call_one(prompt: str) -> dict:
        messages = _make_messages(prompt, model_cfg)
        if model_cfg.think_prefix:
            prefix = "[THINK]" if model_cfg.mistral_format else "<think>"
            messages.append({"role": "assistant", "content": prefix})

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
                reason = _classify_finish(resp.choices[0].finish_reason, raw_text)

                if reason == "incomplete_thinking" and attempt < max_retries:
                    continue

                # Prefer dedicated reasoning field (LMStudio: reasoning/reasoning_content)
                api_reasoning = (getattr(msg, "reasoning", None)
                                 or getattr(msg, "reasoning_content", None))
                if api_reasoning:
                    reasoning = api_reasoning.strip() or None
                    response = raw_text.strip()
                    if response and _has_think_artifact(response):
                        for tag in ("[/THINK]", "</think>"):
                            if response.lstrip().startswith(tag):
                                response = response.lstrip()[len(tag):].lstrip()
                                break
                else:
                    reasoning, response = _parse_response(raw_text)

                return {"reasoning": reasoning, "response": response,
                        "finish_reason": reason, "n_new_tokens": n_new}
            except Exception as e:
                if attempt == max_retries:
                    _console.print(f"  [red]API call failed: {e}[/red]")
                    return {"reasoning": None, "response": "",
                            "finish_reason": "failed", "n_new_tokens": 0}
                wait = 5 * (2 ** attempt)
                _console.print(f"  [yellow]API error, retrying in {wait}s: {e}[/yellow]")
                time.sleep(wait)
        return {"reasoning": None, "response": "",
                "finish_reason": "failed", "n_new_tokens": 0}

    workers = min(max_workers, max(1, len(prompts)))
    results: list[dict | None] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_to_idx = {ex.submit(call_one, p): i for i, p in enumerate(prompts)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()
            if on_complete is not None:
                on_complete(results[i])
    return results  # type: ignore[return-value]


def generate_batch(
    loaded: LoadedModel | LoadedOpenAIModel,
    prompts: list[str],
    model_cfg: ResolvedModelConfig,
    max_retries: int = 1,
    on_complete: Any = None,
) -> list[dict]:
    if isinstance(loaded, LoadedOpenAIModel):
        return generate_batch_openai(loaded, prompts, model_cfg, max_retries,
                                     on_complete=on_complete)
    return generate_batch_vllm(loaded, prompts, model_cfg, max_retries)


# ─────────────────────────────────────────────────────────────────────────────
# JSONL I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_store(dataset_path: Path) -> dict[str, ConversationEntry]:
    """Load the canonical dataset file into an ID-keyed store."""
    store: dict[str, ConversationEntry] = {}
    if not dataset_path.exists():
        return store
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = ConversationEntry.model_validate_json(line)
            except Exception:
                continue
            store[entry.id] = entry
    return store


def write_dataset_store(dataset_path: Path, store: dict[str, ConversationEntry]) -> None:
    """Atomically rewrite the canonical dataset file."""
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dataset_path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for entry_id in sorted(store):
            f.write(store[entry_id].model_dump_json() + "\n")
    tmp.replace(dataset_path)


# ─────────────────────────────────────────────────────────────────────────────
# ResponseGenerator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _ChunkStats:
    eos: int = 0
    max_length: int = 0
    incomplete: int = 0
    failed: int = 0

    def record(self, reason: str) -> None:
        if reason == "eos":
            self.eos += 1
        elif reason == "max_length":
            self.max_length += 1
        elif reason == "incomplete_thinking":
            self.incomplete += 1
        else:
            self.failed += 1

    def inline(self) -> str:
        return (f"[green]✓{self.eos}[/green] "
                f"[yellow]⚠{self.max_length}[/yellow] "
                f"[red]✗{self.incomplete + self.failed}[/red]")


class ResponseGenerator:
    """Main pipeline. Load from a config file, then call run().

    Example:
        gen = ResponseGenerator.from_config("config.yaml")
        gen.run()                          # all models & datasets
        gen.run(models=["qwen3-8b"])       # one model
        gen.run(datasets=["harmful"])      # one dataset
    """

    def __init__(self, config: GeneratorConfig, config_path: Path | None = None):
        self.config = config
        self._config_path = config_path
        self._output_dir = (
            (config_path.parent / config.output_dir).resolve()
            if config_path is not None else Path(config.output_dir)
        )
        self._dataset_path = self._output_dir / "dataset.jsonl"
        self._hf_token = self._load_hf_token()
        self._prompt_cache: dict[str, list[tuple[str, dict]]] = {}
        self._dataset_store = load_dataset_store(self._dataset_path)

    @classmethod
    def from_config(cls, path: str | Path) -> ResponseGenerator:
        path = Path(path)
        return cls(GeneratorConfig.from_yaml(path), config_path=path)

    # ── HF token ─────────────────────────────────────────────────────────────

    def _load_hf_token(self) -> str | None:
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

    def _get_prompts(self, dataset_cfg: DatasetConfig) -> list[tuple[str, dict]]:
        if dataset_cfg.name not in self._prompt_cache:
            prompts = load_prompts(dataset_cfg, self._hf_token)
            self._prompt_cache[dataset_cfg.name] = prompts
            _console.print(f"    Loaded {len(prompts)} prompts.")
        return self._prompt_cache[dataset_cfg.name]

    # ── notifications ───────────────────────────────────────────────────────

    def _notify_ntfy(self, title: str, message: str,
                     topic: str = "eldoprano_master") -> None:
        try:
            requests.post(
                f"https://ntfy.sh/{topic}",
                data=message.encode("utf-8"),
                headers={"Title": title.encode("utf-8").decode("latin-1",
                                                               errors="replace")},
                timeout=5,
            )
        except Exception as e:
            _console.print(f"  [dim][notify failed] {e}[/dim]")

    # ── public API ───────────────────────────────────────────────────────────

    def run(self, models: list[str] | None = None,
            datasets: list[str] | None = None) -> None:
        run_id = str(uuid.uuid4())

        model_cfgs = [m.resolve(self.config.defaults) for m in self.config.models
                      if models is None or m.name in models]
        dataset_cfgs = [d for d in self.config.datasets
                        if datasets is None or d.name in datasets]

        if not model_cfgs:
            _console.print("No models matched — nothing to do.")
            return
        if not dataset_cfgs:
            _console.print("No datasets matched — nothing to do.")
            return

        _console.print(f"\nRun ID     : {run_id}")
        _console.print(f"Models     : {[m.name for m in model_cfgs]}")
        _console.print(f"Datasets   : {[d.name for d in dataset_cfgs]}")
        _console.print(f"Mode       : {self.config.generation.run_mode}")
        _console.print(f"Chunk size : {self.config.generation.write_chunk_size}")
        _console.print(f"Output     : {self._dataset_path}\n")

        failed: list[str] = []
        for model_cfg in model_cfgs:
            if _shutdown_requested:
                break
            active = [d for d in dataset_cfgs if d.name not in model_cfg.skip_datasets]
            if not self._has_work(model_cfg, active):
                _console.print(f"  [dim][SKIP] {model_cfg.name}: already complete.[/dim]")
                continue

            try:
                self._run_for_model(model_cfg, active, run_id)
            except Exception as e:
                from rich.markup import escape
                _console.print(f"\n[red][SKIP] {model_cfg.name} failed: "
                               f"{type(e).__name__}: {escape(str(e))}[/red]")
                failed.append(model_cfg.name)
                gc.collect()
                _cuda_empty()

        if _shutdown_requested:
            _console.print("\n[yellow]Stopped. Progress saved — re-run to resume.[/yellow]")
        if failed:
            _console.print(f"\n[red]Skipped: {failed}[/red]")

        status = "STOPPED" if _shutdown_requested else ("PARTIAL" if failed else "SUCCESS")
        title = "GPU Free — Response Generation Finished"
        msg = (f"Status: {status}\n"
               f"Models: {', '.join(m.name for m in model_cfgs)}\n"
               f"Datasets: {', '.join(d.name for d in dataset_cfgs)}")
        if failed:
            msg += f"\nFailed: {', '.join(failed)}"
        self._notify_ntfy(title, msg)

    # ── has_work (pre-load check) ───────────────────────────────────────────

    def _has_work(self, model_cfg: ResolvedModelConfig,
                  dataset_cfgs: list[DatasetConfig]) -> bool:
        for d_cfg in dataset_cfgs:
            all_items = self._get_prompts(d_cfg)
            existing = set(self._dataset_store)
            if any(make_entry_id(d_cfg.name, p, model_cfg.model_id) not in existing
                   for p, _ in all_items):
                return True
        return False

    # ── per-model pipeline ───────────────────────────────────────────────────

    def _run_for_model(self, model_cfg: ResolvedModelConfig,
                       dataset_cfgs: list[DatasetConfig],
                       run_id: str) -> None:
        _console.print(f"\n{'='*60}")
        _console.print(f"Model  : [bold]{model_cfg.name}[/bold]  ({model_cfg.model_id})")
        _console.print(f"Backend: {model_cfg.backend}")
        if model_cfg.backend == "vllm":
            _console.print(f"Quant  : {model_cfg.quantization or 'float16'}")
            _console.print(f"GPU mem: {model_cfg.gpu_memory_utilization}")
        _console.print("=" * 60)

        if model_cfg.backend == "openai":
            loaded: LoadedModel | LoadedOpenAIModel = load_openai_model(model_cfg)
        else:
            loaded = load_vllm_model(model_cfg, self._hf_token)
            _console.print(f"  [green][OK][/green] "
                           f"quant={loaded.actual_quantization} "
                           f"max_model_len={loaded.actual_max_model_len} "
                           f"eager={loaded.actual_enforce_eager}")

        chunk_size = self.config.generation.write_chunk_size
        max_retries = self.config.generation.max_retries

        try:
            if self.config.generation.run_mode == "interleaved":
                self._run_interleaved(loaded, model_cfg, dataset_cfgs,
                                      chunk_size, max_retries, run_id)
            else:
                self._run_sequential(loaded, model_cfg, dataset_cfgs,
                                     chunk_size, max_retries, run_id)
        finally:
            if isinstance(loaded, LoadedModel):
                del loaded.llm
                gc.collect()
                _cuda_empty()

    # ── sequential & interleaved ─────────────────────────────────────────────

    def _run_sequential(self, loaded, model_cfg, dataset_cfgs,
                        chunk_size, max_retries, run_id) -> None:
        for d_cfg in dataset_cfgs:
            if _shutdown_requested:
                break
            self._process_dataset(loaded, model_cfg, d_cfg, chunk_size,
                                  max_retries, run_id)

    def _run_interleaved(self, loaded, model_cfg, dataset_cfgs,
                         chunk_size, max_retries, run_id) -> None:
        """Round-robin one chunk per dataset per cycle."""
        queues: dict[str, list[tuple[str, dict]]] = {}
        for d_cfg in dataset_cfgs:
            all_items = self._get_prompts(d_cfg)
            existing = set(self._dataset_store)
            remaining = [(p, meta) for p, meta in all_items
                         if make_entry_id(d_cfg.name, p, model_cfg.model_id) not in existing]
            queues[d_cfg.name] = remaining
            if remaining:
                _console.print(f"  {d_cfg.name}: {len(remaining)}/{len(all_items)} "
                               "new prompts to generate")

        d_lookup = {d.name: d for d in dataset_cfgs}
        totals = {n: len(q) for n, q in queues.items()}
        total_all = sum(totals.values())
        if total_all == 0:
            return

        stats = _ChunkStats()
        with self._make_progress() as progress:
            task = progress.add_task("", total=total_all,
                                     ds=f"[interleaved: {len(queues)} datasets]",
                                     stats=stats.inline())
            while any(queues.values()):
                if _shutdown_requested:
                    break
                for d_name in list(queues.keys()):
                    if _shutdown_requested:
                        break
                    q = queues[d_name]
                    if not q:
                        continue
                    chunk, queues[d_name] = q[:chunk_size], q[chunk_size:]
                    self._process_chunk(
                        loaded, model_cfg, d_lookup[d_name], chunk,
                        max_retries, run_id,
                        stats, progress, task,
                    )

    # ── dataset processor ───────────────────────────────────────────────────

    def _process_dataset(self, loaded, model_cfg, d_cfg, chunk_size,
                         max_retries, run_id) -> None:
        all_items = self._get_prompts(d_cfg)
        existing = set(self._dataset_store)
        remaining = [(p, meta) for p, meta in all_items
                     if make_entry_id(d_cfg.name, p, model_cfg.model_id) not in existing]
        if not remaining:
            return

        _console.print(f"\nDataset: [cyan]{d_cfg.name}[/cyan] → {self._dataset_path}")
        _console.print(f"  {len(remaining)}/{len(all_items)} new prompts")

        stats = _ChunkStats()
        with self._make_progress() as progress:
            task = progress.add_task(
                "", total=len(remaining),
                ds=f"[cyan]{d_cfg.name}[/cyan]",
                stats=stats.inline(),
            )
            for start in range(0, len(remaining), chunk_size):
                if _shutdown_requested:
                    break
                chunk = remaining[start:start + chunk_size]
                self._process_chunk(loaded, model_cfg, d_cfg, chunk,
                                    max_retries, run_id,
                                    stats, progress, task)

    # ── chunk processing ────────────────────────────────────────────────────

    def _make_progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("{task.fields[ds]}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("{task.fields[stats]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=_console,
            transient=False,
            speed_estimate_period=3600,
        )

    def _process_chunk(
        self,
        loaded: LoadedModel | LoadedOpenAIModel,
        model_cfg: ResolvedModelConfig,
        d_cfg: DatasetConfig,
        chunk: list[tuple[str, dict]],
        max_retries: int,
        run_id: str,
        stats: _ChunkStats,
        progress: Progress,
        task: Any,
    ) -> None:
        prompts = [p for p, _ in chunk]
        t0 = time.time()

        # For OpenAI backends, advance progress per-sample as each call finishes
        # so Rich can compute ETA from the very first result.
        openai_mode = isinstance(loaded, LoadedOpenAIModel)
        if openai_mode:
            def _on_complete(result: dict) -> None:
                stats.record(result["finish_reason"])
                progress.update(task, advance=1, stats=stats.inline())
            results = generate_batch(loaded, prompts, model_cfg,
                                     max_retries=max_retries, on_complete=_on_complete)
        else:
            results = generate_batch(loaded, prompts, model_cfg, max_retries=max_retries)

        dt = max(1e-6, time.time() - t0)
        generated_at = datetime.now(timezone.utc).isoformat()
        total_tokens = 0
        for (prompt, row_meta), result in zip(chunk, results):
            total_tokens += result["n_new_tokens"]
            entry = self._build_entry(
                prompt=prompt,
                row_meta=row_meta,
                result=result,
                model_cfg=model_cfg,
                d_cfg=d_cfg,
                loaded=loaded,
                run_id=run_id,
                generated_at=generated_at,
            )
            self._dataset_store[entry.id] = entry
            if not openai_mode:
                stats.record(result["finish_reason"])

        write_dataset_store(self._dataset_path, self._dataset_store)

        tok_per_s = total_tokens / dt
        if openai_mode:
            progress.update(task, stats=f"{stats.inline()} [dim]{tok_per_s:.0f} tok/s[/dim]")
        else:
            progress.update(
                task,
                advance=len(chunk),
                stats=f"{stats.inline()} [dim]{tok_per_s:.0f} tok/s[/dim]",
            )

    def _build_entry(
        self,
        *,
        prompt: str,
        row_meta: dict,
        result: dict,
        model_cfg: ResolvedModelConfig,
        d_cfg: DatasetConfig,
        loaded: LoadedModel | LoadedOpenAIModel,
        run_id: str,
        generated_at: str,
    ) -> ConversationEntry:
        prompt_id = make_prompt_id(d_cfg.name, prompt)
        entry_id = make_entry_id(d_cfg.name, prompt, model_cfg.model_id)

        messages: list[dict[str, str]] = []
        if model_cfg.system_prompt:
            messages.append({"role": "system", "content": model_cfg.system_prompt})
        messages.append({"role": "user", "content": prompt})
        if result["reasoning"] is not None:
            messages.append({"role": "reasoning", "content": result["reasoning"]})
        messages.append({"role": "assistant", "content": result["response"]})

        # Prompt safety.
        if d_cfg.safety == "safe":
            prompt_safety = "safe"
        elif d_cfg.safety == "unsafe":
            prompt_safety = "unsafe"
        elif d_cfg.safety == "mixed" and d_cfg.safety_field is not None:
            prompt_safety = ("unsafe"
                             if row_meta.get(d_cfg.safety_field) == d_cfg.unsafe_value
                             else "safe")
        else:
            prompt_safety = "unknown"

        entry = ConversationEntry(
            id=entry_id,
            messages=messages,
            annotations=[],
            model=model_cfg.model_id,
            judge="",
            metadata={
                "dataset_name": d_cfg.name,
                "dataset_source": d_cfg.source,
                "dataset_path": d_cfg.path,
                "dataset_row": row_meta,
                "prompt_id": prompt_id,
                "prompt_safety": prompt_safety,
                "generation": {
                    "run_id": run_id,
                    "generated_at": generated_at,
                    "model_name": model_cfg.name,
                    "model_id": model_cfg.model_id,
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
                    "gpu_memory_utilization": (
                        model_cfg.gpu_memory_utilization if model_cfg.backend == "vllm" else None
                    ),
                    "actual_max_model_len": getattr(loaded, "actual_max_model_len", None),
                    "actual_enforce_eager": getattr(loaded, "actual_enforce_eager", None),
                    "finish_reason": result["finish_reason"],
                    "n_new_tokens": result["n_new_tokens"],
                    "has_reasoning": result["reasoning"] is not None,
                    "generation_hash": make_generation_hash(result["reasoning"]),
                },
                "generation_hash": make_generation_hash(result["reasoning"]),
                "quality": {
                    "approved": _is_approved(
                        result["finish_reason"],
                        result["reasoning"],
                        result["response"],
                    ),
                    "issues": [],
                    "checked_at": generated_at,
                },
                "approved": _is_approved(result["finish_reason"], result["reasoning"], result["response"]),
            },
        )
        return entry


# ─────────────────────────────────────────────────────────────────────────────
# --show-sample debug helper
# ─────────────────────────────────────────────────────────────────────────────

def _show_samples(
    gen: ResponseGenerator,
    models: list[str] | None,
    datasets: list[str] | None,
) -> None:
    from transformers import AutoTokenizer

    model_cfgs = [m.resolve(gen.config.defaults) for m in gen.config.models
                  if models is None or m.name in models]
    dataset_cfgs = [d for d in gen.config.datasets
                    if datasets is None or d.name in datasets]

    for model_cfg in model_cfgs:
        _console.print(f"\n{'='*60}")
        _console.print(f"Model: {model_cfg.name}  ({model_cfg.model_id})")
        _console.print("=" * 60)
        tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.model_id, token=gen._hf_token, trust_remote_code=True
        )
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            thinking = True
        except TypeError:
            thinking = False
        _console.print(f"Thinking supported: {thinking}")

        for d_cfg in dataset_cfgs:
            items = load_prompts(d_cfg, gen._hf_token)
            if not items:
                continue
            prompt, _ = items[0]
            messages = _make_messages(prompt, model_cfg)
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **({"enable_thinking": True} if thinking else {}),
            )
            _console.print(f"\n  Dataset: {d_cfg.name}")
            _console.print(f"  Raw prompt (first 120): {prompt[:120]!r}")
            _console.print("  Formatted input:")
            _console.print("-" * 40)
            _console.print(formatted)
            _console.print("-" * 40)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate LLM responses and save as little_steer JSONL."
    )
    parser.add_argument("--config", default="./config.yaml",
                        help="Path to config.yaml")
    parser.add_argument("--models", nargs="+", metavar="NAME", default=None,
                        help="Model names to run (default: all)")
    parser.add_argument("--datasets", nargs="+", metavar="NAME", default=None,
                        help="Dataset names to run (default: all)")
    parser.add_argument("--show-sample", action="store_true",
                        help="Print formatted prompt for first item of each "
                             "(model, dataset) pair, then exit.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)
    gen = ResponseGenerator.from_config(args.config)

    if args.show_sample:
        _show_samples(gen, args.models, args.datasets)
        return

    gen.run(models=args.models, datasets=args.datasets)


if __name__ == "__main__":
    main()
