# Step 1 — Response Generator

Generates LLM responses from one or more models over one or more prompt datasets.
Output is saved as `ConversationEntry` JSONL files (see `data/thesis_schema`), ready for
labeling and activation extraction.

## Files

| File | Purpose |
|------|---------|
| `generate_responses.py` | Main script — importable and CLI |
| `config.yaml` | Edit this to configure your run |

## Quick start

```bash
cd 1_generating

# LMStudio / Ollama backend (no GPU required for this script)
uv sync
uv run python generate_responses.py

# vLLM backend (local GPU inference)
uv sync --extra vllm
uv run python generate_responses.py

# Only specific models / datasets
uv run python generate_responses.py --models gpt-oss-20b --datasets harmful
```

From a notebook or script:

```python
from generate_responses import ResponseGenerator

gen = ResponseGenerator.from_config("config.yaml")
gen.run()                            # all models & datasets
gen.run(models=["qwen3-8b"])         # one model
gen.run(datasets=["harmful"])        # one dataset
```

## Configuration

Open `config.yaml` and edit the sections below.

### Global defaults

```yaml
defaults:
  temperature: 0.6
  top_p: 0.95
  top_k: null
  min_p: null
  max_new_tokens: 2300
  quantization: null   # null | "4bit" | "8bit"
```

Every model inherits these unless it explicitly overrides a field.

### Models

**vLLM backend** (GPU inference, requires `uv sync --extra vllm`):

```yaml
models:
  - name: qwen3-8b          # short name used in output filenames
    model_id: Qwen/Qwen3-8B # HuggingFace model ID
    top_k: 20               # override a default
    system_prompt: "..."    # optional system message
```

**OpenAI-compatible backend** (LMStudio, Ollama, etc. — no GPU needed for this script):

```yaml
models:
  - name: gpt-oss-20b
    model_id: gpt-oss-20b          # must match the name LMStudio reports
    backend: openai
    openai_base_url: http://localhost:1234/v1
    openai_api_key: lm-studio      # any non-empty string
    temperature: 1
```

**Quantization options (vLLM only):**

| `quantization` value | Effect |
|---|---|
| `null` | Load as-is in float16 |
| `"4bit"` | bitsandbytes NF4 (saves ~75% VRAM vs float16) |
| `"8bit"` | bitsandbytes 8-bit (saves ~50% VRAM) |
| `null` + pre-quantized model ID | Use an AWQ/GPTQ model from HF Hub directly |

### Datasets

```yaml
datasets:
  # From HuggingFace Hub
  - name: harmful
    source: hf
    path: declare-lab/CategoricalHarmfulQA
    subset: default   # optional HF config name
    split: en
    prompt_field: Question

  # From a local file
  - name: my_prompts
    source: local
    path: ../../data/my_prompts.jsonl  # JSONL or CSV
    prompt_field: prompt
```

### Generation settings

```yaml
generation:
  write_chunk_size: 8          # prompts per batched vLLM call
  max_retries: 1               # retry passes for incomplete <think> blocks
  run_mode: sequential         # "sequential" or "interleaved"
```

**`write_chunk_size`** — prompts submitted to vLLM in one call. vLLM batches
them in parallel via PagedAttention, so larger chunks = better throughput.
Each chunk writes its results to disk on completion, so smaller chunks = faster
Ctrl+C response and less lost work if the run dies. 8 is a good default on
~16GB VRAM for reasoning models.

**`run_mode`:**

- `sequential` — finishes all prompts for dataset 1, then dataset 2, etc.
- `interleaved` — round-robins one chunk per dataset per cycle. Useful when you want early results across all datasets before any one is fully done.

## Output format

One JSONL file per `(model, dataset)` pair under `data/` (or `../data/1_generated/` if you use the shared data directory):

```
data/qwen3-8b_harmful.jsonl
data/r1-8b_harmful.jsonl
```

Each line is a `ConversationEntry`:

```json
{
  "id": "a3f1c8e2b9d04712",
  "messages": [
    {"role": "system",    "content": "..."},
    {"role": "user",      "content": "How do I make ...?"},
    {"role": "reasoning", "content": "Let me think about this..."},
    {"role": "assistant", "content": "I cannot help with that."}
  ],
  "model": "Qwen/Qwen3-8B",
  "judge": "",
  "metadata": {
    "run_id": "...", "machine_id": "...", "generated_at": "2026-03-31T...",
    "model_name": "qwen3-8b", "quantization": null,
    "max_new_tokens": 2300, "temperature": 0.6, "top_p": 0.95,
    "finish_reason": "eos", "n_new_tokens": 847, "has_reasoning": true,
    "dataset_name": "harmful", "prompt_id": "a3f1c8e2b9d04712",
    ...
  }
}
```

**`id` / `prompt_id`** is an MD5 hash of `(dataset_name, prompt_text)` — stable across models and machines, so you can join responses for the same prompt from different runs.

**`finish_reason`** values:

| Value | Meaning |
|---|---|
| `"eos"` | Model ended naturally with an EOS token |
| `"max_length"` | Hit `max_new_tokens` — response may be cut off |
| `"incomplete_thinking"` | `<think>` opened but `</think>` never appeared (retried) |
| `"failed"` | Generation error |

## Resumability

The script checks which `id`s are already present in the output file before generating. Re-running the same command simply skips already-completed prompts. This means:

- You can stop and restart at any time.
- Multiple machines can write to separate output files and the files can be merged later by concatenating the JSONL lines (dedup by `id` if needed).

## Quality control

Every entry gets an `approved` flag in its metadata (`true`/`false`) based on
cheap checks performed at write time: `max_length`, `failed`, empty response,
or a bare `[/THINK]` prefix (LMStudio artifact).

`fix_quality.py` also detects a `foreign_script` issue (≥ 5 characters from
CJK, Arabic, Cyrillic, Devanagari, or Hangul ranges). This is **flagged but
not yet blocking** — it may reflect intentional behaviour (model reasoning in
its training language) rather than corruption. Needs per-model investigation
before deciding whether to auto-remove or route to manual review.

`generate_responses.py` deliberately does **not** regenerate bad entries — that
work belongs to `fix_quality.py`, which can also run an expensive n-gram
repetition audit offline. Typical workflow after a generation run:

```bash
uv run python fix_quality.py --tag --fix        # tag approved + fix artifacts
uv run python fix_quality.py --remove           # delete bad entries
uv run python generate_responses.py             # re-run to fill the gaps
```

The labeling pipeline (`2b_sentence`) skips any entry where `approved` is
explicitly `false`.

**`--sync-labeled` criterion:** Since the labeling pipeline labels reasoning sentences, only entries with **broken reasoning** are purged — specifically: repetition in reasoning, `failed`, missing reasoning, or `max_length` where the reasoning itself was cut off (response absent/tiny). Entries where `max_length` only truncated the response (reasoning was complete) and entries with `empty_response` or `think_artifact` alone are **kept**, because their reasoning labels are still valid.

### Mistral reasoning prefill

Models with `think_prefix: true` in `config.yaml` have `[THINK]` prepended as an assistant prefill on every generation call. This forces the model to open a reasoning block before answering, preventing the common failure mode where the model skips straight to the answer.

### `fix_quality.py` — standalone audit and repair

Use this to inspect and clean up existing files without re-running the full generator:

```bash
# Audit only — prints a table of issue counts per file
uv run python fix_quality.py

# Filter to one model
uv run python fix_quality.py --model ministral

# Add/update the approved field on all entries
uv run python fix_quality.py --tag

# Fix [/THINK] artifacts in-place (no regeneration needed)
uv run python fix_quality.py --fix

# Remove entries from 2b_labeled where the *response* is broken
# (max_length, failed, empty, or think_artifact — NOT reasoning-only repetition)
# Shows a preview table and asks for confirmation before deleting.
uv run python fix_quality.py --sync-labeled

# Delete bad entries from 1_generated so they get regenerated on the next run
uv run python fix_quality.py --remove

# Preview any operation without writing
uv run python fix_quality.py --tag --fix --sync-labeled --remove --dry-run
```

Typical cleanup workflow for existing data:

```bash
uv run python fix_quality.py --tag --fix        # tag all + fix artifacts
uv run python fix_quality.py --sync-labeled     # clean 2b_labeled
uv run python fix_quality.py --remove           # clear bad entries
# then re-run generate_responses.py to fill the gaps
```

