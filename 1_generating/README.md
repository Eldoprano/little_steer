# Step 1 — Response Generator

Generates LLM responses from one or more models over one or more prompt datasets.
Output is saved as `ConversationEntry` JSONL files (see `data/thesis_schema`), ready for
labeling and activation extraction.

## Files

| File | Purpose |
|------|---------|
| `generate_responses.py` | Main script — importable and CLI |
| `config.yaml` | Edit this to configure your run |
| `batch_calibration.json` | Auto-generated, gitignored — caches the best batch size per model per machine |

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
gen.calibrate()                      # calibrate only
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
  batch_size: auto   # "auto" uses calibration cache; or set a fixed int
  calibrate_on_start: true
  max_retries: 3
  run_mode: sequential   # "sequential" or "interleaved"
```

**`run_mode` explained:**

- `sequential` — finishes all prompts for dataset 1, then dataset 2, etc.
- `interleaved` — round-robins one batch per dataset per cycle. Useful when you want early results across all datasets before any one is fully done.

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

## Batch size calibration

On the first run for a new model (when `batch_size: auto`), the script probes increasing batch sizes `[1, 2, 4, 8, 16, 32]` with short dummy generations until it hits OOM. The result is saved to `batch_calibration.json` keyed by `hostname/model_id/quantization`, so subsequent runs on the same machine skip calibration entirely.

To re-calibrate (e.g. after freeing VRAM or changing hardware):

```bash
rm batch_calibration.json
python generate_responses.py --config config.yaml --calibrate-only
```
