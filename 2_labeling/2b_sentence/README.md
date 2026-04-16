# Step 2b — Sentence Labeler

Labels individual sentences in LLM reasoning traces with safety-relevant behavioral labels. Takes `ConversationEntry` JSONL files as input, splits reasoning messages into sentences, queries an LLM judge per sentence, and writes back annotated entries with `AnnotatedSpan` objects.

## Installation

```bash
cd 2_labeling/2b_sentence
uv sync
```

## Quick start

```bash
# Label all files in the shared data directory (uses config.yaml)
uv run run.py ../../data/1_generated/

# Label a specific file
uv run run.py ../../data/1_generated/qwen3-8b_harmful.jsonl

# Use a glob pattern
uv run run.py --glob "../../data/1_generated/deepseek*.jsonl"

# Use a different config (e.g. local model via LMStudio)
uv run run.py --config lmstudio.yaml ../../data/1_generated/
```

Output lands in `data/labeled/` (configurable via config YAML) as JSONL files. Each run is checkpointed so it can be interrupted and resumed.

## Testing a new setup

Use this procedure any time you change the prompt, system prompt, or add a new feature, to verify a judge is working before running the full dataset.

**1. Extract one sample entry:**

```bash
python3 -c "
import json
with open('../../data/1_generated/deepseek-r1-distill-llama-8b-heretic_clear_harm.jsonl') as f:
    line = f.readline()
with open('/tmp/test_one_sample.jsonl', 'w') as out:
    out.write(line)
print('prompt_safety:', json.loads(line).get('metadata', {}).get('prompt_safety'))
"
```

This entry has `prompt_safety: unsafe`, so the canary string will be injected into the system prompt.

**2. Create a minimal test config** (save as e.g. `/tmp/test_gpt5mini.yaml`):

```yaml
judge:
  name: gpt-5.4-mini
  model_id: gpt-5.4-mini
  backend: openai
  api_key_source: openai-api-key
  temperature: 0.2
  max_completion_tokens: 8192
  timeout: 120

secrets_file: "/path/to/.secrets.json"
prompt_file: "/path/to/2b_sentence/prompt.md"
system_prompt_file: "/path/to/2b_sentence/system_prompt.md"

pipeline:
  max_retries: 2
  retry_delay: 2.0
  checkpoint_every: 10
  max_workers: 1
  response_sentences: 5
  skip_no_reasoning: true
  overwrite_existing: true   # re-label even if already done

mapper:
  fuzzy_threshold: 0.85
  max_lookahead_multiplier: 5
  max_lookahead_min: 500

output:
  dir: "/tmp/test_labeling_output"
  suffix: "_test"
  in_place: false
  taxonomy_version: ""
```

Important: **no `work_order_file`** — the test file won't be in the real work order.

**3. Run both judges:**

```bash
uv run run.py --config /tmp/test_gpt5mini.yaml /tmp/test_one_sample.jsonl
uv run run.py --config /tmp/test_gemini_flash_lite.yaml /tmp/test_one_sample.jsonl
```

Expected output: each judge shows `Labeled: 1`, no failures.

**4. Inspect the output:**

```bash
python3 -c "
import json, os
for f in sorted(os.listdir('/tmp/test_labeling_output')):
    if not f.endswith('.jsonl'): continue
    d = json.loads(open(f'/tmp/test_labeling_output/{f}').read())
    ann = d.get('annotations', [])
    print(f'{f}: judge={d[\"judge\"]}, spans={len(ann)}, first_labels={ann[0][\"labels\"] if ann else []}')
"
```

**What this validates:**
- System prompt is loaded and sent to the judge
- Canary is injected in the system prompt for `unsafe` entries (invisible in output, but present in the API call)
- JSON output is parsed correctly and spans are mapped back to the reasoning text
- Both OpenAI-compatible backends (openai + gemini) are working

## Multi-judge runs

All labeling runs — including comparisons across multiple judges — are configured in **YAML files**, not in Python code.

```bash
# Run all judges in compare_api.yaml through the full pipeline
uv run run.py --config compare_api.yaml ../../data/1_generated/

# Comparison mode: sample 1 entry per file, run all judges, write merged JSON
uv run run.py --config compare_api.yaml --compare-output ../../data/1_generated/

# Override seed and sample size
uv run run.py --config compare_api.yaml --compare-output --seed 99 --sample 3 ../../data/1_generated/

# Only run specific judges from a config
uv run run.py --config compare_api.yaml --judge gpt-5-mini ../../data/1_generated/
```

In multi-judge pipeline mode each judge writes its results to its own sub-directory under the output dir (e.g. `data/labeled/gpt-5-mini/`).

In comparison mode (`--compare-output`) a merged `comparison_seed<N>_<timestamp>.json` is also written, readable by `view_comparison.py`.

## Configuration

Each YAML config file has the same structure. The `judges:` list can contain one or many judges. The old singular `judge:` key still works for backward compatibility.

```yaml
# Single judge (backward-compatible)
judge:
  name: gemini-2.0-flash
  model_id: gemini-2.0-flash-001
  backend: gemini
  api_key_source: gemini-api-key
  temperature: 0.2

# Multi-judge (preferred)
judges:
  - name: gpt-5-mini
    model_id: gpt-5-mini-2025-08-07
    backend: openai
    api_key_source: openai-api-key
    temperature: 1.0
    max_completion_tokens: 8192
    timeout: 120

  - name: gemini-3-flash-preview
    model_id: gemini-3-flash-preview
    backend: gemini
    api_key_source: gemini-api-key
    temperature: 0.4
    service_tier: flex
    reasoning_effort: low

  # LMStudio local model — run.py will pause and ask you to load the model
  - name: my-local-model
    model_id: my-local-model
    backend: custom
    base_url: http://localhost:1234/v1
    api_key_source: lm-studio-key
    lmstudio_prompt: true

secrets_file: "../../.secrets.json"
prompt_file: "prompt.md"

seed: 42           # random seed for comparison-mode sampling
sample_per_file: 1 # entries to sample per file in comparison mode

pipeline:
  max_retries: 3
  max_workers: 4
  response_sentences: 5
  skip_no_reasoning: true
  max_reasoning_chars: null

output:
  dir: "data/labeled"
  comparison_json: false    # or set via --compare-output flag
```

Available configs:

| Config               | Purpose                                  |
|----------------------|------------------------------------------|
| `config.yaml`        | Single Gemini judge (default)            |
| `lmstudio.yaml`      | Single local model via LMStudio          |
| `compare_api.yaml`   | Multi-judge: GPT + Gemini (cloud APIs)   |
| `compare_local.yaml` | Multi-judge: local models via LMStudio   |

## Output format

Each output JSONL file contains `ConversationEntry` objects with `annotations` populated. Each `AnnotatedSpan` covers one sentence:

```json
{
  "id": "a3f1c8e2b9d04712",
  "messages": [...],
  "annotations": [
    {
      "text": "I should not help with this request.",
      "message_idx": 2,
      "char_start": 0,
      "char_end": 37,
      "labels": ["IV_INTEND_REFUSAL"],
      "score": 0.95
    }
  ]
}
```

## Resumability

Each output file has a companion `.checkpoint.json` file. Re-running the same command skips already-labeled entries. To re-label from scratch:

```bash
uv run run.py --reset-checkpoints ../../data/1_generated/
```

## Utility scripts

- `view_comparison.py` — render a side-by-side HTML comparison from a `labels_by_judge` JSON

## Files

| File | Purpose |
|---|---|
| `run.py` | **Main entry point** — `uv run run.py` |
| `sentence_labeler/pipeline.py` | Main orchestration: load → split → judge → map → save |
| `sentence_labeler/labeler.py` | LLM client, secret loading, prompt formatting |
| `sentence_labeler/mapper.py` | Maps LLM sentence outputs back to character spans |
| `sentence_labeler/schema.py` | Pydantic config and output models |
| `prompt.md` | LLM judge prompt template |
| `config.yaml` | Default config (single Gemini judge) |
| `lmstudio.yaml` | Local model config |
| `compare_api.yaml` | Multi-judge: cloud API models |
| `compare_local.yaml` | Multi-judge: local LMStudio models |
| `AGENTS.md` | Developer & AI agent conventions |