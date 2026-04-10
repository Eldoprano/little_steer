# Step 2b — Sentence Labeler

Labels individual sentences in LLM reasoning traces with safety-relevant behavioral labels. Takes `ConversationEntry` JSONL files as input, splits reasoning messages into sentences, queries an LLM judge per sentence, and writes back annotated entries with `AnnotatedSpan` objects.

## Installation

```bash
cd 2_labeling/2b_sentence
uv sync
```

## Quick start

```bash
# Label all files in the shared data directory
uv run sentence-labeler ../../data/1_generated/

# Label a specific file
uv run sentence-labeler ../../data/1_generated/qwen3-8b_harmful.jsonl

# Use a glob pattern
uv run sentence-labeler --glob "../../data/1_generated/deepseek*.jsonl"

# Use a different config (e.g. local model via LMStudio)
uv run sentence-labeler --config lmstudio.yaml ../../data/1_generated/
```

Output lands in `data/labeled/` (configurable via `config.yaml`) as JSONL files. Each run is checkpointed so it can be interrupted and resumed.

## Configuration

Edit `config.yaml` to set the judge backend, prompt, and output path.

```yaml
judge:
  name: gemini-2.0-flash
  model_id: gemini-2.0-flash-001
  backend: gemini                  # openai | gemini | openrouter | vllm | custom
  base_url: https://generativelanguage.googleapis.com/v1beta/openai/
  api_key_source: gemini-api-key   # key name in ../../.secrets.json
  temperature: 0.2

secrets_file: "../../.secrets.json"
prompt_file: "prompt.md"

output:
  dir: "data/labeled"
```

Available configs:
- `config.yaml` — Gemini backend (default)
- `lmstudio.yaml` — local model via LMStudio

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
      "labels": ["IV_INTEND_REFUSAL_OR_SAFE_ACTION"],
      "score": 0.95
    }
  ]
}
```

## Resumability

Each output file has a companion `.checkpoint.json` file. Re-running the same command skips already-labeled entries. To re-label from scratch, delete the checkpoint file.

## Utility scripts

- `compare_labelers.py` — compare label distributions across two output directories
- `compare_api_labelers.py` — compare outputs from different LLM backends on the same inputs
- `view_comparison.py` — render a side-by-side HTML comparison

## Files

| File | Purpose |
|---|---|
| `sentence_labeler/run.py` | CLI entry point (`sentence-labeler` command) |
| `sentence_labeler/pipeline.py` | Main orchestration: load → split → judge → map → save |
| `sentence_labeler/labeler.py` | LLM client, secret loading, prompt formatting |
| `sentence_labeler/mapper.py` | Maps LLM sentence outputs back to character spans |
| `sentence_labeler/schema.py` | Pydantic config models |
| `prompt.md` | LLM judge prompt template |
| `config.yaml` | Default config (Gemini) |
| `lmstudio.yaml` | Local model config |
