# sentence_labeler

Python package implementing the automated sentence labeling pipeline. Run it via the parent directory's `run.py`, not directly.

## Architecture

Three-stage pipeline:

```
labeler.py  →  pipeline.py  →  mapper.py
(LLM calls)   (orchestration)  (span alignment)
```

1. **labeler.py** — Formats prompts, calls the LLM judge, parses structured output
2. **pipeline.py** — Loads input JSONL, dispatches concurrent LLM calls, writes output + checkpoints
3. **mapper.py** — Maps LLM sentence outputs back to character offsets in the original reasoning text using fuzzy string matching

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Main orchestration: load → split → judge → map → save |
| `labeler.py` | LLM client, secret loading, prompt formatting, response parsing |
| `mapper.py` | Maps LLM sentence outputs back to `(char_start, char_end)` spans |
| `schema.py` | Pydantic config models (`PipelineConfig`, `JudgeConfig`, `OutputConfig`) |
| `run.py` | CLI entry point (also accessible as `sentence-labeler` after `uv sync`) |

## Usage

```bash
# From the parent directory (2b_sentence/)
uv run run.py ../../data/1_generated/
uv run run.py --config compare_api.yaml ../../data/1_generated/
```

See the parent `README.md` for full CLI documentation and config options.
