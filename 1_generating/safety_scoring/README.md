# Safety Scoring

Adds safety scores to every entry in `data/1_generated/` using two guard models,
then propagates those scores to all matching `data/2b_labeled/` files.

## Models

| Guard | Model | Output |
|---|---|---|
| **WildGuard** | `allenai/wildguard` | `prompt_harmfulness`, `response_harmfulness`, `response_refusal` |
| **Qwen3Guard** | `Qwen/Qwen3Guard-Gen-8B` | `prompt_safety`, `prompt_categories`, `response_safety`, `response_categories`, `response_refusal` |

Both run via **vLLM with bitsandbytes 4-bit quantization** (~4–5 GB each) so they fit in 16 GB VRAM.
They are loaded and unloaded sequentially — never both in memory at once.

## Installation

From `1_generating/`, run:

```bash
uv sync
```

`wildguard` is listed as a dependency in `pyproject.toml`.  
`vllm` and `transformers` must be installed (already in the environment).

## Usage

```bash
cd 1_generating

# Score everything with both guards (default)
uv run python safety_scoring/score.py

# Preview what would be scored without running anything
uv run python safety_scoring/score.py --dry-run

# One guard at a time
uv run python safety_scoring/score.py --guard wildguard
uv run python safety_scoring/score.py --guard qwen3guard

# Only score files matching a pattern (fnmatch on the file stem)
uv run python safety_scoring/score.py --files "gpt-oss*"
uv run python safety_scoring/score.py --files "*_lima"

# Tune batch size and save frequency
uv run python safety_scoring/score.py --batch-size 16 --save-every 30
```

## Pause & Resume

- **Ctrl+C once** — finishes the current batch, saves checkpoint, then exits cleanly.
- **Ctrl+C twice** — force exits immediately (current batch is lost but previous saves are safe).
- Re-running the script automatically skips already-scored entries by reading the checkpoint files.

## Output format

Safety scores are stored under `metadata.safety_scores` in each JSONL entry:

```json
{
  "id": "8143ff374b97f57e",
  "metadata": {
    "safety_scores": {
      "wildguard": {
        "prompt_harmfulness": "unharmful",
        "response_harmfulness": "unharmful",
        "response_refusal": "compliance",
        "is_parsing_error": false,
        "scored_at": "2026-04-20T12:00:00+00:00"
      },
      "qwen3guard": {
        "prompt_safety": "Safe",
        "prompt_categories": ["None"],
        "response_safety": "Safe",
        "response_categories": ["None"],
        "response_refusal": "No",
        "scored_at": "2026-04-20T12:01:00+00:00"
      }
    }
  }
}
```

`response_*` fields are `null` for entries that have no assistant response.

## Checkpoints

Progress is saved per `(source_file_stem, guard)` in:

```
data/1_generated/.safety_scores/
  {model}_{dataset}_wildguard.checkpoint.json
  {model}_{dataset}_qwen3guard.checkpoint.json
```

Each checkpoint stores `scored_ids` (list) and `failed_ids` (dict with error reason).
