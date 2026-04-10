# Step 2a — Label Evolution

A CLI tool for incrementally discovering and evolving a taxonomy of behavioral labels for LLM reasoning traces, used in AI safety research.

## What it does

Given a dataset of LLM reasoning traces (responses to harmful prompts), this system:
1. Samples one sub-text (paragraph) from the dataset at a time
2. Sends it to an LLM labeler with the current taxonomy
3. Receives label assignments and an optional taxonomy operation (CREATE, MERGE, SPLIT, RENAME, DELETE)
4. Applies the operation if valid, saves state, and continues

The goal is to converge on a stable taxonomy that describes what reasoning models do when they encounter harmful prompts.

## Data findings

The data lives in `../../data/1_generated/` (or set `LABEL_DATA_DIR` env var to override) as JSONL files, one per model-dataset combination. Each record contains a conversation with `user`, `reasoning`, and `assistant` roles.

| Model | Sub-texts | Notes |
|---|---|---|
| deepseek-r1-distill-llama-8b | ~39k | Conversational reasoning, short-to-medium paragraphs |
| qwen3.5-4B / qwen3.5-9B | ~20k each | Structured, often numbered reasoning |
| phi-4-reasoning-14B | ~24k | Extended deliberative reasoning, longest paragraphs |
| ministral-3-8B-Reasoning-2512 | ~10k | Only records with a `reasoning` role are used |

**Split strategy:** All models use the same approach — find the `role: "reasoning"` message and split its content by blank lines (`\n\n`). Paragraphs under 15 characters are dropped. Records without a reasoning role are silently skipped (this affects most Ministral records).

## Taxonomy stability design

The most important design constraint is that the taxonomy must stabilize over time, not churn endlessly. This is enforced by:

- **NONE is the default** — the LLM prompt makes explicit that NONE is always correct unless the taxonomy *genuinely cannot* label the text
- **Operations require all-or-nothing justification** — the prompt lists 4 conditions that must all be true before proposing a change
- **Structural validation** — proposed operations are validated against the current taxonomy before applying (e.g., CREATE rejected if label already exists or taxonomy is full)
- **10% revisit sampling** — already-seen sub-texts are re-processed occasionally; if the taxonomy handles them correctly without change, that's a convergence signal

## Installation

```bash
cd 2_labeling/2a_evolving
uv sync
```

## CLI commands

### `run` — run the evolution loop

```bash
uv run python run.py run \
  --models deepseek qwen \
  --labeler anthropic/claude-haiku-4-5-20251001 \
  --steps 100 \
  --max-labels 20
```

Options:
- `--models` — one or more substrings matched against JSONL filenames (case-insensitive). `deepseek` matches all 10 deepseek files including heretic variants.
- `--labeler` — OpenAI-compatible model string (e.g. `gpt-4o-mini`, `anthropic/claude-...` if using a proxy, etc.)
- `--steps` — steps to run (if resuming, adds on top of existing steps)
- `--max-labels` — cap on active labels; CREATE is blocked when reached
- `--seed path/to/seed.json` — optional initial taxonomy (see seed format below)
- `--run-id ID` — resume an existing run

Example with LMStudio (OpenAI-compatible local server):
```bash
OPENAI_BASE_URL=http://localhost:1234/v1 OPENAI_API_KEY=lm-studio \
  LABEL_DATA_DIR=../../1_generating/data \
  uv run python run.py run --models deepseek --labeler gpt-oss-20b --steps 50
```

Example with Ollama:
```bash
uv run python run.py run --models deepseek --labeler ollama/qwen3:8b --steps 50
```

Example resuming:
```bash
uv run python run.py run --models deepseek qwen --labeler anthropic/claude-haiku-4-5-20251001 --steps 100 --run-id abc123def456
```

### `visualize` — show taxonomy state

```bash
uv run python run.py visualize [--run-id ID]
```

Shows the active taxonomy with usage counts, run statistics, and recent operations. Defaults to the most recent run.

### `list` — list all runs

```bash
uv run python run.py list
```

### `inspect` — explore a model's data

```bash
uv run python run.py inspect --model phi [--run-id ID]
```

Shows:
- Matched JSONL files and their split strategy
- Sub-text count and length statistics
- (With `--run-id`) label distribution and operations triggered by this model

## State file structure

Each run creates `runs/{run_id}/state.json`:

```json
{
  "run_id": "abc123def456",
  "created_at": "2026-04-03T17:00:00Z",
  "config": {
    "models": ["deepseek-r1-distill-llama-8b_xs_test"],
    "labeler": "anthropic/claude-haiku-4-5-20251001",
    "max_labels": 20,
    "seed_file": null,
    "sampling_seed": 42
  },
  "taxonomy": {
    "active": {
      "Label Name": {
        "label_id": "abc12345",
        "name": "Label Name",
        "description": "...",
        "created_at": "...",
        "usage_count": 14
      }
    },
    "graveyard": {}
  },
  "history": {
    "labels_ever": {},
    "operations": [
      {
        "step": 3,
        "operation": "CREATE",
        "details": {"name": "...", "description": "..."},
        "triggered_by": {
          "text": "excerpt...",
          "composite_id": "deepseek-r1-distill-llama-8b_xs_test::8143ff374b97f57e",
          "sub_text_idx": 2
        },
        "justification": "No existing label covers..."
      }
    ],
    "snapshots": [
      {"step": 0, "taxonomy": {...}},
      {"step": 25, "taxonomy": {...}}
    ]
  },
  "processed": {
    "deepseek-r1-distill-llama-8b_xs_test::8143ff374b97f57e": {
      "sub_texts": [
        {"idx": 0, "times_processed": 1, "triggered_change": false, "labels": ["Label A"]}
      ]
    }
  },
  "stats": {
    "steps_completed": 100,
    "total_revisits": 10,
    "total_changes": 7,
    "errors": 0,
    "total_invalid_proposals": 2
  }
}
```

Key design decisions:
- **Atomic writes**: state is written to `.tmp` then renamed via `os.replace()` — safe against crashes
- **Composite IDs**: `"{model_stem}::{sample_id}"` — needed because the same sample ID appears across different model files (they share the same original prompt)
- **Snapshots every 25 steps**: full taxonomy captured for animation/replay without replaying all operations
- **`labels_ever`**: all labels including deleted/merged ones, keyed by `label_id`, for lineage tracking
- **`graveyard`**: deleted/merged/renamed labels stay accessible for audit

## Seed file format

```json
{
  "label_id_or_any_key": {
    "name": "Action Planning & Design",
    "description": "Developing comprehensive plans or sequences of actions.",
    "usage_count": 32
  }
}
```

Any JSON file in the seed format above can be passed as `--seed`.

## Design decisions

**Paragraph splitting over sentences:** Paragraphs preserve coherent thought units better than sentences. The reasoning in these models tends to shift topic between blank-line-separated paragraphs. Sentence splitting was rejected as too granular (loses context) and too many per trace.

**Single state file per run:** Simple, atomic, easy to inspect. No database needed for this scale.

**OpenAI-Compatible Client:** Uses the official OpenAI Python client, supporting any OpenAI-compatible backend (Anthropic via proxy, LMStudio, Ollama, etc.) by setting `OPENAI_BASE_URL`.

**NONE-first prompt design:** The labeler prompt was designed around the insight that small models will propose changes at every step if not constrained. The prompt lists explicit conditions for when a change is allowed, and states that NONE is "always the default." This is the most important correctness property of the system.

## Known limitations

- **No retroactive re-labeling**: when a label is renamed or merged, previously assigned labels in `processed` become stale. The taxonomy history captures the change but old assignments are not updated.
- **`--models` is a substring match**: `--models deepseek` matches heretic and self-heretic variants. Use a more specific substring (e.g. `deepseek-r1-distill-llama-8b_`) to restrict to exact files.
- **Ministral coverage**: most Ministral records lack a `reasoning` role and are skipped. Only ~10k sub-texts are available from this model versus ~20-40k for others.
- **No deduplication across files**: the same reasoning content may appear in multiple dataset splits for the same model. The `composite_id` includes the filename stem, so these are treated as separate samples.
