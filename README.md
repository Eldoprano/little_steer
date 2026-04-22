# little_steer — Thesis Code

Research code for the master's thesis *"Monitoring What Models Think: Steering Vectors for AI Safety and Control"* (Hochschule Kempten, 2026).

The repository now uses a single canonical dataset during development:

- `data/dataset.jsonl` is the source of truth
- generation writes into it
- labeling appends `label_runs` into it
- safety scoring appends `safety_runs` into it
- downstream representation code still reads top-level `annotations`

Parquet or other exports should be treated as build artifacts, not the editable store.

## Repository Structure

```text
little_steer/
├── data/
│   ├── thesis_schema/          # Shared schema package
│   ├── dataset.jsonl           # Canonical development dataset
│   ├── migrate_to_canonical.py # Legacy → canonical migration
│   ├── dataset_health.py       # Consistency checker for canonical dataset
│   └── export_hf.py            # Export helpers
├── 1_generating/               # Response generation + quality + safety scoring
├── 2_labeling/
│   ├── 2a_evolving/            # Taxonomy evolution
│   ├── 2b_sentence/            # Sentence labeling + viewer
│   └── 2c_human_labeling/      # Human labeling frontend
└── 3_representations/          # Activations, vectors, evaluation
```

## Pipeline

```text
Prompts
  │
  ▼
1_generating/generate_responses.py
  │
  ▼
data/dataset.jsonl
  ├── generation content + metadata
  ├── label_runs
  ├── safety_runs
  └── top-level active annotations/judge for downstream compatibility
  │
  ├── 2_labeling/2b_sentence/run.py
  ├── 1_generating/safety_scoring/score.py
  ├── 1_generating/fix_quality.py
  ├── data/dataset_health.py
  └── 3_representations/
```

## Setup

Each numbered directory is its own `uv` project.

```bash
cd 1_generating && uv sync
cd 2_labeling/2a_evolving && uv sync
cd 2_labeling/2b_sentence && uv sync
cd 3_representations && uv sync --extra dev --extra ml
```

## Main Commands

### 1. Generate responses

Writes or replaces canonical entries in `data/dataset.jsonl`.

```bash
cd 1_generating
uv run python generate_responses.py --config config.yaml
```

Entry identity is sample-level: `(dataset, prompt, model)`.

### 2. Label reasoning traces

For the canonical workflow, point the labeler at `data/dataset.jsonl`. It updates entries in place, appending a matching `label_run` and mirroring the active run to top-level `annotations` and `judge`.

```bash
cd 2_labeling/2b_sentence
uv run run.py ../../data/dataset.jsonl
```

### 3. Score safety

Appends or updates `safety_runs` in the same canonical dataset.

```bash
cd 1_generating/safety_scoring
uv run python score.py
```

### 4. Audit quality

Stores quality findings in `metadata.quality` and mirrors the blocking decision to `metadata.approved`.

```bash
cd 1_generating
uv run python fix_quality.py --tag
uv run python fix_quality.py --fix
```

### 5. Check dataset health

Validates schema, run hashes, active-label consistency, and span integrity.

```bash
uv run --with pydantic --with rich python data/dataset_health.py
```

### 6. Migrate legacy data

Builds `data/dataset.jsonl` from old generated/labeled folders and writes a migration report.

```bash
uv run --with pydantic --with rich python data/migrate_to_canonical.py
```

## Canonical Dataset Shape

The shared schema lives in `data/thesis_schema/`.

`ConversationEntry` now includes:

- `id`: stable sample id for `(dataset, prompt, model)`
- `messages`: `system` / `user` / `reasoning` / `assistant`
- `annotations`: active spans for downstream readers
- `judge`: active judge for downstream readers
- `metadata.prompt_id`: stable prompt id for grouping across models
- `label_runs`: all labeler runs kept on the entry
- `safety_runs`: all safety scorer runs kept on the entry

If an entry is regenerated, derived fields are intentionally reset and the entry naturally re-enters labeling/scoring queues.

## Current State

The branch already contains:

- schema support for `label_runs` and `safety_runs`
- generator output to `data/dataset.jsonl`
- safety scoring against the canonical dataset
- sentence labeler support for in-place canonical updates
- viewer and frontend type updates for the expanded schema
- migration and health-check scripts

## Notes

- Some legacy helper scripts and comments still reference `data/1_generated` or `data/2b_labeled`; the main active flow is now the canonical dataset.
- `3_representations` remains compatible because it still consumes top-level `annotations`.
