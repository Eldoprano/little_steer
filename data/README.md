# data/

This directory serves two purposes:

1. **Shared Python package** (`thesis_schema`) — the data schema used by all pipeline steps
2. **Data storage** — outputs produced by each step, organized by pipeline stage

---

## Shared Schema — `thesis_schema/`

A tiny Python package (only `pydantic` as a dependency) that defines the data structures shared across all steps.

```python
from thesis_schema import ConversationEntry, AnnotatedSpan
```

### `ConversationEntry`

One conversation with a model. Fields:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | MD5 hash of `(dataset_name, prompt_text)` — stable across runs |
| `messages` | `list[dict]` | HuggingFace-style `[{"role": ..., "content": ...}]`. Roles: `system`, `user`, `assistant`, `reasoning` |
| `annotations` | `list[AnnotatedSpan]` | Labeled spans (populated by step 2b) |
| `model` | `str` | HuggingFace model ID |
| `judge` | `str` | Labeler model used (empty before step 2b) |
| `metadata` | `dict` | Run metadata (generation params, timestamps, etc.) |

### `AnnotatedSpan`

A labeled span within a message. Fields:

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The annotated text |
| `message_idx` | `int` | Index into `messages`; `-1` = absolute position across all messages |
| `char_start` | `int` | Start character offset |
| `char_end` | `int` | End character offset |
| `labels` | `list[str]` | Assigned behavioral labels |
| `score` | `float \| None` | Confidence score (optional) |
| `meta` | `dict` | Extensible metadata |

---

## Data Directories

All data directories are gitignored. Use `LABEL_DATA_DIR` env var to point steps to a non-default location.

| Directory | Produced by | Contents |
|---|---|---|
| `1_generated/` | `1_generating/` | `{model}_{dataset}.jsonl` — raw model responses |
| `2a_evolved/` | `2_labeling/2a_evolving/` | `runs/{run_id}/state.json` — taxonomy evolution state |
| `2b_labeled/` | `2_labeling/2b_sentence/` | `{name}.jsonl` — responses with sentence-level annotations |
| `3_representations/` | `3_representations/` | `.pt` activation caches, steering vector files |

---

## Installing the schema package

Each step that needs the schema declares it as a local uv dependency:

```toml
# in each step's pyproject.toml
[tool.uv.sources]
thesis-schema = { path = "../../data" }   # adjust depth as needed
```

Then import:

```python
from thesis_schema import ConversationEntry, AnnotatedSpan
```
