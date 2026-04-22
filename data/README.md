# data/

This directory contains both the shared schema package and the canonical development dataset.

## Canonical Store

The primary editable dataset is:

- `data/dataset.jsonl`

This file is the source of truth for development. It replaces the old staged workflow where generation and labeling lived in separate JSONL trees.

Related utilities:

- `dataset_health.py` checks canonical consistency
- `migrate_to_canonical.py` builds the canonical file from legacy outputs
- `export_hf.py` exports the canonical dataset for downstream use

## Shared Schema

Import from:

```python
from thesis_schema import AnnotatedSpan, ConversationEntry, LabelRun, SafetyRun
```

### `ConversationEntry`

Core fields:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Stable sample id for `(dataset_name, prompt_text, model_id)` |
| `messages` | `list[dict]` | Message list with roles `system`, `user`, `reasoning`, `assistant` |
| `annotations` | `list[AnnotatedSpan]` | Active spans mirrored from the active `label_run` |
| `model` | `str` | Generator model id |
| `judge` | `str` | Active judge mirrored from the active `label_run` |
| `metadata` | `dict` | Prompt ids, generation metadata, quality state, active run metadata |
| `label_runs` | `list[LabelRun]` | All labeling runs stored on the entry |
| `safety_runs` | `list[SafetyRun]` | All safety-scoring runs stored on the entry |

Important metadata:

| Key | Description |
|---|---|
| `metadata.prompt_id` | Stable prompt id for `(dataset_name, prompt_text)` |
| `metadata.generation_hash` | Hash of the current reasoning content |
| `metadata.quality` | Quality findings and approval decision |
| `metadata.active_label_run` | Canonical key for the currently mirrored label run |

### `AnnotatedSpan`

| Field | Type | Description |
|---|---|---|
| `text` | `str` | The annotated text |
| `message_idx` | `int` | Index into `messages` |
| `char_start` | `int` | Start character offset |
| `char_end` | `int` | End character offset |
| `labels` | `list[str]` | Assigned behavior labels |
| `score` | `float` | Judge-provided score / safety weight |
| `meta` | `dict` | Extra metadata such as sentence index or original judge text |

### `LabelRun`

Stores one labeling pass for a specific `(judge_name, taxonomy_version, generation_hash)`.

Includes:

- judge identity
- taxonomy version
- generation hash
- assessment
- raw sentence annotations
- mapped spans
- token usage / finish metadata

### `SafetyRun`

Stores one safety-scoring pass for a specific `(guard_name, generation_hash)`.

Includes:

- guard identity
- generation hash
- scored timestamp
- structured result payload

## Development Model

Recommended workflow:

1. Generate into `data/dataset.jsonl`
2. Label in place
3. Score safety in place
4. Run `dataset_health.py`
5. Export to other formats only when needed

Parquet should be treated as export-only, not as the editable source of truth.
