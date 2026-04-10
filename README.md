# little_steer — Thesis Code

Research code for the master's thesis *"Monitoring What Models Think: Steering Vectors for AI Safety and Control"* (Hochschule Kempten, 2026).

The core idea: use Representation Engineering (RepE) to extract steering vectors from reasoning model activations to detect and steer safety-relevant behaviours — not from text output, but from internal activations.

---

## Repository Structure

```
little_steer/
├── data/                       # Shared data schema + pipeline outputs
│   ├── thesis_schema/          # Shared Python package: ConversationEntry, AnnotatedSpan
│   ├── 1_generated/            # JSONL outputs from step 1 (gitignored)
│   ├── 2a_evolved/             # Taxonomy evolution outputs (gitignored)
│   ├── 2b_labeled/             # Sentence labeler outputs (gitignored)
│   └── 3_representations/      # Activation caches, steering vectors (.pt files, gitignored)
│
├── 1_generating/               # Step 1 — Generate model responses
├── 2_labeling/
│   ├── 2a_evolving/            # Step 2a — Evolve a label taxonomy via LLM
│   └── 2b_sentence/            # Step 2b — Label sentences with the taxonomy
└── 3_representations/          # Step 3 — Extract activations, build and evaluate steering vectors
```

Each numbered directory is an independent uv project with its own `.venv`. The `data/` directory holds the shared schema (imported as `thesis_schema`) and the output data produced by each step.

---

## Pipeline

```
Prompts (HuggingFace datasets)
        │
        ▼
  1_generating/          →  data/1_generated/*.jsonl
        │
        ▼
  2_labeling/2a_evolving/  →  data/2a_evolved/  (taxonomy discovery, optional)
  2_labeling/2b_sentence/  →  data/2b_labeled/*.jsonl  (sentence-level annotations)
        │
        ▼
  3_representations/     →  data/3_representations/  (activations, vectors, metrics)
```

---

## Setup

Each step manages its own environment. `uv` is required.

```bash
# Step 1 — response generation (needs vLLM + GPU)
cd 1_generating && uv sync

# Step 2a — label evolution (LLM API calls only)
cd 2_labeling/2a_evolving && uv sync

# Step 2b — sentence labeler (LLM API calls only)
cd 2_labeling/2b_sentence && uv sync

# Step 3 — steering vectors (needs torch + GPU)
cd 3_representations && uv sync --extra ml
```

### Secrets

API keys live in `.secrets.json` at the repo root (gitignored):

```json
{
    "hf_token": "hf_...",
    "gemini-api-key": "...",
    "openai-api-key": "..."
}
```

---

## Steps

### Step 1 — `1_generating/`

Generates responses from one or more LLMs across safety-relevant prompt datasets. Supports vLLM (local GPU) and any OpenAI-compatible server (LMStudio, Ollama). Outputs `ConversationEntry` JSONL files with separate `reasoning` and `assistant` message roles.

```bash
cd 1_generating
uv run python generate_responses.py --config config.yaml
```

Output: `data/1_generated/{model}_{dataset}.jsonl`

### Step 2a — `2_labeling/2a_evolving/`

Discovers and evolves a taxonomy of safety-relevant behaviours by iteratively labelling paragraphs from the generated responses with an LLM. Supports CREATE / MERGE / SPLIT / RENAME / DELETE operations on the taxonomy.

```bash
cd 2_labeling/2a_evolving
uv run python run.py evolve --steps 200
uv run python run.py status
```

### Step 2b — `2_labeling/2b_sentence/`

Applies a fixed label taxonomy at sentence level to annotate each sentence in the reasoning traces. Produces `AnnotatedSpan`-enriched JSONL files. Supports multiple LLM backends.

```bash
cd 2_labeling/2b_sentence
uv run sentence-labeler --config config.yaml --input ../../data/1_generated/
```

Output: `data/2b_labeled/{name}.jsonl`

### Step 3 — `3_representations/`

Core ML library. Loads annotated JSONL datasets, runs forward passes through a model (via nnterp/nnsight), extracts layer activations, and builds/evaluates steering vectors.

See [`3_representations/README.md`](3_representations/README.md) for the full API reference.

```bash
cd 3_representations
uv sync --extra ml
uv run python scripts/full_run.py
```

### Shared Schema — `data/thesis_schema/`

Pydantic models shared across all steps:

```python
from thesis_schema import ConversationEntry, AnnotatedSpan
```

- **`ConversationEntry`** — one conversation: messages (system/user/assistant/reasoning) + annotations
- **`AnnotatedSpan`** — a labeled span within a message: `text`, `char_start`, `char_end`, `labels`, `score`

---

## Data Path Override

If you keep generated data elsewhere, set an environment variable before running step 2a:

```bash
LABEL_DATA_DIR=/path/to/your/jsonl/files uv run python run.py evolve
```
