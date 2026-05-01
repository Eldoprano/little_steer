# little_steer

A Python library for extracting activations from transformer models, building steering vectors, and probing/visualizing safety-relevant behaviour in reasoning models.

Built on top of [nnterp](https://github.com/nnsight/nnterp) and [nnsight](https://nnsight.net/). Model-agnostic: works with any HuggingFace causal LM.

---

## Installation

```bash
cd 3_representations

# With ML dependencies (torch, nnsight, nnterp, transformers, etc.)
uv sync --extra ml

# Schema/pydantic only (no GPU needed — for importing ConversationEntry elsewhere)
uv sync
```

**ML dependencies (optional extra):** `torch`, `transformers`, `nnsight`, `nnterp`, `scikit-learn`, `numpy`, `matplotlib`, `tqdm`, `rich`

**Core dependency (always installed):** `thesis-schema` (pydantic models shared across all steps)

---

## Quick Start

```python
import little_steer as ls

# 1. Load dataset
entries = ls.load_dataset("dataset.jsonl")

# 2. Load model
model = ls.LittleSteerModel("Qwen/Qwen3-8B")

# 3. Define extraction plan
plan = ls.ExtractionPlan("my_plan", specs={
    "last_token":     ls.ExtractionSpec(ls.TokenSelection("last"), layers=list(range(10, 32))),
    "whole_sentence": ls.ExtractionSpec(ls.TokenSelection("all"),  layers=list(range(10, 32))),
})

# 4. Extract activations (one forward pass per conversation)
extractor = ls.ActivationExtractor(model)
result = extractor.extract(entries, plan)

# 5. Build steering vectors
builder = ls.SteeringVectorBuilder()
vectors = builder.build(result, target_label="II_STATE_SAFETY_CONCERN",
                        methods=["mean_centering", "pca"],
                        baseline_label="VI_NEUTRAL_FILLER_TRANSITION")

# 6. Score across layers to find the best layer
scores = ls.score_dataset(model, entries, vectors)
best = max(scores, key=lambda s: s.discrimination)
print(f"Best layer: {best.layer}  discrimination={best.discrimination:+.3f}")

# 7. Visualize token-level similarity in a notebook
display(ls.show_token_similarity(model, entries[0], vectors.vectors[0]))

# 8. Evaluate with full metrics
results = ls.evaluate_dataset(model, entries, vectors, aggregation="mean")
fig, ax = ls.plot_layer_metrics(results, metric="auroc")

# 9. Steer generation
output = ls.steered_generate(model, messages, vectors.vectors[0], alpha=15.0)
```

---

## Data Format

### Loading and saving

```python
entries = ls.load_dataset("dataset.jsonl")      # List[ConversationEntry]
ls.save_dataset(entries, "out.jsonl")
for entry in ls.iter_dataset("dataset.jsonl"):  # streaming
    ...

# Convert from old JSON format
entries = ls.convert_file("old.json", "new.jsonl")
```

### ConversationEntry

```python
entry = ls.ConversationEntry(
    id="conv_001",
    messages=[
        {"role": "user",      "content": "How do I..."},
        {"role": "reasoning", "content": "The user is asking about..."},
        {"role": "assistant", "content": "Here's how..."},
    ],
    annotations=[
        ls.AnnotatedSpan(
            text="The user is asking about...",
            message_idx=1,        # which message (0-indexed)
            char_start=0,
            char_end=27,
            labels=["I_REPHRASE_PROMPT"],
            score=2.0,            # optional labeler confidence
        )
    ],
    model="qwen3-8b",
    judge="gemini-2.0-flash",
    metadata={"source": "strongreject"},
)
```

**Roles:** `"user"`, `"assistant"`, `"reasoning"` (mapped to `<think>` blocks for Qwen/DeepSeek), `"system"`

**AnnotatedSpan fields:**
- `message_idx`: Index into `messages`. Use `-1` for absolute char positions across concatenated content.
- `char_start`, `char_end`: Character offsets within the message's content.
- `labels`: List of category strings.
- `score`: Optional float (e.g. labeler confidence or agreement score).

---

## Model Loading

```python
model = ls.LittleSteerModel("Qwen/Qwen3-8B")

# Custom chat template
model = ls.LittleSteerModel("my-model", custom_chat_template="{% for m in messages %}...")

# Multimodal / unusual config
model = ls.LittleSteerModel("some-model", use_pretrained_loading=True)

# Key properties
model.num_layers    # int
model.hidden_size   # int
model.tokenizer     # HuggingFace tokenizer

# Formatting
text = model.format_messages(entry.messages)
text, offsets = model.format_messages_with_offsets(entry.messages)
# offsets: {message_idx: char_offset_in_formatted_text}
```

### VRAM Management

```python
vram = ls.VRAMManager()
cfg = vram.estimate_batch_config(model)
print(cfg.batch_size, cfg.max_seq_len, cfg.quantization)
```

---

## Activation Extraction

### TokenSelection — how to pick tokens from a span

```python
# Strategy options: "first", "last", "all", "first_n", "last_n"
ls.TokenSelection("last")
ls.TokenSelection("first_n", n=3)
ls.TokenSelection("all", bleed_before=5, bleed_after=5)  # extend window

# Aggregation (how to collapse selected tokens into one vector):
#   "mean"                  — average (default)
#   "none"                  — keep all as (n_tokens, hidden_dim) matrix
#   "top_confident_mean"    — mean over tokens with top-1 next-token prob >= threshold
#   "entropy_weighted_mean" — weighted mean, weight = 1/entropy
#   "decision_point_mean"   — mean over highest-entropy tokens (model choice points)
ls.TokenSelection("all", aggregation="top_confident_mean", confidence_threshold=0.5)
ls.TokenSelection("all", aggregation="entropy_weighted_mean")
ls.TokenSelection("all", aggregation="decision_point_mean", high_entropy_fraction=0.3)

# Bleed (window shaping before strategy):
#   positive bleed_before: extend window left by N tokens (add preceding context)
#   negative bleed_before: skip the first N tokens of the span
#   Same for bleed_after
ls.TokenSelection("first", bleed_before=-2)      # skip 2, then take first
ls.TokenSelection("all", bleed_before=5)         # extend 5 left, take all
```

### ExtractionSpec and ExtractionPlan

```python
spec = ls.ExtractionSpec(
    token_selection=ls.TokenSelection("all"),
    layers=[20, 25, 30],    # None = all layers
)

plan = ls.ExtractionPlan("experiment_v1", specs={
    "last_token":     ls.ExtractionSpec(ls.TokenSelection("last"),  layers=[20, 25]),
    "whole_sentence": ls.ExtractionSpec(ls.TokenSelection("all"),   layers=[20, 25]),
    "decision_pts":   ls.ExtractionSpec(
        ls.TokenSelection("all", aggregation="decision_point_mean"), layers=[20, 25]
    ),
})

# All specs share ONE forward pass per conversation — efficiency win
plan.add_spec("extra", ls.ExtractionSpec(ls.TokenSelection("first"), layers=[15]))
plan.label_filter = ["II_STATE_SAFETY_CONCERN"]  # only extract these labels
```

### ActivationExtractor

```python
extractor = ls.ActivationExtractor(model, max_seq_len=4096)
result = extractor.extract(entries, plan, show_progress=True)
print(result.summary())

# Access extracted activations
acts = result.get("whole_sentence", "I_REPHRASE_PROMPT", layer=20)
# acts: List[torch.Tensor], each (hidden_dim,)

# Iterate all
for spec_name, label, layer, acts in result.iter_all():
    print(spec_name, label, layer, len(acts))

# Save / load
result.save("result.pt")
result = ls.ExtractionResult.load("result.pt")
```

**ExtractionResult** properties: `.specs()`, `.labels()`, `.layers()`, `.count(spec, label, layer)`

---

## Steering Vectors

### Building

```python
builder = ls.SteeringVectorBuilder()

# For one label, all (spec, method, layer) combinations
vectors = builder.build(
    result,
    target_label="II_STATE_SAFETY_CONCERN",
    methods=["mean_centering", "pca", "mean_difference", "linear_probe"],
    baseline_label="VI_NEUTRAL_FILLER_TRANSITION",  # required for mean_difference, linear_probe
    pca_components=1,
    probe_C=1.0,
)

# For all labels at once
all_vecs = builder.build_all_labels(result, methods=["pca"])
# Returns {label: SteeringVectorSet}
```

**Methods:**
| Method | Description | Requires baseline? |
|---|---|---|
| `"mean_centering"` | target mean − centroid of all other categories | No |
| `"pca"` | Top PCA component of target activations | No |
| `"mean_difference"` | target mean − baseline mean | Yes |
| `"linear_probe"` | Logistic regression weight vector | Yes |

### SteeringVector and SteeringVectorSet

```python
vec = vectors.vectors[0]
vec.vector          # torch.Tensor (hidden_dim,)
vec.layer           # int
vec.label           # str
vec.method          # str
vec.extraction_spec # str
vec.metadata        # dict (n_samples, probe_accuracy, etc.)
vec.normalized()    # returns copy with L2-normalised vector

# Filter (all args optional, combined with AND)
pca_vecs = vectors.filter(method="pca")
layer_20  = vectors.filter(layer=20)
combined  = vectors.filter(method="pca", spec="last_token")

# Group
by_layer  = vectors.group_by("layer")    # {layer_idx: SteeringVectorSet}
by_method = vectors.group_by("method")

# Info
vectors.labels()    # ['II_STATE_SAFETY_CONCERN', ...]
vectors.methods()   # ['pca', 'mean_centering', ...]
vectors.layers()    # [20, 25, 30, ...]
vectors.specs()     # ['last_token', 'whole_sentence', ...]
len(vectors)
print(vectors.summary())

# Save / load
vectors.save("vectors.pt")
vectors = ls.SteeringVectorSet.load("vectors.pt")
```

---

## Probing — Reading Vector Signal

### probe_text

Run one forward pass on a text string and compute cosine similarity against a vector set.

```python
text = model.format_messages([{"role": "user", "content": "..."}], add_generation_prompt=True)
sims = ls.probe_text(model, text, vectors.filter(method="pca"))
# Returns: {(label, method, layer): float}

for (label, method, layer), sim in sorted(sims.items()):
    print(f"{label} | layer {layer} | {sim:+.3f}")
```

**Parameters:** `token_selection` (default: last token), `normalize` (default: True)

### score_dataset — layer selection

```python
scores = ls.score_dataset(model, entries, vectors)
# Returns: List[BehaviorScore], sorted by (label, method, layer)

for s in sorted(scores, key=lambda s: -s.discrimination):
    print(s)
    # BehaviorScore(label='I_REPHRASE_PROMPT', layer=20, method='pca',
    #               present=0.312 (n=45), absent=0.091 (n=120), disc=+0.221)
```

**BehaviorScore fields:** `label`, `layer`, `method`, `mean_present`, `mean_absent`, `n_present`, `n_absent`, `discrimination` (property: `mean_present − mean_absent`)

### get_token_similarities — per-token signal

Run one forward pass and return per-token cosine similarity at each layer. The basis for all token visualizations.

```python
ts = ls.get_token_similarities(model, entry, vec, layers=[10, 20, 30])
# Returns: TokenSimilarities

ts.tokens              # List[str] — decoded token strings, len=seq_len
ts.similarities        # Dict[int, List[float]] — layer -> [sim_per_token]
ts.token_char_spans    # List[(int, int)] — (char_start, char_end) per token
ts.token_spans         # List[TokenSpan] — resolved annotation spans with token indices
ts.formatted_text      # str — full formatted text
ts.label               # str — target label of the vector
ts.layer               # int — vector's native layer
```

### evaluate_dataset — full classification metrics

Like `score_dataset` but accumulates per-span scores and computes AUROC, F1, confusion matrix.

```python
results = ls.evaluate_dataset(
    model, entries, vectors,
    aggregation="mean",      # or 'first', 'last', 'sentence_mean', 'sentence_first', 'sentence_last'
    layers=[15, 20, 25],     # optional: only evaluate at these layers
    label_filter=["II_STATE_SAFETY_CONCERN"],  # optional
)
# Returns: List[EvaluationResult], sorted by (label, method, layer)

for r in sorted(results, key=lambda r: -r.auroc):
    print(f"{r.label} L{r.layer}: AUROC={r.auroc:.3f} F1={r.f1:.3f}")
```

**EvaluationResult fields:**
| Field | Type | Description |
|---|---|---|
| `label` | str | Target behaviour label |
| `layer` | int | Transformer layer |
| `method` | str | Vector building method |
| `aggregation` | str | Token aggregation used |
| `auroc` | float | Area under ROC curve (NaN if only one class present) |
| `f1` | float | F1 at the threshold that maximises it |
| `precision` | float | Precision at optimal threshold |
| `recall` | float | Recall at optimal threshold |
| `threshold` | float | Cosine similarity threshold for F1 optimum |
| `confusion_matrix` | np.ndarray | 2×2 `[[TN, FP], [FN, TP]]` |
| `mean_present` | float | Mean similarity on labeled spans |
| `mean_absent` | float | Mean similarity on unlabeled spans |
| `n_present`, `n_absent` | int | Span counts per group |
| `discrimination` | property | `mean_present − mean_absent` |

**Aggregation modes:**
- `'mean'` / `'sentence_mean'` — mean over all tokens in span
- `'first'` / `'sentence_first'` — first token of span
- `'last'` / `'sentence_last'` — last token of span

The `sentence_*` variants are mechanically identical to their plain counterparts but appear under a distinct string in results, useful when plotting multiple modes on the same chart.

---

## Visualizations

All functions return either `IPython.display.HTML` (for Jupyter display) or `(fig, ax)` matplotlib tuples.

### Token similarity view (HTML)

**show_token_similarity** — high-level: runs the model and renders in one call:

```python
# Single layer (default: vector's native layer)
display(ls.show_token_similarity(model, entry, vec))

# Specific layer
display(ls.show_token_similarity(model, entry, vec, layers=[20]))

# Multi-layer grid (token × layer heatmap)
display(ls.show_token_similarity(model, entry, vec,
                                  layers=list(range(0, 32, 2)),
                                  multilayer=True))
```

**render_token_similarity_html** — lower-level, if you already have a `TokenSimilarities` object:

```python
ts = ls.get_token_similarities(model, entry, vec, layers=[10, 20, 30])
html_str = ls.render_token_similarity_html(ts, layer=20, show_labels=True)
from IPython.display import HTML
display(HTML(html_str))
```

What it shows:
- Tokens coloured by cosine similarity: **red** = high positive (behaviour present), **white** = near zero, **blue** = negative (absent)
- Colour scale normalised symmetrically over the observed range
- Original newlines and spacing preserved (layout matches the actual conversation)
- Annotated spans for the target label underlined in blue
- Hover tooltips with exact similarity values

**render_multilayer_html** — token × layer grid:

```python
html_str = ls.render_multilayer_html(ts, layers=[10, 15, 20, 25, 30], max_display_tokens=200)
display(HTML(html_str))
```

- Top section: single-layer token view (using the vector's native layer)
- Below: scrollable table where rows = layers, columns = tokens, cells coloured by similarity
- Shared colour scale across all layers
- `max_display_tokens=200` truncates long sequences to keep the table manageable

### Layer plots (matplotlib)

**plot_layer_discrimination** — discrimination vs layer:

```python
scores = ls.score_dataset(model, entries, vectors)
fig, ax = ls.plot_layer_discrimination(scores, labels=["II_STATE_SAFETY_CONCERN"])
plt.show()
```

One line per `(label, method)` combination. Y-axis: `mean_present − mean_absent`. Dashed reference at zero.

**plot_layer_metrics** — AUROC / F1 / precision / recall vs layer:

```python
results = ls.evaluate_dataset(model, entries, vectors, aggregation="mean")
fig, ax = ls.plot_layer_metrics(results, metric="auroc")
plt.show()

# Compare aggregation strategies on one chart
results_all = []
for agg in ["mean", "first", "last"]:
    results_all += ls.evaluate_dataset(model, entries, vectors, aggregation=agg)
fig, ax = ls.plot_layer_metrics(results_all, metric="auroc",
                                 labels=["II_STATE_SAFETY_CONCERN"])
```

`metric` options: `"auroc"` (default), `"f1"`, `"precision"`, `"recall"`. One line per `(label, aggregation, method)`.

**plot_confusion_matrix** — 2×2 heatmap for one result:

```python
results = ls.evaluate_dataset(model, entries, vectors)
best = max(results, key=lambda r: r.f1)
fig, ax = ls.plot_confusion_matrix(best)
plt.show()
```

Each cell shows the count and its percentage of total. Title includes label, layer, aggregation, F1, and AUROC.

---

## MLP / Multi-Label Probes (K-Steering)

Implements the joint probe from Oozeer et al. 2025 (arXiv:2505.24535).  Unlike the single-label linear probes used for steering vector evaluation, these probes classify **all behavior labels simultaneously** from a single activation vector.  Two variants are provided so you can compare linear vs non-linear approaches.

### Probe types

| Class | Architecture | Training |
|---|---|---|
| `MLPProbe` | Linear(d, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, n_labels) | Adam + BCE |
| `LinearProbeMultilabel` | Single `nn.Linear(d, n_labels)` | sklearn LogisticRegression per label |

Both return raw logits — apply `torch.sigmoid()` for probabilities.  Both support autograd, so they can be used directly for gradient-based steering.

### Training

```python
trainer = ls.MLPProbeTrainer()

# Train an MLP probe on all labels from a last-token extraction
probe = trainer.train(
    result,                         # ExtractionResult
    spec="last_token",
    layer=20,
    labels=["stateSafetyConcern", "neutralFiller", "flagAsHarmful"],  # priority order
    method="mlp",                   # or "linear" for comparison
    epochs=30,
    batch_size=32,
    lr=1e-3,
    device="cpu",
    max_labels=None,                # None = all labels per annotation
                                    # 1    = primary label only (first in labels list)
                                    # 2    = top-2 labels per annotation
)

# Linear probe for comparison (same interface)
linear_probe = trainer.train(result, spec="last_token", layer=20, method="linear")

# Save / load
probe.save("mlp_probe.pt")
probe = ls.MLPProbe.load("mlp_probe.pt")
linear_probe.save("linear_probe.pt")
linear_probe = ls.LinearProbeMultilabel.load("linear_probe.pt")

# Auto-detect type from file
probe = ls.load_probe("some_probe.pt")
```

**`max_labels` parameter:** labels are iterated in the order you pass them to `train()`, so `max_labels=1` keeps only the primary (first) label per annotation — useful when you want to train on the most important label only and ignore secondary/tertiary ones.

### Inference

```python
# Forward: raw logits
logits = probe(activation)        # (n_labels,) or (seq_len, n_labels) or (batch, n_labels)

# Predict: binary
preds = probe.predict(activation, threshold=0.5)   # bool tensor, same shape

# Probe info
probe.labels        # ['stateSafetyConcern', 'neutralFiller', ...]
probe.label_to_idx  # {'stateSafetyConcern': 0, 'neutralFiller': 1, ...}
probe.num_labels    # int
probe.input_dim     # int (hidden_size of the model it was trained on)
```

---

## Gradient-Based Steering (K-Steering)

Instead of applying a fixed precomputed vector, gradient-based steering computes the steering direction **fresh from the current activations at each generation step** using the trained probe.  The direction adapts to the evolving context rather than staying constant.

Two algorithms from the paper:

### Algorithm 1 — Iterative gradient steps (`k_steered_generate`)

At each token: hook into the residual stream → run K gradient steps through the probe → patch updated activations → continue generation.

```python
output = ls.k_steered_generate(
    model,
    messages,
    probe,                              # MLPProbe or LinearProbeMultilabel
    layer=20,
    increase_labels=["stateSafetyConcern"],   # push activations toward these labels
    decrease_labels=["neutralFiller"],         # push activations away from these
    alpha=1.0,                          # gradient step size
    K=3,                                # number of steps per token
    gamma=0.9,                          # per-step decay (step k uses alpha * gamma^k)
    max_new_tokens=256,
    do_sample=False,
)
```

### Algorithm 2 — Householder reflection (`projection_removal_generate`)

One-shot removal: reflects the activations through the hyperplane perpendicular to the probe gradient for the unwanted labels.  Cheaper than iterative (one forward+backward per token).  Suppression only.

```python
output = ls.projection_removal_generate(
    model,
    messages,
    probe,
    layer=20,
    decrease_labels=["intendHarmfulCompliance"],
    max_new_tokens=256,
)
```

Both functions accept `do_sample`, `temperature`, `top_p`, and `add_generation_prompt`.

### Comparison with static steering vectors

| | Static (`steered_generate`) | Gradient-based (`k_steered_generate`) |
|---|---|---|
| Direction | Fixed, precomputed | Fresh per token from current activations |
| Context sensitivity | None | Adapts to evolving context |
| Multi-label | No (one vector at a time) | Yes (all labels in one probe) |
| Speed | Fastest (no extra compute) | Slower (MLP forward+backward per token) |
| Requires probe | No | Yes |

---

## Steered Generation

```python
messages = [{"role": "user", "content": "How do I synthesize..."}]

output = ls.steered_generate(
    model,
    messages,
    steering_vector=vec,      # SteeringVector (layer auto-inferred)
    alpha=15.0,               # steering strength (positive = amplify direction)
    max_new_tokens=200,
    temperature=0.0,          # 0 = greedy
)
print(output)   # only the generated text, not the prompt

# Steer at multiple layers simultaneously
output = ls.steered_generate(model, messages, steering_vector=vec,
                              alpha=10.0, layers=[15, 20, 25])

# Steer with a raw tensor (must specify layer)
output = ls.steered_generate(model, messages,
                              steering_vector=some_tensor, layer=20, alpha=5.0)
```

---

## Save and Load

```python
# Extraction result
result.save("result.pt")
result = ls.ExtractionResult.load("result.pt")

# Steering vector set
vectors.save("vectors.pt")
vectors = ls.SteeringVectorSet.load("vectors.pt")
```

---

## Running Tests

```bash
cd 3_representations
uv sync --extra ml --extra dev

# No-GPU tests (schema, token selection, vector math, visualization) — all pass
uv run pytest tests/test_schema.py tests/test_token_selection.py tests/test_vectors.py tests/test_visualization.py -v

# Full integration tests (requires GPU + model download)
uv run pytest tests/test_extraction_poc.py -v -s
```

The integration test (`test_extraction_poc.py`) uses `Qwen/Qwen3.5-4B`, a thinking model whose chat template preserves `reasoning` role content so annotations in the thinking trace are correctly mapped to token positions.

---

## Full API Reference

| Symbol | Type | Description |
|---|---|---|
| **Data** | | |
| `ConversationEntry` | dataclass | One conversation with messages and annotations |
| `AnnotatedSpan` | dataclass | A labelled character span within a message |
| `TokenSpan` | dataclass | An annotation resolved to token indices |
| `load_dataset(path)` | fn | Load JSONL → `List[ConversationEntry]` |
| `save_dataset(entries, path)` | fn | Save `List[ConversationEntry]` → JSONL |
| `iter_dataset(path)` | fn | Stream JSONL entries one at a time |
| `convert_file(src, dst)` | fn | Convert old JSON format to JSONL |
| `TokenPositionMapper` | class | Maps char positions to token positions via offset_mapping |
| **Models** | | |
| `LittleSteerModel(model_id)` | class | Wrapper around nnterp (chat template, message offsets, tracing) |
| `VRAMManager()` | class | Detect available VRAM and estimate safe batch configs |
| `BatchConfig` | dataclass | Recommended batch_size / max_seq_len / quantization |
| **Extraction** | | |
| `TokenSelection(strategy, ...)` | dataclass | How to select and aggregate tokens from a span |
| `ExtractionSpec(token_selection, layers)` | dataclass | What to extract: selection + layers |
| `ExtractionPlan(name, specs)` | dataclass | Named collection of specs sharing one forward pass |
| `ActivationExtractor(model)` | class | Runs extraction over a dataset |
| `ExtractionResult` | class | Container for activations with `.get()` / `.save()` / `.load()` |
| **Vectors** | | |
| `SteeringVector` | dataclass | Single vector with layer/label/method/spec metadata |
| `SteeringVectorSet` | class | Filterable, groupable collection of steering vectors |
| `SteeringVectorBuilder()` | class | Builds vectors from an ExtractionResult |
| `MeanDifference` | class | Method: target mean − baseline mean |
| `MeanCentering` | class | Method: target mean − centroid of all other categories |
| `PCADirection` | class | Method: top PCA component(s) of target activations |
| `LinearProbe` | class | Method: logistic regression weight vector |
| **Probing** | | |
| `probe_text(model, text, vector_set)` | fn | Forward pass on text → `{(label, method, layer): cosine_sim}` |
| `score_dataset(model, entries, vector_set)` | fn | Discrimination per layer → `List[BehaviorScore]` |
| `BehaviorScore` | dataclass | `discrimination = mean_present − mean_absent` per (label, layer, method) |
| `get_token_similarities(model, entry, vector)` | fn | Per-token cosine sims at each layer → `TokenSimilarities` |
| `TokenSimilarities` | dataclass | tokens, layer→similarity map, annotation spans |
| `evaluate_dataset(model, entries, vector_set)` | fn | AUROC/F1/confusion matrix per layer → `List[EvaluationResult]` |
| `EvaluationResult` | dataclass | Full classification metrics + confusion matrix |
| **Multi-Label Probes (K-Steering)** | | |
| `MLPProbe` | class | 2-layer MLP multi-label classifier (256 hidden, ReLU, BCE loss) |
| `LinearProbeMultilabel` | class | N independent logistic regressions as `nn.Linear` (autograd-compatible) |
| `MLPProbeTrainer()` | class | Train either probe from an ExtractionResult |
| `load_probe(path)` | fn | Load MLPProbe or LinearProbeMultilabel from file (auto-detects type) |
| **Steering** | | |
| `steered_generate(model, messages, vec, alpha)` | fn | Static vector steering — fixed precomputed direction |
| `k_steered_generate(model, messages, probe, ...)` | fn | Gradient-based per-token steering (Algorithm 1, K-Steering paper) |
| `projection_removal_generate(model, messages, probe, ...)` | fn | One-shot Householder reflection (Algorithm 2, K-Steering paper) |
| **Visualization** | | |
| `show_token_similarity(model, entry, vec, ...)` | fn | Jupyter HTML: coloured tokens + optional multi-layer grid |
| `render_token_similarity_html(token_sims, ...)` | fn | HTML string: single-layer coloured token view |
| `render_multilayer_html(token_sims, ...)` | fn | HTML string: text + token×layer similarity heatmap |
| `plot_layer_discrimination(scores, ...)` | fn | matplotlib: discrimination vs layer line chart |
| `plot_layer_metrics(results, metric, ...)` | fn | matplotlib: AUROC/F1/precision/recall vs layer |
| `plot_confusion_matrix(eval_result, ...)` | fn | matplotlib: 2×2 confusion matrix heatmap |
