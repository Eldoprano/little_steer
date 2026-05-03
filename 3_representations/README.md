# little_steer

A Python library for extracting activations from transformer models, building steering vectors, and probing/visualizing safety-relevant behaviour in reasoning models.

Built on top of [nnterp](https://github.com/nnsight/nnterp) and [nnsight](https://nnsight.net/). Model-agnostic: works with any HuggingFace causal LM.

> **What's new in 0.2.0** — see the *Session-cached detection*, *Hard-to-game scoring*, *Persona / Assistant-Axis*, *Vector transforms*, and *Response-only steering* sections below. The marimo demo at `3_representations/showcase.py` walks through everything.

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

## Session — cached detection

`Session` owns a model + dataset and caches forward passes across calls. Use it whenever you'll be calling more than one read-side method on the same data — every detection function below is implemented as a thin wrapper around it.

```python
session = ls.Session(model, entries, cache_size=64)

# Discrimination per (label, layer, method)
scores = session.score(vectors)

# Full classification metrics — multiple aggregations share ONE forward pass per entry
results = session.evaluate(vectors, aggregations=["mean", "first", "last"])

# Per-token, per-layer cosine similarities (for visualisation)
ts = session.token_similarities(entries[0], vectors[0], layers=[10, 20, 30])

# Probe-based detection (sigmoid probabilities per (token, label))
detection = session.detect_with_probe(entries[0], probe, layer=20)

# Or vector-based detection (cosine sim per (token, label))
detection = session.detect_with_vectors(entries[0], vectors.filter(layer=20), layer=20)

session.clear_cache()  # free CPU memory if needed
```

The big speedup: the old `evaluate_dataset(..., aggregation="mean")` had to be called three times to compare aggregations. With Session, pass `aggregations=["mean", "first", "last"]` and it's ~3× cheaper.

The legacy `score_dataset` / `evaluate_dataset` / `probe_text` / `get_token_similarities` / `get_probe_predictions` / `get_multilabel_token_scores` functions still exist and produce identical output — they construct a Session under the hood.

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

# Response-only steering: inject only on assistant tokens, leave the prompt alone.
# Cleaner for measuring "what does steering the model's response do?" because
# the model's READING of the prompt is unmodified.
output = ls.steered_generate(model, messages, steering_vector=vec,
                              alpha=15.0, response_only=True)
```

---

## Hard-to-game scoring

Discrimination and AUROC can be inflated by token-matching: if the spans for "I_INTEND_REFUSAL" all contain the word "refuse", a vector that's basically a "refuse-token detector" will score well without encoding any deeper concept. The `little_steer.scoring` module ships five complementary scores:

| Function | Measures | Cost |
|---|---|---|
| `causal_steering_score` | Does the vector *cause* the behaviour to appear when injected? | High (needs a judge LLM + many generations) |
| `token_ablation_score` | Discrimination retained after masking behaviour-keyword tokens | Low |
| `neutral_pca_score` | Discrimination retained after projecting out neutral-text PCs (Anthropic emotion-vector recipe) | Medium |
| `logit_lens_top_tokens` | Top tokens the vector upweights via the unembedding (diagnostic) | ~Free |
| `embedding_keyword_overlap` | Cosine of vector to mean keyword token embedding (diagnostic) | ~Free |

The headline metric is `causal_steering_score` — generate at increasing alphas, ask a judge whether the target behaviour appeared, and regress. A vector that doesn't actually cause the behaviour produces a flat curve, which can't be gamed by token co-occurrence in the training set.

```python
# Logit lens — quick sanity check that the vector talks about the right thing.
readout = ls.logit_lens_top_tokens(model, vec, k=20)
for tok, logit in readout.top_positive:
    print(f"  +{logit:7.2f}  {tok!r}")

# Embedding keyword overlap — diagnostic for surface-level vectors.
overlap = ls.embedding_keyword_overlap(model, vec,
                                       keywords=["concern", "unsafe", "harmful"])
print(f"cosine to mean keyword embedding: {overlap.cosine:+.3f}")
# > 0.5 is a red flag; 0.1–0.3 is typical for a deep-behavioural vector.

# Token-confound ablation — re-score after masking behaviour keywords.
ablation = ls.token_ablation_score(
    model, eval_entries, vec,
    keywords=["concern", "unsafe", "harmful"],
)
print(f"retained {ablation.retained_fraction:.0%} of discrimination")

# Neutral-PCA confound removal — Anthropic emotion-vector recipe.
pca_score = ls.neutral_pca_score(
    model, vec,
    neutral_entries=benign_prompts,
    eval_entries=eval_entries,
    n_components=8,
)

# Causal steering score — the headline metric. Requires a judge.
def my_judge(prompt: str, response: str, label: str) -> float:
    # Wrap your favourite LLM judge here. Returns a probability in [0, 1].
    ...

causal = ls.causal_steering_score(
    model, vec,
    neutral_prompts=[...],            # mostly-benign prompts
    judge=my_judge,
    alphas=(0, 5, 10, 15, 20),
    n_per_alpha=10,
)
print(f"slope = {causal.slope:+.3f} per unit alpha")
```

`ScoringReport` collects everything into a Rich table:

```python
report = ls.ScoringReport(
    logit_lens=[ls.logit_lens_top_tokens(model, v) for v in vectors],
    embedding_overlap=[ls.embedding_keyword_overlap(model, v, keywords) for v in vectors],
    ablation=[...],
    neutral_pca=[...],
    causal=[...],
)
report.render()
```

---

## Persona / Assistant-Axis (Lu et al. 2026)

`little_steer.persona` ships the linear-algebra primitives for the Assistant-Axis recipe. The paper's full data-collection pipeline (275 roles × 5 system prompts × 240 questions, in-role filtering) is out of scope — bring your own role/prompt collection.

```python
# Build a vector per (role, layer) plus the first PCA component across roles.
axis = ls.extract_assistant_axis(
    model,
    role_prompts={
        "teacher": ["You are a kindergarten teacher. Explain photosynthesis.", ...],
        "lawyer":  ["You are a corporate lawyer. Draft a non-disclosure clause.", ...],
        "robot":   ["You are a sci-fi robot from 2123. Describe your routine.", ...],
    },
    neutral_prompts=["What is 2+2?", "List three rivers in Europe.", ...],
    layers=list(range(int(model.num_layers * 0.4),
                       int(model.num_layers * 0.7))),
)
assistant_axis = axis.filter(label="ASSISTANT_AXIS").vectors[0]

# Track persona drift across a multi-turn conversation.
drifts = ls.persona_drift(model, multi_turn_conversation, assistant_axis)
# One float per assistant turn — decreasing values = drift away from Assistant.

# Calibrate alpha by measuring typical residual-stream norm.
norm_at_layer = ls.residual_norm(model, calibration_dataset, layer=20,
                                  sample_size=200)
scaled = ls.normalize_to_residual_scale(vec.vector, target_norm=norm_at_layer)
# Now alpha=1.0 means "one residual-stream unit", comparable across layers.
```

---

## Vector transforms

`little_steer.vectors.transforms` exposes three primitives that come up in nearly every confound-removal / multi-attribute-steering experiment:

```python
# Project a basis OUT of a vector (Gram–Schmidt internally; basis need not be orthonormal).
cleaned = ls.project_out(vec.vector, neutral_pca_basis)

# Rescale a vector to a target L2 norm.
scaled = ls.normalize_to_residual_scale(vec.vector, target_norm=42.0)

# Weighted sum of vectors at the same layer (multi-attribute intervention).
combined = ls.compose([safety_vec, sycophancy_vec], weights=[1.0, -0.5])
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

## Marimo showcase

`3_representations/showcase.py` is a marimo notebook that walks through the new API end-to-end: cached `Session` detection, hard-to-game scoring, response-only steering, and an Assistant-Axis preview. Open with:

```bash
cd 3_representations
uv run marimo edit showcase.py
```

---

## Running Tests

```bash
cd 3_representations
uv sync --extra ml --extra dev

# No-GPU tests (schema, token selection, vector math, visualization,
# transforms, span-eval helpers) — all pass
uv run pytest tests/test_schema.py tests/test_token_selection.py \
              tests/test_vectors.py tests/test_visualization.py \
              tests/test_transforms.py tests/test_span_eval.py -v

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
| `project_out(vec, basis)` | fn | Remove a basis (e.g. neutral PCs) from a vector |
| `normalize_to_residual_scale(vec, target)` | fn | Rescale to a target L2 norm |
| `compose([vec, ...], [w, ...])` | fn | Weighted sum of vectors |
| **Session** | | |
| `Session(model, entries)` | class | Cache-aware container; backbone of all detection |
| `session.score(vectors)` | method | BehaviorScore per (label, layer, method) |
| `session.evaluate(vectors, aggregations=[...])` | method | Full classification metrics; aggregations share one forward pass |
| `session.token_similarities(entry, vector)` | method | Per-token similarities at one or more layers |
| `session.detect_with_probe(entry, probe, layer)` | method | Sigmoid probabilities per (token, label) |
| `session.detect_with_vectors(entry, vectors, layer)` | method | Cosine sim per (token, label) |
| `session.probe_text(text, vectors)` | method | Cosine sim of raw-text activation against vectors |
| `session.clear_cache()` | method | Drop cached forward passes (free memory) |
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
| `steered_generate(model, messages, vec, alpha, response_only=False)` | fn | Static vector steering — fixed precomputed direction; `response_only=True` only fires on assistant tokens |
| `multi_steered_generate(model, messages, [(vec, layer, alpha), ...])` | fn | Compose multiple vectors at once |
| `k_steered_generate(model, messages, probe, ...)` | fn | Gradient-based per-token steering (Algorithm 1, K-Steering paper) |
| `projection_removal_generate(model, messages, probe, ...)` | fn | One-shot Householder reflection (Algorithm 2, K-Steering paper) |
| **Hard-to-game scoring** | | |
| `causal_steering_score(model, vec, neutral_prompts, judge, alphas)` | fn | Generate with each alpha, judge whether behaviour appeared, regress slope |
| `token_ablation_score(model, entries, vec, keywords)` | fn | Mask keyword tokens; score retained discrimination |
| `neutral_pca_score(model, vec, neutral_entries, eval_entries)` | fn | Project out neutral-text PCs; score retained discrimination |
| `logit_lens_top_tokens(model, vec, k=20)` | fn | Top tokens the vector upweights via the unembedding |
| `embedding_keyword_overlap(model, vec, keywords)` | fn | Cosine of vector to mean keyword token embedding |
| `ScoringReport` | dataclass | Container with a `.render()` method (Rich table) |
| **Persona / Assistant-Axis (Lu et al. 2026)** | | |
| `extract_assistant_axis(model, role_prompts, neutral_prompts, layers)` | fn | Per-role mean-centred vectors + first PCA component (the axis) |
| `persona_drift(model, conversation, axis_vector)` | fn | Project each assistant turn onto the axis to track drift |
| `residual_norm(model, dataset, layer)` | fn | Mean L2 norm of the residual stream — calibration for alpha |
| **Visualization** | | |
| `show_token_similarity(model, entry, vec, ...)` | fn | Jupyter HTML: coloured tokens + optional multi-layer grid |
| `render_token_similarity_html(token_sims, ...)` | fn | HTML string: single-layer coloured token view |
| `render_multilayer_html(token_sims, ...)` | fn | HTML string: text + token×layer similarity heatmap |
| `plot_layer_discrimination(scores, ...)` | fn | matplotlib: discrimination vs layer line chart |
| `plot_layer_metrics(results, metric, ...)` | fn | matplotlib: AUROC/F1/precision/recall vs layer |
| `plot_confusion_matrix(eval_result, ...)` | fn | matplotlib: 2×2 confusion matrix heatmap |
