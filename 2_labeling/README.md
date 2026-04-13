# Step 2 — Labeling

Labels LLM reasoning traces with safety-relevant behavioral annotations. The pipeline has two sub-steps:

| Sub-step | Directory | Purpose |
|---|---|---|
| 2a | `2a_evolving/` | Taxonomy evolution — refine label definitions using LLM feedback |
| 2b | `2b_sentence/` | Sentence labeling — assign behavioral labels to individual sentences in reasoning traces |

## Pipeline position

```
1_generating/  →  2_labeling/  →  3_representations/
   (traces)       (annotated)        (vectors)
```

Input: `ConversationEntry` JSONL files from `data/1_generated/`
Output: annotated `ConversationEntry` JSONL files in `data/2b_labeled/`

## Quick start

```bash
# Automated sentence labeling (LLM judge)
cd 2b_sentence
uv sync
uv run run.py ../../data/1_generated/

# Human labeling (web interface)
cd 2b_sentence/human_labeling
npm install
npm run dev
```

See each sub-directory for detailed instructions.
