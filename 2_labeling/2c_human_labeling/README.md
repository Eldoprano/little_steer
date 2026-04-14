# Human Labeling Interface

A mobile-first web app for manually labeling sentences in LLM reasoning traces. Built with Vite + React + TypeScript, designed to work on a phone so labeling can happen away from the desk.

## What it does

1. Load JSONL files (the same `ConversationEntry` format used throughout the pipeline)
2. For each entry, display sentences extracted from the reasoning message one at a time
3. The user assigns behavioral labels from the taxonomy to each sentence
4. After all sentences are labeled, an assessment screen captures the overall reasoning trajectory (e.g. `concern_then_refuse`, `comply_no_deliberation`) and alignment with the final output
5. Progress is persisted in `localStorage` -- safe to close and resume

## Quick start

```bash
cd 2_labeling/2c_human_labeling
npm install
npm run dev
```

Then open the local URL in a browser (or on a phone on the same network). The dev server automatically discovers labeled JSONL files in `data/2b_labeled/` and presents them for selection. You can also upload files manually through the UI.

## Build

```bash
npm run build     # outputs to dist/
npm run preview   # serve the production build locally
```

## Files

| File | Purpose |
|---|---|
| `src/App.tsx` | Main app: screen routing, state management, entry navigation |
| `src/components/Labeler.tsx` | Sentence display and label selection UI |
| `src/components/AssessmentScreen.tsx` | Post-labeling assessment form (trajectory, alignment) |
| `src/components/DataLoader.tsx` | JSONL file upload screen |
| `src/components/StatsView.tsx` | Progress stats and label export |
| `src/store.ts` | localStorage persistence for entries and progress |
| `src/taxonomy.ts` | Label taxonomy definition (shared with the automated labeler) |
| `src/types.ts` | TypeScript types matching the `ConversationEntry` / `AnnotatedSpan` schema |

## Design

- Uses the Everforest Dark color scheme
- The Vite dev server exposes `/api/files` and `/api/data/<name>` endpoints to serve labeled data from `data/2b_labeled/`
- All labeling state lives in `localStorage` -- no external backend needed
- Entries are filtered to only include those with reasoning messages and existing annotations
- The assessment captures three things: reasoning trajectory type, turning point (sentence index), and whether the final output aligns with the reasoning
