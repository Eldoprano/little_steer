/**
 * localStorage persistence for labeling progress.
 *
 * Keys:
 *   labeler_entries   – serialised ConversationEntry[] (the loaded dataset)
 *   labeler_progress  – PersistedState
 */

import type { ConversationEntry, PersistedState } from './types';

const ENTRIES_KEY = 'labeler_entries';
const PROGRESS_KEY = 'labeler_progress';

// ── Entry storage ────────────────────────────────────────────────────────────

export function saveEntries(entries: ConversationEntry[]): void {
  try {
    localStorage.setItem(ENTRIES_KEY, JSON.stringify(entries));
  } catch {
    console.warn('Could not persist entries to localStorage (storage full?)');
  }
}

export function loadEntries(): ConversationEntry[] | null {
  try {
    const raw = localStorage.getItem(ENTRIES_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as ConversationEntry[];
  } catch {
    return null;
  }
}

export function clearEntries(): void {
  localStorage.removeItem(ENTRIES_KEY);
}

// ── Progress storage ─────────────────────────────────────────────────────────

export function defaultProgress(): PersistedState {
  return { handle: null, progress: {}, currentEntryIndex: 0, currentSentenceIndex: 0 };
}

/** Migrate old progress records that lack sentenceScores or sentenceConfidences. */
function migrateProgress(p: PersistedState): PersistedState {
  const migrated: PersistedState['progress'] = {};
  for (const [id, ep] of Object.entries(p.progress)) {
    migrated[id] = { 
      ...ep, 
      sentenceScores: ep.sentenceScores ?? {},
      sentenceConfidences: ep.sentenceConfidences ?? {}
    };
  }
  return { ...p, handle: p.handle ?? null, progress: migrated };
}

export function saveProgress(state: PersistedState): void {
  try {
    localStorage.setItem(PROGRESS_KEY, JSON.stringify(state));
  } catch {
    console.warn('Could not persist progress to localStorage (storage full?)');
  }
}

export function loadProgress(): PersistedState {
  try {
    const raw = localStorage.getItem(PROGRESS_KEY);
    if (!raw) return defaultProgress();
    return migrateProgress(JSON.parse(raw) as PersistedState);
  } catch {
    return defaultProgress();
  }
}

export function clearProgress(): void {
  localStorage.removeItem(PROGRESS_KEY);
}

// ── Filtering helpers ────────────────────────────────────────────────────────

/**
 * Return only entries that have LLM-generated sentence annotations
 * (so the human labels can be compared with model labels).
 */
export function filterLabeledEntries(entries: ConversationEntry[]): ConversationEntry[] {
  return entries.filter(
    (e) =>
      Array.isArray(e.annotations) &&
      e.annotations.length > 0 &&
      e.messages.some((m) => m.role === 'reasoning'),
  );
}

/**
 * Extract sentences from an entry, sorted by char_start.
 *
 * If the entry has LLM-generated annotation spans, uses those as sentence
 * boundaries (preserving the judge's segmentation for comparison).
 * Falls back to simple heuristic splitting when no annotations are present
 * (e.g. entries loaded directly from data/1_generated/ before LLM labeling).
 */
export function getSentences(entry: ConversationEntry) {
  const reasoningIdx = entry.messages.findIndex((m) => m.role === 'reasoning');
  if (reasoningIdx === -1) return [];

  const text = entry.messages[reasoningIdx].content;

  // We ignore judge-generated annotations for defining boundaries because they are 
  // often corrupted (splitting words, inconsistent offsets). Instead, we 
  // define our own "Gold Standard" segments using a local heuristic splitter.
  return splitReasoningText(text).map((s, i) => ({
    index: i,
    text: s.text,
    char_start: s.start,
    char_end: s.end,
    message_idx: reasoningIdx,
  }));
}

/**
 * Convert a Unicode code point offset (e.g. from Python) to a JS UTF-16 code unit offset.
 */
function codePointToCodeUnitOffset(str: string, cpOffset: number): number {
  let cuOffset = 0;
  let cpCount = 0;
  while (cpCount < cpOffset && cuOffset < str.length) {
    const cp = str.codePointAt(cuOffset);
    if (cp === undefined) break;
    cuOffset += cp > 0xFFFF ? 2 : 1;
    cpCount++;
  }
  return cuOffset;
}

/**
 * Convert a JS UTF-16 code unit offset back to a Unicode code point offset.
 */
function codeUnitToCodePointOffset(str: string, cuOffset: number): number {
  let cpCount = 0;
  let currentCuOffset = 0;
  while (currentCuOffset < cuOffset && currentCuOffset < str.length) {
    const cp = str.codePointAt(currentCuOffset);
    if (cp === undefined) break;
    currentCuOffset += cp > 0xFFFF ? 2 : 1;
    cpCount++;
  }
  return cpCount;
}

/**
 * Robust sentence splitter for reasoning traces.
 * Splits on sentence-ending punctuation followed by whitespace/newlines.
 */
function splitReasoningText(text: string): Array<{ text: string; start: number; end: number }> {
  const raw: Array<{ text: string; start: number; end: number }> = [];

  // Match: 
  // 1. A sequence of characters that ends with (. ! ? or :) AND (whitespace or end of string)
  // OR 
  // 2. Lines as explicit separators
  const re = /.*?[.!?:](\s+|$)|[^\n]+(\n|$)/g;
  
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const t = m[0].trim();
    if (!t) continue;
    
    const offset = m[0].indexOf(t);
    raw.push({ 
      text: t, 
      start: m.index + offset, 
      end: m.index + offset + t.length 
    });
  }

  // Merge logical fragments (e.g. "1.", "2.") or very short accidental splits
  const processing = [...raw];
  const merged: typeof raw = [];
  
  for (let i = 0; i < processing.length; i++) {
    const current = processing[i];

    // 1. Forward merge: list markers (e.g. "1.", "A.") or tiny fragments
    // These should belong to the sentence they introduce.
    if (current.text.length <= 3 && i < processing.length - 1) {
      processing[i + 1].text = current.text + ' ' + processing[i + 1].text;
      processing[i + 1].start = current.start;
      continue;
    }

    const last = merged[merged.length - 1];
    // 2. Backward merge: continuation of previous sentence or tiny trailing fragment
    if (
      last &&
      (!/[.!?:]$/.test(last.text) ||
        /^[a-z,;]/.test(current.text) ||
        (current.text.length <= 3 && i === processing.length - 1))
    ) {
      last.text = (last.text + ' ' + current.text).replace(/\s+/g, ' ').trim();
      last.end = current.end;
    } else {
      merged.push({ ...current });
    }
  }

  return merged.length > 0 ? merged : [{ text: text.trim(), start: 0, end: text.length }];
}

// ── Export helpers ───────────────────────────────────────────────────────────

/**
 * Build a ConversationEntry with human labels — matching the exact output
 * format used by the automated labeler (ConversationEntry schema).
 */
export function buildHumanLabeledEntry(
  original: ConversationEntry,
  sentenceLabels: Record<number, string[]>,
  sentenceScores: Record<number, number>,
  sentenceConfidences: Record<number, Record<string, number>>,
  assessment: {
    trajectory: string;
    turning_point: number;
    alignment: string;
  },
  handle: string,
): ConversationEntry {
  const sentences = getSentences(original);

  const reasoningIdx = original.messages.findIndex((m) => m.role === 'reasoning');
  const reasoningText = reasoningIdx >= 0 ? original.messages[reasoningIdx].content : '';

  const annotations = sentences.map((s) => {
    const labels = sentenceLabels[s.index] ?? [];
    // Empty labels = human explicitly found no matching behavior
    const resolvedLabels = labels.length === 0 ? ['none'] : labels;
    
    // Confidence parallel array
    const confs = resolvedLabels.map(l => {
      if (l === 'none') return 1;
      return sentenceConfidences[s.index]?.[l] ?? 1;
    });

    const safetyScore = sentenceScores[s.index] ?? 0;
    return {
      text: s.text,
      message_idx: s.message_idx,
      char_start: codeUnitToCodePointOffset(reasoningText, s.char_start),
      char_end: codeUnitToCodePointOffset(reasoningText, s.char_end),
      labels: resolvedLabels,
      confidence: confs,
      score: safetyScore,
      meta: {},
    };
  });

  const judgeName = `human_${handle}`;
  let cleanId = original.id;
  if (cleanId.startsWith('dataset.jsonl:')) {
    cleanId = cleanId.substring('dataset.jsonl:'.length);
  }

  return {
    id: cleanId,
    messages: original.messages,
    annotations,
    model: original.model,
    judge: judgeName,
    metadata: {
      ...original.metadata,
      assessment: {
        trajectory: assessment.trajectory as import('./types').TrajectoryType,
        turning_point: assessment.turning_point,
        alignment: assessment.alignment as import('./types').AlignmentType,
      },
      has_reasoning: true,
      labeled_at: new Date().toISOString(),
      human_labeled: true,
      judge_name: judgeName,
    },
  };
}

/**
 * Reconstruct sentenceLabels and sentenceScores maps from an existing human label run.
 * This ensures that if a user returns to a previously labeled entry, their work is visible.
 */
export function reconstructProgress(entry: ConversationEntry, judgeName: string) {
  const run = entry.label_runs?.find(r => r.judge_name === judgeName);
  if (!run || !run.spans) return null;

  const sentenceLabels: Record<number, string[]> = {};
  const sentenceScores: Record<number, number> = {};
  const sentenceConfidences: Record<number, Record<string, number>> = {};

  run.spans.forEach((span, idx) => {
    // We assume the spans in the label run correspond 1:1 with sentences 
    // generated by getSentences(entry) at the time of labeling.
    const labels = (span.labels || []).filter(l => l !== 'none');
    sentenceLabels[idx] = labels;
    sentenceScores[idx] = span.score ?? 0;

    if (span.confidence && span.confidence.length > 0) {
      const confMap: Record<string, number> = {};
      // Match confidence back to label based on index
      // Since 'none' was filtered out, we need to be careful if 'none' was present.
      // But buildHumanLabeledEntry adds 'none' as the only label if empty.
      
      let labelIdx = 0;
      (span.labels || []).forEach((lbl, i) => {
        if (lbl !== 'none') {
          confMap[lbl] = span.confidence![i] ?? 1;
        }
      });
      sentenceConfidences[idx] = confMap;
    }
  });

  return {
    sentenceLabels,
    sentenceScores,
    sentenceConfidences,
    assessment: run.assessment as unknown as import('./types').Assessment,
    completed: run.status === 'completed'
  };
}

/**
 * Export all completed entries as a JSONL string.
 */
export function exportAsJsonl(
  entries: ConversationEntry[],
  progressMap: Record<string, { 
    sentenceLabels: Record<number, string[]>; 
    sentenceScores: Record<number, number>; 
    sentenceConfidences: Record<number, Record<string, number>>;
    assessment?: { trajectory: string; turning_point: number; alignment: string }; 
    completed: boolean 
  }>,
  handle: string,
): string {
  const lines: string[] = [];
  for (const entry of entries) {
    const prog = progressMap[entry.id];
    if (!prog?.completed || !prog.assessment) continue;
    const humanEntry = buildHumanLabeledEntry(
      entry, 
      prog.sentenceLabels, 
      prog.sentenceScores ?? {}, 
      prog.sentenceConfidences ?? {},
      prog.assessment, 
      handle
    );
    lines.push(JSON.stringify(humanEntry));
  }
  return lines.join('\n');
}
