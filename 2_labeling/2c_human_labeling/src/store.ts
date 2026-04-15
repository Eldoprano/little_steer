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
  return { progress: {}, currentEntryIndex: 0, currentSentenceIndex: 0 };
}

/** Migrate old progress records that lack sentenceScores. */
function migrateProgress(p: PersistedState): PersistedState {
  const migrated: PersistedState['progress'] = {};
  for (const [id, ep] of Object.entries(p.progress)) {
    migrated[id] = { ...ep, sentenceScores: ep.sentenceScores ?? {} };
  }
  return { ...p, progress: migrated };
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

  const annotated = entry.annotations
    .filter((a) => a.message_idx === reasoningIdx)
    .sort((a, b) => a.char_start - b.char_start);

  if (annotated.length > 0) {
    return annotated.map((a, i) => ({
      index: i,
      text: a.text,
      char_start: a.char_start,
      char_end: a.char_end,
      message_idx: a.message_idx,
    }));
  }

  // Fallback: split the raw reasoning text into sentence-like chunks
  const text = entry.messages[reasoningIdx].content;
  return splitReasoningText(text).map((s, i) => ({
    index: i,
    text: s.text,
    char_start: s.start,
    char_end: s.end,
    message_idx: reasoningIdx,
  }));
}

/**
 * Heuristic sentence splitter for raw reasoning traces (no LLM annotations).
 * Splits on sentence-ending punctuation and newlines, merges very short fragments.
 */
function splitReasoningText(text: string): Array<{ text: string; start: number; end: number }> {
  const raw: Array<{ text: string; start: number; end: number }> = [];

  // Match runs of non-sentence-ending, non-newline text, optionally ending in punctuation
  const re = /[^\n.!?]+[.!?]*/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    const t = m[0].trim();
    if (t.length < 5) continue;
    // Locate trimmed text within the match (accounts for leading whitespace)
    const offset = m[0].indexOf(t);
    raw.push({ text: t, start: m.index + offset, end: m.index + offset + t.length });
  }

  // Merge fragments shorter than 30 chars into the previous sentence
  const merged: typeof raw = [];
  for (const seg of raw) {
    if (merged.length > 0 && merged[merged.length - 1].text.length < 30) {
      const prev = merged[merged.length - 1];
      prev.text = prev.text + ' ' + seg.text;
      prev.end = seg.end;
    } else {
      merged.push({ ...seg });
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
  assessment: {
    trajectory: string;
    turning_point: number;
    alignment: string;
  },
): ConversationEntry {
  const sentences = getSentences(original);

  const annotations = sentences.map((s) => {
    const labels = sentenceLabels[s.index] ?? [];
    // Empty labels = human explicitly found no matching behavior
    const resolvedLabels = labels.length === 0 ? ['none'] : labels;
    const safetyScore = sentenceScores[s.index] ?? 0;
    return {
      text: s.text,
      message_idx: s.message_idx,
      char_start: s.char_start,
      char_end: s.char_end,
      labels: resolvedLabels,
      score: safetyScore,
      meta: {},
    };
  });

  return {
    id: original.id,
    messages: original.messages,
    annotations,
    model: original.model,
    judge: 'human',
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
    },
  };
}

/**
 * Export all completed entries as a JSONL string.
 */
export function exportAsJsonl(
  entries: ConversationEntry[],
  progressMap: Record<string, { sentenceLabels: Record<number, string[]>; sentenceScores: Record<number, number>; assessment?: { trajectory: string; turning_point: number; alignment: string }; completed: boolean }>,
): string {
  const lines: string[] = [];
  for (const entry of entries) {
    const prog = progressMap[entry.id];
    if (!prog?.completed || !prog.assessment) continue;
    const humanEntry = buildHumanLabeledEntry(entry, prog.sentenceLabels, prog.sentenceScores ?? {}, prog.assessment);
    lines.push(JSON.stringify(humanEntry));
  }
  return lines.join('\n');
}
