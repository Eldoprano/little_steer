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
    return JSON.parse(raw) as PersistedState;
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
 * Extract sentences from an entry's annotations, sorted by char_start.
 * Sentences are the existing LLM-identified annotation spans in the reasoning message.
 */
export function getSentences(entry: ConversationEntry) {
  const reasoningIdx = entry.messages.findIndex((m) => m.role === 'reasoning');
  if (reasoningIdx === -1) return [];

  return entry.annotations
    .filter((a) => a.message_idx === reasoningIdx)
    .sort((a, b) => a.char_start - b.char_start)
    .map((a, i) => ({
      index: i,
      text: a.text,
      char_start: a.char_start,
      char_end: a.char_end,
      message_idx: a.message_idx,
    }));
}

// ── Export helpers ───────────────────────────────────────────────────────────

/**
 * Build a ConversationEntry with human labels — matching the exact output
 * format used by the automated labeler (ConversationEntry schema).
 */
export function buildHumanLabeledEntry(
  original: ConversationEntry,
  sentenceLabels: Record<number, string[]>,
  assessment: {
    trajectory: string;
    turning_point: number;
    alignment: string;
  },
): ConversationEntry {
  const sentences = getSentences(original);

  const annotations = sentences.map((s) => {
    const labels = sentenceLabels[s.index];
    // Map NONE selection → VII_NEUTRAL_FILLER (schema requires non-empty labels)
    const resolvedLabels =
      labels && labels.length > 0 && labels[0] !== 'NONE'
        ? labels
        : ['VII_NEUTRAL_FILLER'];
    const isNone = !labels || labels.length === 0 || labels[0] === 'NONE';
    return {
      text: s.text,
      message_idx: s.message_idx,
      char_start: s.char_start,
      char_end: s.char_end,
      labels: resolvedLabels,
      score: 0.0,
      meta: isNone ? { human_none: true } : {},
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
  progressMap: Record<string, { sentenceLabels: Record<number, string[]>; assessment?: { trajectory: string; turning_point: number; alignment: string }; completed: boolean }>,
): string {
  const lines: string[] = [];
  for (const entry of entries) {
    const prog = progressMap[entry.id];
    if (!prog?.completed || !prog.assessment) continue;
    const humanEntry = buildHumanLabeledEntry(entry, prog.sentenceLabels, prog.assessment);
    lines.push(JSON.stringify(humanEntry));
  }
  return lines.join('\n');
}
