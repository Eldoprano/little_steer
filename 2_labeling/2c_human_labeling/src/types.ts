// Types matching the ConversationEntry / AnnotatedSpan schema exactly.

export type MessageRole = 'system' | 'user' | 'assistant' | 'reasoning';

export interface Message {
  role: MessageRole;
  content: string;
}

export interface AnnotatedSpan {
  text: string;
  message_idx: number;
  char_start: number;
  char_end: number;
  labels: string[];
  score: number;
  meta: Record<string, unknown>;
}

// TODO: Again hardcoded trajectories
export type TrajectoryType =
  | 'concern_then_refuse'
  | 'concern_then_comply'
  | 'comply_no_deliberation'
  | 'refuse_no_deliberation'
  | 'deliberation_then_refuse'
  | 'deliberation_then_comply';

export type AlignmentType = 'aligned' | 'misaligned' | 'ambiguous';

export interface Assessment {
  trajectory: TrajectoryType;
  turning_point: number;
  alignment: AlignmentType;
}

export interface LabelRun {
  judge_name: string;
  judge_model_id: string;
  taxonomy_version: string;
  labeled_at: string;
  generation_hash: string;
  reasoning_truncated: boolean;
  assessment: Record<string, unknown>;
  sentence_annotations: Record<string, unknown>[];
  spans: AnnotatedSpan[];
  usage: Record<string, unknown>;
  finish_reason: string;
  status: string;
  error?: string | null;
}

export interface SafetyRun {
  guard_name: string;
  guard_model_id: string;
  scored_at: string;
  generation_hash: string;
  result: Record<string, unknown>;
  status: string;
  error?: string | null;
}

export interface EntryMetadata {
  assessment?: Assessment;
  dataset_name?: string;
  has_reasoning?: boolean;
  labeled_at?: string;
  reasoning_truncated?: boolean;
  [key: string]: unknown;
}

export interface ConversationEntry {
  id: string;
  messages: Message[];
  annotations: AnnotatedSpan[];
  model: string;
  judge: string;
  metadata: EntryMetadata;
  label_runs?: LabelRun[];
  safety_runs?: SafetyRun[];
}

// ── Human labeling state ────────────────────────────────────────────────────

/** Labels assigned to one sentence (by 0-based sentence index within the entry). */
export type SentenceLabels = string[];

/** Labeling progress for one entry. */
export interface EntryProgress {
  /** sentenceIndex → ordered list of labels (empty = not yet labeled). */
  sentenceLabels: Record<number, SentenceLabels>;
  /** sentenceIndex → safety category: -1 (harmful), 0 (neutral), +1 (safe). */
  sentenceScores: Record<number, number>;
  /** Filled in when the user submits the assessment form. */
  assessment?: Assessment;
  completed: boolean;
}

/** App-level persisted state stored in localStorage. */
export interface PersistedState {
  /** Map from entry id to its labeling progress. */
  progress: Record<string, EntryProgress>;
  /** Which entry the user is currently working on (index into filteredEntries). */
  currentEntryIndex: number;
  /** Which sentence within that entry is currently being labeled. */
  currentSentenceIndex: number;
}

/** A sentence extracted from the LLM annotations, ready to display. */
export interface Sentence {
  index: number;      // 0-based position among all sentences in entry
  text: string;
  char_start: number;
  char_end: number;
  message_idx: number;
}
