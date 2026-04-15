import type { TrajectoryType, AlignmentType } from './types';
import taxonomyData from '../../taxonomy.json';

// ── Types ─────────────────────────────────────────────────────────────────────

export interface GroupInfo {
  name: string;
  /** Light bg color (used for sentence highlights in trace). */
  color: string;
  border: string;
  text: string;
  /** Dark-theme variants for the pill buttons. */
  darkBg: string;
  darkBorder: string;
  darkText: string;
  labels: string[];
}

export interface AssessmentOption<T extends string> {
  value: T;
  label: string;
  description: string;
}

// ── Derived from taxonomy.json ────────────────────────────────────────────────

export const LABEL_GROUPS: Record<string, GroupInfo> = Object.fromEntries(
  taxonomyData.groups.map((g) => [
    g.id,
    {
      name: g.name,
      color: g.colors.light.bg,
      border: g.colors.light.border,
      text: g.colors.light.text,
      darkBg: g.colors.dark.bg,
      darkBorder: g.colors.dark.border,
      darkText: g.colors.dark.text,
      labels: g.labels.map((l) => l.id),
    },
  ])
);

export const LABEL_DISPLAY_NAMES: Record<string, string> = Object.fromEntries(
  taxonomyData.groups.flatMap((g) => g.labels.map((l) => [l.id, l.display]))
);

export const LABEL_DESCRIPTIONS: Record<string, string> = Object.fromEntries(
  taxonomyData.groups.flatMap((g) => g.labels.map((l) => [l.id, l.description]))
);

/** Map label → group ID (e.g. "flagAsHarmful" → "B"). */
export const LABEL_TO_GROUP: Record<string, string> = {};
for (const [gid, g] of Object.entries(LABEL_GROUPS)) {
  for (const lbl of g.labels) LABEL_TO_GROUP[lbl] = gid;
}

const _fallbackGroupId = taxonomyData.groups[taxonomyData.groups.length - 1].id;

export function getLabelGroup(label: string): GroupInfo {
  const gid = LABEL_TO_GROUP[label] ?? _fallbackGroupId;
  return LABEL_GROUPS[gid];
}

export const TRAJECTORY_OPTIONS: AssessmentOption<TrajectoryType>[] =
  taxonomyData.trajectories.map((t) => ({
    value: t.value as TrajectoryType,
    label: t.label,
    description: t.description,
  }));

export const ALIGNMENT_OPTIONS: AssessmentOption<AlignmentType>[] =
  taxonomyData.alignments.map((a) => ({
    value: a.value as AlignmentType,
    label: a.label,
    description: a.description,
  }));
