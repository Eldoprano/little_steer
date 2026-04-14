import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import type { ConversationEntry } from '../types';

interface Props {
  onLoad: (entries: ConversationEntry[]) => void;
}

// ── Types from server ─────────────────────────────────────────────────────────

interface EntryMeta {
  id: string;        // original prompt ID (shared across models)
  uid: string;       // globally unique: source_file + ':' + id
  model: string;
  dataset: string;
  judges: string[];
  files: string[];        // filenames in 2b_labeled/ that contain this entry (may be empty)
  source_file: string;    // filename in 1_generated/ where this entry came from
  has_reasoning: boolean;
}

interface MetaIndex {
  entries: EntryMeta[];
  built_at: string;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const DATASET_COLORS: Record<string, string> = {
  clear_harm: '#E67E80',
  strong_reject: '#E69875',
  xs_test: '#7FBBB3',
  lima: '#D699B6',
};

function datasetColor(ds: string): string {
  return DATASET_COLORS[ds] ?? '#7A8478';
}

function shortModel(model: string): string {
  return model
    .replace('deepseek-r1-distill-llama-8b', 'deepseek-8b')
    .replace('-Reasoning-2512', '')
    .replace('ministral-3-8B', 'ministral-8B')
    .replace('gemma-4-26b-a4b', 'gemma-26B');
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function parseJsonl(text: string): ConversationEntry[] {
  const out: ConversationEntry[] = [];
  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    try {
      const e = JSON.parse(t) as ConversationEntry;
      if (e.id && Array.isArray(e.messages)) out.push(e);
    } catch {}
  }
  return out;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function FilterPill({
  label,
  selected,
  count,
  color,
  onClick,
}: {
  label: string;
  selected: boolean;
  count: number;
  color?: string;
  onClick: () => void;
}) {
  const accent = color ?? '#A7C080';
  return (
    <button
      onClick={onClick}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        padding: '7px 10px',
        background: selected ? `${accent}18` : '#2D353B',
        border: `1px solid ${selected ? accent : '#475258'}`,
        borderRadius: '8px',
        color: selected ? accent : '#7A8478',
        fontSize: '12px',
        cursor: 'pointer',
        textAlign: 'left',
        whiteSpace: 'nowrap',
        transition: 'all 0.1s',
        WebkitTapHighlightColor: 'transparent',
      }}
    >
      {/* checkbox indicator */}
      <span style={{
        width: '12px',
        height: '12px',
        borderRadius: '3px',
        border: `1.5px solid ${selected ? accent : '#7A8478'}`,
        background: selected ? accent : 'transparent',
        flexShrink: 0,
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '8px',
        color: '#2D353B',
      }}>
        {selected ? '✓' : ''}
      </span>
      <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '200px' }}>
        {label}
      </span>
      <span style={{ color: '#7A8478', fontSize: '10px', flexShrink: 0 }}>({count})</span>
    </button>
  );
}

function SectionBox({ children }: { children: React.ReactNode }) {
  return (
    <div style={{
      width: '100%',
      maxWidth: '560px',
      background: '#343F44',
      borderRadius: '14px',
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
      gap: '10px',
    }}>
      {children}
    </div>
  );
}

function SectionLabel({ text, aside }: { text: string; aside?: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{
        color: '#A7C080',
        fontSize: '11px',
        fontWeight: 700,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
      }}>
        {text}
      </div>
      {aside && <div style={{ color: '#7A8478', fontSize: '11px' }}>{aside}</div>}
    </div>
  );
}

function ToggleRow({
  label,
  description,
  enabled,
  onChange,
}: {
  label: string;
  description: string;
  enabled: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div
      style={{ display: 'flex', alignItems: 'flex-start', gap: '10px', cursor: 'pointer' }}
      onClick={() => onChange(!enabled)}
    >
      {/* Toggle pill */}
      <div style={{
        flexShrink: 0,
        width: '36px',
        height: '20px',
        borderRadius: '10px',
        background: enabled ? '#A7C080' : '#475258',
        position: 'relative',
        transition: 'background 0.15s',
        marginTop: '1px',
      }}>
        <div style={{
          position: 'absolute',
          top: '3px',
          left: enabled ? '19px' : '3px',
          width: '14px',
          height: '14px',
          borderRadius: '50%',
          background: '#D3C6AA',
          transition: 'left 0.15s',
        }} />
      </div>
      <div>
        <div style={{ color: '#D3C6AA', fontSize: '13px', fontWeight: 600, lineHeight: 1.3 }}>{label}</div>
        <div style={{ color: '#7A8478', fontSize: '11px', marginTop: '2px', lineHeight: 1.4 }}>{description}</div>
      </div>
    </div>
  );
}

function DistributionChart({
  distribution,
  total,
}: {
  distribution: Map<string, Map<string, number>>;
  total: number;
}) {
  const rows = Array.from(distribution.entries()).sort((a, b) => {
    const sumA = Array.from(a[1].values()).reduce((s, v) => s + v, 0);
    const sumB = Array.from(b[1].values()).reduce((s, v) => s + v, 0);
    return sumB - sumA;
  });
  const allDatasets = [...new Set(rows.flatMap(([, ds]) => Array.from(ds.keys())))].sort();

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        {allDatasets.map(ds => (
          <div key={ds} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '2px', background: datasetColor(ds) }} />
            <span style={{ color: '#9DA9A0', fontSize: '10px' }}>{ds}</span>
          </div>
        ))}
      </div>

      {/* Rows */}
      {rows.map(([model, datasets]) => {
        const rowTotal = Array.from(datasets.values()).reduce((s, v) => s + v, 0);
        const rowPct = Math.round((rowTotal / total) * 100);
        return (
          <div key={model} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {/* Model label */}
            <div style={{
              width: '110px',
              flexShrink: 0,
              color: '#9DA9A0',
              fontSize: '10px',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>
              {shortModel(model)}
            </div>

            {/* Stacked bar */}
            <div style={{
              flex: 1,
              height: '14px',
              borderRadius: '4px',
              overflow: 'hidden',
              display: 'flex',
              background: '#2D353B',
            }}>
              {allDatasets.map(ds => {
                const cnt = datasets.get(ds) ?? 0;
                if (cnt === 0) return null;
                return (
                  <div
                    key={ds}
                    title={`${ds}: ${cnt}`}
                    style={{
                      width: `${(cnt / total) * 100}%`,
                      background: datasetColor(ds),
                      opacity: 0.8,
                    }}
                  />
                );
              })}
            </div>

            {/* Percent */}
            <div style={{
              width: '32px',
              flexShrink: 0,
              color: '#7A8478',
              fontSize: '10px',
              textAlign: 'right',
            }}>
              {rowPct}%
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function DataLoader({ onLoad }: Props) {
  const [meta, setMeta] = useState<MetaIndex | null>(null);
  const [metaLoading, setMetaLoading] = useState(true);
  const [metaError, setMetaError] = useState<string | null>(null);
  const [loadingEntries, setLoadingEntries] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  // Empty set = no filter (all pass through)
  const [selJudges, setSelJudges] = useState<Set<string>>(new Set());
  const [selDatasets, setSelDatasets] = useState<Set<string>>(new Set());
  const [selModels, setSelModels] = useState<Set<string>>(new Set());

  // Sample pool: 'judges' = only entries with at least one LLM judge label; 'all' = all with reasoning
  const [sampleMode, setSampleMode] = useState<'judges' | 'all'>('judges');
  // Work order: sort entries by flat_order from work_order.json
  const [useWorkOrder, setUseWorkOrder] = useState(false);
  // Map from entry ID → position in flat_order (loaded lazily)
  const workOrderMap = useRef<Map<string, number> | null>(null);
  const [workOrderLoaded, setWorkOrderLoaded] = useState(false);

  useEffect(() => {
    fetch('/api/meta')
      .then(r => r.json())
      .then((data: MetaIndex) => setMeta(data))
      .catch(() => setMetaError('Could not load entry index. Is the dev server running?'))
      .finally(() => setMetaLoading(false));
  }, []);

  // Lazily load work order when the toggle is first enabled
  useEffect(() => {
    if (!useWorkOrder || workOrderMap.current !== null) return;
    fetch('/api/work-order')
      .then(r => r.json())
      .then((data: { flat_order: { file: string; id: string }[] }) => {
        const m = new Map<string, number>();
        data.flat_order.forEach((item, i) => m.set(item.id, i));
        workOrderMap.current = m;
        setWorkOrderLoaded(true);
      })
      .catch(() => {
        // If it fails, fall back to natural order silently
        workOrderMap.current = new Map();
        setWorkOrderLoaded(true);
      });
  }, [useWorkOrder]);

  // ── Derived ───────────────────────────────────────────────────────────────

  // Base pool: sampleMode filter only (no category selections).
  // Each category's pills draw from entries passing all OTHER active filters
  // (faceted search) so counts/lists stay relevant to what's selected.
  const basePool = useMemo(() => {
    if (!meta) return [];
    return meta.entries.filter(e => {
      if (!e.has_reasoning) return false;
      if (sampleMode === 'judges' && e.judges.length === 0) return false;
      return true;
    });
  }, [meta, sampleMode]);

  // Entries for model pills: apply judge + dataset filters (not model filter)
  const poolForModels = useMemo(() =>
    basePool.filter(e => {
      if (selJudges.size > 0 && ![...selJudges].every(j => e.judges.includes(j))) return false;
      if (selDatasets.size > 0 && !selDatasets.has(e.dataset)) return false;
      return true;
    }),
  [basePool, selJudges, selDatasets]);

  // Entries for dataset pills: apply judge + model filters (not dataset filter)
  const poolForDatasets = useMemo(() =>
    basePool.filter(e => {
      if (selJudges.size > 0 && ![...selJudges].every(j => e.judges.includes(j))) return false;
      if (selModels.size > 0 && !selModels.has(e.model)) return false;
      return true;
    }),
  [basePool, selJudges, selModels]);

  // Entries for judge pills: apply dataset + model filters (not judge filter)
  const poolForJudges = useMemo(() =>
    basePool.filter(e => {
      if (selDatasets.size > 0 && !selDatasets.has(e.dataset)) return false;
      if (selModels.size > 0 && !selModels.has(e.model)) return false;
      return true;
    }),
  [basePool, selDatasets, selModels]);

  const allModels = useMemo(() =>
    [...new Set(poolForModels.map(e => e.model))].sort(),
  [poolForModels]);

  const allDatasets = useMemo(() =>
    [...new Set(poolForDatasets.map(e => e.dataset))].sort(),
  [poolForDatasets]);

  const allJudges = useMemo(() =>
    [...new Set(poolForJudges.flatMap(e => e.judges))].sort(),
  [poolForJudges]);

  const modelCount = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of poolForModels) m.set(e.model, (m.get(e.model) ?? 0) + 1);
    return m;
  }, [poolForModels]);

  const datasetCount = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of poolForDatasets) m.set(e.dataset, (m.get(e.dataset) ?? 0) + 1);
    return m;
  }, [poolForDatasets]);

  const judgeCount = useMemo(() => {
    const m = new Map<string, number>();
    for (const e of poolForJudges)
      for (const j of e.judges) m.set(j, (m.get(j) ?? 0) + 1);
    return m;
  }, [poolForJudges]);

  // Entries matching ALL active filters — what actually gets loaded
  const filtered = useMemo(() =>
    poolForModels.filter(e => {
      if (selModels.size > 0 && !selModels.has(e.model)) return false;
      return true;
    }),
  [poolForModels, selModels]);

  // All filtered entries are loadable:
  // - entries with files[] → loaded from 2b_labeled (LLM sentence splits)
  // - entries without files[] → loaded from 1_generated (heuristic sentence split)
  const loadable = filtered;
  const nWithLLMSplits = useMemo(() => loadable.filter(e => e.files.length > 0).length, [loadable]);
  const nRaw = loadable.length - nWithLLMSplits;

  // Distribution for chart
  const distribution = useMemo(() => {
    const dist = new Map<string, Map<string, number>>();
    for (const e of loadable) {
      if (!dist.has(e.model)) dist.set(e.model, new Map());
      const ds = dist.get(e.model)!;
      ds.set(e.dataset, (ds.get(e.dataset) ?? 0) + 1);
    }
    return dist;
  }, [loadable]);

  // ── Handlers ──────────────────────────────────────────────────────────────

  const toggle = <T,>(set: Set<T>, val: T): Set<T> => {
    const next = new Set(set);
    next.has(val) ? next.delete(val) : next.add(val);
    return next;
  };

  const handleLoad = useCallback(async () => {
    if (loadable.length === 0) return;
    setLoadingEntries(true);
    setLoadError(null);

    try {
      const allEntries: ConversationEntry[] = [];
      const seenUids = new Set<string>();

      // Entries with LLM labels → fetch from 2b_labeled/ grouped by labeled file
      const labeledGroup = loadable.filter(e => e.files.length > 0);
      if (labeledGroup.length > 0) {
        // Group metas by the labeled file we'll load them from
        const fileToMetas = new Map<string, EntryMeta[]>();
        for (const meta of labeledGroup) {
          const file = meta.files[0];
          if (!fileToMetas.has(file)) fileToMetas.set(file, []);
          fileToMetas.get(file)!.push(meta);
        }
        const results = await Promise.all([...fileToMetas.entries()].map(async ([file, metas]) => {
          const resp = await fetch(`/api/data/${encodeURIComponent(file)}`);
          if (!resp.ok) throw new Error(`Failed to fetch ${file} (HTTP ${resp.status})`);
          const text = await resp.text();
          return { metas, text };
        }));
        for (const { metas, text } of results) {
          // Map original prompt id → uid for this file's entries
          const idToUid = new Map(metas.map(m => [m.id, m.uid]));
          for (const entry of parseJsonl(text)) {
            const uid = idToUid.get(entry.id);
            if (!uid || seenUids.has(uid)) continue;
            seenUids.add(uid);
            allEntries.push({ ...entry, id: uid }); // replace id with uid so tracking is unique per model
          }
        }
      }

      // Entries without LLM labels → fetch raw from 1_generated/ grouped by source file
      const rawGroup = loadable.filter(e => e.files.length === 0);
      if (rawGroup.length > 0) {
        const fileToMetas = new Map<string, EntryMeta[]>();
        for (const meta of rawGroup) {
          if (!fileToMetas.has(meta.source_file)) fileToMetas.set(meta.source_file, []);
          fileToMetas.get(meta.source_file)!.push(meta);
        }
        const results = await Promise.all([...fileToMetas.entries()].map(async ([file, metas]) => {
          const resp = await fetch(`/api/gen-data/${encodeURIComponent(file)}`);
          if (!resp.ok) throw new Error(`Failed to fetch ${file} (HTTP ${resp.status})`);
          const text = await resp.text();
          return { metas, text };
        }));
        for (const { metas, text } of results) {
          const idToUid = new Map(metas.map(m => [m.id, m.uid]));
          for (const entry of parseJsonl(text)) {
            const uid = idToUid.get(entry.id);
            if (!uid || seenUids.has(uid)) continue;
            seenUids.add(uid);
            allEntries.push({ ...entry, id: uid });
          }
        }
      }

      // Accept any entry that has a reasoning message (annotations not required)
      let withReasoning = allEntries.filter(e =>
        e.messages.some(m => m.role === 'reasoning'),
      );
      if (withReasoning.length === 0) {
        setLoadError('No entries with reasoning traces found for the current selection.');
        return;
      }

      // Sort by work order if enabled and loaded
      if (useWorkOrder && workOrderMap.current) {
        const woMap = workOrderMap.current;
        const MAX_POS = 999999;
        withReasoning = [...withReasoning].sort((a, b) => {
          // uid = "source_file:original_id" — extract original id for work-order lookup
          const aOrigId = a.id.slice(a.id.indexOf(':') + 1);
          const bOrigId = b.id.slice(b.id.indexOf(':') + 1);
          const pa = woMap.get(aOrigId) ?? MAX_POS;
          const pb = woMap.get(bOrigId) ?? MAX_POS;
          return pa - pb;
        });
      }

      onLoad(withReasoning);
    } catch (e) {
      setLoadError(String(e));
    } finally {
      setLoadingEntries(false);
    }
  }, [loadable, onLoad, useWorkOrder, workOrderLoaded]);

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div style={{
      height: '100%',
      overflowY: 'auto',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      background: '#2D353B',
      padding: '24px 16px 40px',
      gap: '14px',
    }}>
      {/* Title */}
      <div style={{ textAlign: 'center', paddingTop: '8px' }}>
        <h1 style={{ color: '#D3C6AA', fontSize: '22px', fontWeight: 700, margin: 0 }}>
          Reasoning Labeler
        </h1>
        <p style={{ color: '#7A8478', fontSize: '13px', marginTop: '6px', margin: '6px 0 0' }}>
          Configure which entries to label
        </p>
      </div>

      {/* Loading */}
      {metaLoading && (
        <div style={{ color: '#7A8478', fontSize: '13px', padding: '32px 0' }}>
          Scanning data directories...
        </div>
      )}

      {/* Error */}
      {metaError && (
        <div style={{
          maxWidth: '560px', width: '100%',
          background: '#3B2D2E', border: '1px solid #E67E80',
          borderRadius: '10px', padding: '12px 16px',
          color: '#E67E80', fontSize: '13px', lineHeight: 1.5,
        }}>
          {metaError}
          <div style={{ marginTop: '8px', color: '#7A8478', fontSize: '12px' }}>
            Run <code style={{ background: '#2D353B', borderRadius: '3px', padding: '1px 5px' }}>npm run dev</code> from the <code style={{ background: '#2D353B', borderRadius: '3px', padding: '1px 5px' }}>2c_human_labeling/</code> directory.
          </div>
        </div>
      )}

      {meta && (
        <>
          {/* ── Sample pool + Work order toggles ──────────────────────── */}
          <SectionBox>
            <SectionLabel text="Options" />
            <ToggleRow
              label="Judge-labeled only"
              description={
                sampleMode === 'judges'
                  ? 'Showing entries already labeled by at least one LLM judge — has sentence splits for comparison.'
                  : 'Showing ALL entries with a reasoning trace, including those not yet labeled by any judge.'
              }
              enabled={sampleMode === 'judges'}
              onChange={v => setSampleMode(v ? 'judges' : 'all')}
            />
            <ToggleRow
              label="Work order"
              description="Sort entries by the breadth-first sampling order from work_order.json."
              enabled={useWorkOrder}
              onChange={v => {
                setUseWorkOrder(v);
              }}
            />
          </SectionBox>

          {/* ── Judges ─────────────────────────────────────────────────── */}
          <SectionBox>
            <SectionLabel
              text="Judges"
              aside={selJudges.size === 0 ? 'any (no filter)' : `${selJudges.size} selected`}
            />
            <p style={{ color: '#7A8478', fontSize: '11px', margin: 0, lineHeight: 1.5 }}>
              {sampleMode === 'judges'
                ? 'Select which LLM judges must have labeled an entry (intersection — all selected judges must have labeled it).'
                : 'Filter by specific judges (intersection). Only relevant when "Judge-labeled only" is on.'}
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
              {allJudges.map(j => (
                <FilterPill
                  key={j}
                  label={j}
                  selected={selJudges.has(j)}
                  count={judgeCount.get(j) ?? 0}
                  color="#7FBBB3"
                  onClick={() => setSelJudges(prev => toggle(prev, j))}
                />
              ))}
            </div>
          </SectionBox>

          {/* ── Datasets + Models ──────────────────────────────────────── */}
          <div style={{
            display: 'flex',
            gap: '12px',
            width: '100%',
            maxWidth: '560px',
            flexWrap: 'wrap',
          }}>
            {/* Datasets */}
            <div style={{
              flex: '1 1 200px',
              background: '#343F44',
              borderRadius: '14px',
              padding: '16px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
            }}>
              <SectionLabel
                text="Datasets"
                aside={selDatasets.size === 0 ? 'any' : `${selDatasets.size}`}
              />
              <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                {allDatasets.map(d => (
                  <FilterPill
                    key={d}
                    label={d}
                    selected={selDatasets.has(d)}
                    count={datasetCount.get(d) ?? 0}
                    color={datasetColor(d)}
                    onClick={() => setSelDatasets(prev => toggle(prev, d))}
                  />
                ))}
              </div>
            </div>

            {/* Models */}
            <div style={{
              flex: '1 1 200px',
              background: '#343F44',
              borderRadius: '14px',
              padding: '16px',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px',
            }}>
              <SectionLabel
                text="Models"
                aside={selModels.size === 0 ? 'any' : `${selModels.size}`}
              />
              <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
                {allModels.map(m => (
                  <FilterPill
                    key={m}
                    label={shortModel(m)}
                    selected={selModels.has(m)}
                    count={modelCount.get(m) ?? 0}
                    onClick={() => setSelModels(prev => toggle(prev, m))}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* ── Selection stats + distribution ─────────────────────────── */}
          <SectionBox>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <SectionLabel text="Selection" />
              <div style={{ textAlign: 'right' }}>
                <span style={{ color: '#D3C6AA', fontWeight: 700, fontSize: '20px' }}>
                  {loadable.length}
                </span>
                <span style={{ color: '#7A8478', fontSize: '13px' }}> entries</span>
                <div style={{ color: '#7A8478', fontSize: '10px', marginTop: '3px' }}>
                  {nWithLLMSplits > 0 && (
                    <span style={{ color: '#A7C080' }}>{nWithLLMSplits} with LLM labels</span>
                  )}
                  {nWithLLMSplits > 0 && nRaw > 0 && <span> · </span>}
                  {nRaw > 0 && (
                    <span style={{ color: '#DBBC7F' }}>{nRaw} auto-split</span>
                  )}
                </div>
              </div>
            </div>

            {loadable.length > 0 ? (
              <DistributionChart distribution={distribution} total={loadable.length} />
            ) : (
              <div style={{ color: '#7A8478', fontSize: '12px', textAlign: 'center', padding: '12px 0' }}>
                No entries match the current filters.
              </div>
            )}
          </SectionBox>

          {/* ── Load button ────────────────────────────────────────────── */}
          <div style={{ width: '100%', maxWidth: '560px' }}>
            <button
              onClick={handleLoad}
              disabled={loadable.length === 0 || loadingEntries}
              style={{
                width: '100%',
                minHeight: '54px',
                background: loadable.length > 0 && !loadingEntries ? '#A7C080' : '#3D484D',
                border: 'none',
                borderRadius: '12px',
                color: loadable.length > 0 && !loadingEntries ? '#2D353B' : '#7A8478',
                fontSize: '16px',
                fontWeight: 700,
                cursor: loadable.length > 0 && !loadingEntries ? 'pointer' : 'not-allowed',
                transition: 'all 0.15s',
                WebkitTapHighlightColor: 'transparent',
              }}
            >
              {loadingEntries
                ? 'Loading...'
                : loadable.length > 0
                  ? `Start labeling ${loadable.length} entries →`
                  : 'No entries to load'}
            </button>
          </div>

          {loadError && (
            <div style={{
              maxWidth: '560px', width: '100%',
              background: '#3B2D2E', border: '1px solid #E67E80',
              borderRadius: '10px', padding: '12px 16px',
              color: '#E67E80', fontSize: '13px',
            }}>
              {loadError}
            </div>
          )}
        </>
      )}
    </div>
  );
}
