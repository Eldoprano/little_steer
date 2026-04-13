import type { ConversationEntry } from '../types';
import type { PersistedState } from '../types';
import { exportAsJsonl } from '../store';
import { LABEL_GROUPS, getLabelGroup, LABEL_DISPLAY_NAMES } from '../taxonomy';

interface Props {
  entries: ConversationEntry[];
  state: PersistedState;
  onClose: () => void;
  onReset: () => void;
  onClearData: () => void;
}

function downloadFile(content: string, filename: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export default function StatsView({ entries, state, onClose, onReset, onClearData }: Props) {
  const total = entries.length;
  const completed = entries.filter((e) => state.progress[e.id]?.completed).length;
  const inProgress = entries.filter(
    (e) => !state.progress[e.id]?.completed && Object.keys(state.progress[e.id]?.sentenceLabels ?? {}).length > 0,
  ).length;

  // Label frequency across all completed entries
  const labelCounts: Record<string, number> = {};
  for (const entry of entries) {
    const prog = state.progress[entry.id];
    if (!prog?.completed) continue;
    for (const labels of Object.values(prog.sentenceLabels)) {
      for (const lbl of labels) {
        if (lbl !== 'NONE') {
          labelCounts[lbl] = (labelCounts[lbl] ?? 0) + 1;
        }
      }
    }
  }

  const topLabels = Object.entries(labelCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);

  const maxCount = topLabels[0]?.[1] ?? 1;

  const handleExport = () => {
    const jsonl = exportAsJsonl(
      entries,
      state.progress as Parameters<typeof exportAsJsonl>[1],
    );
    if (!jsonl.trim()) {
      alert('No completed entries to export yet.');
      return;
    }
    const ts = new Date().toISOString().slice(0, 10);
    downloadFile(jsonl, `human_labels_${ts}.jsonl`, 'application/x-ndjson');
  };

  const handleExportProgress = () => {
    const json = JSON.stringify({ entries: entries.map((e) => ({ id: e.id })), state }, null, 2);
    downloadFile(json, 'labeling_progress.json', 'application/json');
  };

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 150,
        background: 'rgba(0,0,0,0.85)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '16px',
      }}
    >
      <div
        style={{
          background: '#0f172a',
          border: '1px solid #1e293b',
          borderRadius: '20px',
          width: '100%',
          maxWidth: '560px',
          maxHeight: '85dvh',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          boxShadow: '0 25px 60px rgba(0,0,0,0.7)',
          userSelect: 'none',
          WebkitUserSelect: 'none',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '14px 16px',
            borderBottom: '1px solid #1e293b',
            flexShrink: 0,
          }}
        >
          <span style={{ color: '#f1f5f9', fontSize: '16px', fontWeight: 700 }}>
            Labeling Stats
          </span>
          <button
            onClick={onClose}
            style={{
              minHeight: '44px',
              minWidth: '44px',
              background: 'transparent',
              border: 'none',
              color: '#64748b',
              fontSize: '22px',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              WebkitTapHighlightColor: 'transparent',
            }}
          >
            ×
          </button>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '14px 16px' }}>
          {/* Progress overview */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr 1fr',
              gap: '8px',
              marginBottom: '16px',
            }}
          >
            {[
              { label: 'Total', value: total, color: '#94a3b8' },
              { label: 'Completed', value: completed, color: '#22c55e' },
              { label: 'In Progress', value: inProgress, color: '#f59e0b' },
            ].map((stat) => (
              <div
                key={stat.label}
                style={{
                  background: '#1e293b',
                  borderRadius: '12px',
                  padding: '12px',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '24px', fontWeight: 800, color: stat.color }}>
                  {stat.value}
                </div>
                <div style={{ fontSize: '11px', color: '#64748b', marginTop: '2px' }}>
                  {stat.label}
                </div>
              </div>
            ))}
          </div>

          {/* Overall progress bar */}
          <div style={{ marginBottom: '20px' }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: '11px',
                color: '#64748b',
                marginBottom: '6px',
              }}
            >
              <span>Overall completion</span>
              <span>{total > 0 ? Math.round((completed / total) * 100) : 0}%</span>
            </div>
            <div
              style={{
                height: '8px',
                background: '#1e293b',
                borderRadius: '4px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  width: `${total > 0 ? (completed / total) * 100 : 0}%`,
                  height: '100%',
                  background: '#22c55e',
                  borderRadius: '4px',
                  transition: 'width 0.3s',
                }}
              />
            </div>
          </div>

          {/* Top labels */}
          {topLabels.length > 0 && (
            <div style={{ marginBottom: '20px' }}>
              <div
                style={{
                  fontSize: '11px',
                  fontWeight: 700,
                  color: '#475569',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  marginBottom: '10px',
                }}
              >
                Top Labels (completed)
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {topLabels.map(([label, count]) => {
                  const group = getLabelGroup(label);
                  const pct = Math.round((count / maxCount) * 100);
                  return (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div
                        style={{
                          fontSize: '11px',
                          color: group.darkText,
                          background: group.darkBg,
                          border: `1px solid ${group.darkBorder}`,
                          borderRadius: '6px',
                          padding: '2px 6px',
                          whiteSpace: 'nowrap',
                          minWidth: '120px',
                        }}
                      >
                        {LABEL_DISPLAY_NAMES[label] ?? label}
                      </div>
                      <div
                        style={{
                          flex: 1,
                          height: '8px',
                          background: '#1e293b',
                          borderRadius: '4px',
                          overflow: 'hidden',
                        }}
                      >
                        <div
                          style={{
                            width: `${pct}%`,
                            height: '100%',
                            background: group.darkBorder,
                            borderRadius: '4px',
                          }}
                        />
                      </div>
                      <span style={{ fontSize: '11px', color: '#64748b', minWidth: '24px', textAlign: 'right' }}>
                        {count}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Entry list */}
          <div>
            <div
              style={{
                fontSize: '11px',
                fontWeight: 700,
                color: '#475569',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                marginBottom: '8px',
              }}
            >
              Responses
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
              {entries.map((e, i) => {
                const prog = state.progress[e.id];
                const done = prog?.completed;
                const started = !done && Object.keys(prog?.sentenceLabels ?? {}).length > 0;
                const isCurrent = i === state.currentEntryIndex;

                return (
                  <div
                    key={e.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      padding: '7px 10px',
                      background: isCurrent ? '#1e293b' : '#0f172a',
                      borderRadius: '8px',
                      border: `1px solid ${isCurrent ? '#334155' : '#1e293b'}`,
                    }}
                  >
                    <span
                      style={{
                        width: '20px',
                        height: '20px',
                        borderRadius: '50%',
                        background: done ? '#15803d' : started ? '#92400e' : '#1e293b',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '10px',
                        flexShrink: 0,
                      }}
                    >
                      {done ? '✓' : started ? '…' : ''}
                    </span>
                    <span style={{ fontSize: '11px', color: '#64748b', whiteSpace: 'nowrap' }}>
                      {i + 1}.
                    </span>
                    <span
                      style={{
                        fontSize: '11px',
                        color: isCurrent ? '#f1f5f9' : '#94a3b8',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        flex: 1,
                      }}
                    >
                      {e.model || e.id.slice(0, 12)}
                    </span>
                    {isCurrent && (
                      <span style={{ fontSize: '9px', color: '#6366f1', fontWeight: 700, textTransform: 'uppercase' }}>
                        current
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Footer buttons */}
        <div
          style={{
            padding: '12px 16px',
            borderTop: '1px solid #1e293b',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px',
            flexShrink: 0,
          }}
        >
          <button
            onClick={handleExport}
            style={{
              minHeight: '48px',
              background: '#4f46e5',
              border: 'none',
              borderRadius: '10px',
              color: '#fff',
              fontSize: '14px',
              fontWeight: 700,
              cursor: 'pointer',
              WebkitTapHighlightColor: 'transparent',
            }}
          >
            Export Completed Labels (.jsonl)
          </button>

          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={handleExportProgress}
              style={{
                flex: 1,
                minHeight: '44px',
                background: '#1e293b',
                border: '1px solid #334155',
                borderRadius: '10px',
                color: '#94a3b8',
                fontSize: '12px',
                cursor: 'pointer',
                WebkitTapHighlightColor: 'transparent',
              }}
            >
              Save Progress Backup
            </button>
            <button
              onClick={() => {
                if (window.confirm('Reset all labeling progress? (entries stay loaded)')) {
                  onReset();
                }
              }}
              style={{
                flex: 1,
                minHeight: '44px',
                background: '#3b1212',
                border: '1px solid #ef4444',
                borderRadius: '10px',
                color: '#fca5a5',
                fontSize: '12px',
                cursor: 'pointer',
                WebkitTapHighlightColor: 'transparent',
              }}
            >
              Reset Progress
            </button>
          </div>

          <button
            onClick={() => {
              if (window.confirm('Clear all data and start fresh?')) {
                onClearData();
              }
            }}
            style={{
              minHeight: '44px',
              background: 'transparent',
              border: '1px solid #1e293b',
              borderRadius: '10px',
              color: '#475569',
              fontSize: '12px',
              cursor: 'pointer',
              WebkitTapHighlightColor: 'transparent',
            }}
          >
            Clear All Data & Reload Files
          </button>
        </div>
      </div>
    </div>
  );
}
