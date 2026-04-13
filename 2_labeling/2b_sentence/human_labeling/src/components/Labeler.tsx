import { useState, useRef, useCallback, useEffect } from 'react';
import type { ConversationEntry, Sentence } from '../types';
import {
  LABEL_GROUPS,
  LABEL_DISPLAY_NAMES,
  LABEL_DESCRIPTIONS,
  getLabelGroup,
} from '../taxonomy';
import { getSentences } from '../store';

// ── Long-press hook ──────────────────────────────────────────────────────────

function useLongPress(onLongPress: () => void, ms = 650) {
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const triggered = useRef(false);

  const start = useCallback(
    (e?: React.TouchEvent | React.MouseEvent) => {
      if (e && 'touches' in e) e.preventDefault();
      triggered.current = false;
      timer.current = setTimeout(() => {
        triggered.current = true;
        onLongPress();
      }, ms);
    },
    [onLongPress, ms],
  );

  const cancel = useCallback(() => {
    if (timer.current) {
      clearTimeout(timer.current);
      timer.current = null;
    }
  }, []);

  return {
    onMouseDown: () => start(),
    onMouseUp: cancel,
    onMouseLeave: cancel,
    onTouchStart: (e: React.TouchEvent) => start(e),
    onTouchEnd: cancel,
    onTouchCancel: cancel,
    onContextMenu: (e: React.MouseEvent) => e.preventDefault(),
  };
}

// ── Reasoning trace ──────────────────────────────────────────────────────────

function ReasoningTrace({
  entry,
  sentences,
  currentIdx,
  sentenceLabels,
  onJump,
}: {
  entry: ConversationEntry;
  sentences: Sentence[];
  currentIdx: number;
  sentenceLabels: Record<number, string[]>;
  onJump: (idx: number) => void;
}) {
  const activeSentRef = useRef<HTMLSpanElement>(null);
  const reasoningIdx = entry.messages.findIndex((m) => m.role === 'reasoning');
  const reasoning = reasoningIdx >= 0 ? entry.messages[reasoningIdx].content : '';

  useEffect(() => {
    activeSentRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, [currentIdx]);

  if (!reasoning) {
    return <div style={{ color: '#7A8478', fontSize: '14px' }}>No reasoning trace.</div>;
  }

  type Seg =
    | { kind: 'plain'; text: string; key: string }
    | { kind: 'sentence'; text: string; idx: number; key: string };

  const segs: Seg[] = [];
  let pos = 0;
  for (const s of sentences) {
    if (pos < s.char_start) {
      segs.push({ kind: 'plain', text: reasoning.slice(pos, s.char_start), key: `p${pos}` });
    }
    segs.push({ kind: 'sentence', text: s.text, idx: s.index, key: `s${s.index}` });
    pos = s.char_end;
  }
  if (pos < reasoning.length) {
    segs.push({ kind: 'plain', text: reasoning.slice(pos), key: `p${pos}` });
  }

  return (
    <div style={{ fontSize: '13px', lineHeight: '1.75', color: '#D3C6AA', whiteSpace: 'pre-wrap' }}>
      {segs.map((seg) => {
        if (seg.kind === 'plain') {
          return <span key={seg.key}>{seg.text}</span>;
        }

        const labels = sentenceLabels[seg.idx];
        const primaryLabel = labels?.[0];
        const group = primaryLabel && primaryLabel !== 'NONE' ? getLabelGroup(primaryLabel) : null;
        const isActive = seg.idx === currentIdx;
        const labeled = (labels?.length ?? 0) > 0;

        let bg = 'transparent';
        let border = 'none';
        let color = '#9DA9A0';
        let outline = 'none';
        let shadow = 'none';

        if (isActive) {
          outline = '2px solid #A7C080';
          bg = 'rgba(167,192,128,0.12)';
          color = '#D3C6AA';
          shadow = '0 0 0 1px rgba(167,192,128,0.25)';
        } else if (labeled && group) {
          bg = group.darkBg;
          border = `1px solid ${group.darkBorder}`;
          color = group.darkText;
        } else if (labeled) {
          // NONE selected
          bg = '#343F44';
          border = '1px solid #475258';
          color = '#7A8478';
        }

        return (
          <span
            key={seg.key}
            ref={isActive ? activeSentRef : undefined}
            onClick={() => onJump(seg.idx)}
            style={{
              background: bg,
              border: border !== 'none' ? border : undefined,
              outline: outline !== 'none' ? outline : undefined,
              boxShadow: shadow !== 'none' ? shadow : undefined,
              borderRadius: '4px',
              color,
              cursor: 'pointer',
              padding: labeled || isActive ? '0 2px' : undefined,
              transition: 'background 0.15s, color 0.15s',
            }}
          >
            {seg.text}
          </span>
        );
      })}
    </div>
  );
}

// ── Chat display ─────────────────────────────────────────────────────────────

function ChatDisplay({ entry }: { entry: ConversationEntry }) {
  const userMsg = entry.messages.find((m) => m.role === 'user');
  const assistantMsg = entry.messages.find((m) => m.role === 'assistant');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {userMsg && (
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <div
            style={{
              background: '#3D484D',
              border: '1px solid #7FBBB3',
              color: '#D3C6AA',
              borderRadius: '16px 16px 4px 16px',
              padding: '8px 12px',
              maxWidth: '92%',
              fontSize: '13px',
              lineHeight: '1.6',
              whiteSpace: 'pre-wrap',
            }}
          >
            <div
              style={{
                fontSize: '9px',
                color: '#7FBBB3',
                marginBottom: '4px',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
              }}
            >
              User
            </div>
            {userMsg.content}
          </div>
        </div>
      )}

      {assistantMsg && (
        <details>
          <summary
            style={{
              color: '#7A8478',
              fontSize: '11px',
              cursor: 'pointer',
              padding: '3px 0',
              listStyle: 'none',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <span style={{ color: '#475258' }}>▶</span> Final response
          </summary>
          <div
            style={{
              marginTop: '6px',
              background: '#343F44',
              border: '1px solid #475258',
              borderRadius: '4px 14px 14px 14px',
              padding: '8px 12px',
              color: '#9DA9A0',
              fontSize: '12px',
              lineHeight: '1.6',
              whiteSpace: 'pre-wrap',
            }}
          >
            {assistantMsg.content}
          </div>
        </details>
      )}
    </div>
  );
}

// ── Single label pill button ──────────────────────────────────────────────────

function LabelButton({
  label,
  priority,
  selected,
  disabled,
  onToggle,
  onDescRequest,
}: {
  label: string;
  priority: number | null;
  selected: boolean;
  disabled: boolean;
  onToggle: () => void;
  onDescRequest: (text: string) => void;
}) {
  const group = getLabelGroup(label);

  const longPress = useLongPress(() => {
    onDescRequest(
      `${LABEL_DISPLAY_NAMES[label] ?? label}\n\n${LABEL_DESCRIPTIONS[label] ?? ''}`,
    );
  });

  const bg = selected ? group.darkBorder : group.darkBg;
  const textColor = selected ? '#232A2E' : group.darkText;
  const borderColor = group.darkBorder;

  return (
    <button
      onClick={() => { if (!disabled) onToggle(); }}
      {...longPress}
      style={{
        minHeight: '44px',
        padding: '5px 10px',
        background: disabled && !selected ? '#232A2E' : bg,
        border: `1.5px solid ${disabled && !selected ? '#343F44' : borderColor}`,
        borderRadius: '22px',
        color: disabled && !selected ? '#475258' : textColor,
        fontSize: '12px',
        fontWeight: selected ? 600 : 400,
        cursor: disabled && !selected ? 'not-allowed' : 'pointer',
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
        transition: 'all 0.1s',
        WebkitTapHighlightColor: 'transparent',
        whiteSpace: 'nowrap',
      }}
    >
      {priority !== null && (
        <span
          style={{
            background: 'rgba(0,0,0,0.35)',
            borderRadius: '50%',
            width: '16px',
            height: '16px',
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '10px',
            fontWeight: 800,
            flexShrink: 0,
            color: '#D3C6AA',
          }}
        >
          {priority}
        </span>
      )}
      {LABEL_DISPLAY_NAMES[label] ?? label}
    </button>
  );
}

// ── Label panel (all groups + NONE) ──────────────────────────────────────────

function LabelPanel({
  currentLabels,
  onToggle,
  onDescRequest,
}: {
  currentLabels: string[];
  onToggle: (label: string) => void;
  onDescRequest: (text: string) => void;
}) {
  const isNone = currentLabels.length === 1 && currentLabels[0] === 'NONE';
  const maxReached = !isNone && currentLabels.length >= 3;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {/* NONE button */}
      <button
        onClick={() => onToggle('NONE')}
        style={{
          minHeight: '44px',
          background: isNone ? '#343F44' : '#232A2E',
          border: `1.5px solid ${isNone ? '#9DA9A0' : '#343F44'}`,
          borderRadius: '8px',
          color: isNone ? '#D3C6AA' : '#7A8478',
          fontSize: '12px',
          fontWeight: isNone ? 700 : 400,
          cursor: 'pointer',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          transition: 'all 0.1s',
          WebkitTapHighlightColor: 'transparent',
        }}
      >
        {isNone ? '✓ NONE' : 'NONE'}
      </button>

      {/* Groups */}
      {Object.entries(LABEL_GROUPS).map(([gid, group]) => (
        <div key={gid}>
          <div
            style={{
              fontSize: '9px',
              fontWeight: 700,
              color: group.darkBorder,
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
              marginBottom: '4px',
            }}
          >
            {group.name}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {group.labels.map((label) => {
              const idx = isNone ? -1 : currentLabels.indexOf(label);
              const sel = idx !== -1;
              const pri = sel ? idx + 1 : null;
              const dis = !sel && maxReached;
              return (
                <LabelButton
                  key={label}
                  label={label}
                  priority={pri}
                  selected={sel}
                  disabled={dis}
                  onToggle={() => onToggle(label)}
                  onDescRequest={onDescRequest}
                />
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Progress bar ─────────────────────────────────────────────────────────────

function ProgressBar({
  value,
  max,
  color = '#A7C080',
}: {
  value: number;
  max: number;
  color?: string;
}) {
  const pct = max === 0 ? 0 : Math.min(100, Math.round((value / max) * 100));
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div
        style={{
          flex: 1,
          height: '5px',
          background: '#343F44',
          borderRadius: '3px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${pct}%`,
            height: '100%',
            background: color,
            borderRadius: '3px',
            transition: 'width 0.3s ease',
          }}
        />
      </div>
      <span style={{ fontSize: '11px', color: '#7A8478', whiteSpace: 'nowrap', minWidth: '48px', textAlign: 'right' }}>
        {value}/{max} · {pct}%
      </span>
    </div>
  );
}

// ── Description tooltip overlay ───────────────────────────────────────────────

function DescTooltip({ text, onClose }: { text: string; onClose: () => void }) {
  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 200,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'rgba(35,42,46,0.85)',
        padding: '24px',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#343F44',
          border: '1px solid #475258',
          borderRadius: '16px',
          padding: '20px 22px',
          maxWidth: '400px',
          color: '#D3C6AA',
          fontSize: '13px',
          lineHeight: '1.65',
          boxShadow: '0 20px 60px rgba(0,0,0,0.5)',
          whiteSpace: 'pre-wrap',
        }}
      >
        <p style={{ margin: '0 0 16px' }}>{text}</p>
        <button
          onClick={onClose}
          style={{
            minHeight: '44px',
            padding: '0 20px',
            background: '#475258',
            border: 'none',
            borderRadius: '8px',
            color: '#D3C6AA',
            fontSize: '13px',
            cursor: 'pointer',
          }}
        >
          Got it
        </button>
      </div>
    </div>
  );
}

// ── Main Labeler ──────────────────────────────────────────────────────────────

interface Props {
  entry: ConversationEntry;
  entryIndex: number;
  totalEntries: number;
  completedCount: number;
  sentenceIndex: number;
  sentenceLabels: Record<number, string[]>;
  onLabelSentence: (labels: string[]) => void;
  onNavigate: (direction: 'back' | 'next') => void;
  onJumpToSentence: (idx: number) => void;
  onShowAssessment: () => void;
  onShowStats: () => void;
  allDone: boolean;
}

export default function Labeler({
  entry,
  entryIndex,
  totalEntries,
  completedCount,
  sentenceIndex,
  sentenceLabels,
  onLabelSentence,
  onNavigate,
  onJumpToSentence,
  onShowAssessment,
  onShowStats,
  allDone,
}: Props) {
  const [descTooltip, setDescTooltip] = useState<string | null>(null);

  const sentences = getSentences(entry);
  const currentSentence = sentences[sentenceIndex] ?? null;
  const currentLabels: string[] = sentenceLabels[sentenceIndex] ?? [];

  const labeledCount = Object.values(sentenceLabels).filter((l) => l.length > 0).length;
  const hasLabel = currentLabels.length > 0;

  const handleToggle = useCallback(
    (label: string) => {
      if (label === 'NONE') {
        onLabelSentence(currentLabels.length === 1 && currentLabels[0] === 'NONE' ? [] : ['NONE']);
        return;
      }
      const isNone = currentLabels.length === 1 && currentLabels[0] === 'NONE';
      if (isNone) {
        onLabelSentence([label]);
        return;
      }
      const idx = currentLabels.indexOf(label);
      if (idx === -1) {
        if (currentLabels.length >= 3) return;
        onLabelSentence([...currentLabels, label]);
      } else {
        onLabelSentence(currentLabels.filter((l) => l !== label));
      }
    },
    [currentLabels, onLabelSentence],
  );

  return (
    <div
      style={{
        height: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        background: '#2D353B',
        overflow: 'hidden',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
    >
      {/* ── Top bar ── */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 14px',
          borderBottom: '1px solid #3D484D',
          flexShrink: 0,
          minHeight: '44px',
          gap: '12px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
          <span style={{ color: '#D3C6AA', fontSize: '13px', fontWeight: 600, whiteSpace: 'nowrap' }}>
            {entryIndex + 1} / {totalEntries}
          </span>
          {entry.model && (
            <span
              style={{
                background: '#343F44',
                border: '1px solid #475258',
                borderRadius: '10px',
                padding: '2px 7px',
                color: '#7A8478',
                fontSize: '10px',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                maxWidth: '200px',
              }}
            >
              {entry.model}
            </span>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', color: '#7A8478', whiteSpace: 'nowrap' }}>
            {completedCount} labeled
          </span>
          <button
            onClick={onShowStats}
            style={{
              minHeight: '36px',
              padding: '0 12px',
              background: '#343F44',
              border: '1px solid #475258',
              borderRadius: '8px',
              color: '#9DA9A0',
              fontSize: '12px',
              cursor: 'pointer',
              WebkitTapHighlightColor: 'transparent',
            }}
          >
            Stats
          </button>
        </div>
      </div>

      {/* ── Body: two columns ── */}
      <div
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        {/* LEFT: prompt + reasoning */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            borderRight: '1px solid #3D484D',
            overflow: 'hidden',
            minHeight: 0,
          }}
        >
          {/* Prompt / conversation */}
          <div
            style={{
              flexShrink: 0,
              maxHeight: '38%',
              overflowY: 'auto',
              padding: '10px 12px',
              borderBottom: '1px solid #3D484D',
            }}
          >
            <div className="col-label">Conversation</div>
            <ChatDisplay entry={entry} />
          </div>

          {/* Reasoning trace */}
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '10px 12px',
              minHeight: 0,
            }}
          >
            <div className="col-label" style={{ marginBottom: '8px' }}>
              Reasoning trace · tap to jump
            </div>
            <ReasoningTrace
              entry={entry}
              sentences={sentences}
              currentIdx={sentenceIndex}
              sentenceLabels={sentenceLabels}
              onJump={onJumpToSentence}
            />
          </div>
        </div>

        {/* RIGHT: current sentence + labels + nav */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            minHeight: 0,
          }}
        >
          {/* Sentence header */}
          <div
            style={{
              flexShrink: 0,
              padding: '8px 12px 6px',
              borderBottom: '1px solid #3D484D',
            }}
          >
            <ProgressBar value={labeledCount} max={sentences.length} />
            <div style={{ marginTop: '7px' }}>
              <span className="col-label">
                Sentence {sentenceIndex + 1} / {sentences.length}
              </span>
              {currentSentence && (
                <div
                  style={{
                    marginTop: '5px',
                    background: 'rgba(167,192,128,0.07)',
                    border: '1px solid rgba(167,192,128,0.2)',
                    borderRadius: '8px',
                    padding: '7px 9px',
                    color: '#D3C6AA',
                    fontSize: '13px',
                    lineHeight: '1.5',
                    maxHeight: '68px',
                    overflowY: 'auto',
                  }}
                >
                  {currentSentence.text}
                </div>
              )}
            </div>
          </div>

          {/* Labels */}
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '8px 12px',
              minHeight: 0,
            }}
          >
            <LabelPanel
              currentLabels={currentLabels}
              onToggle={handleToggle}
              onDescRequest={(text) => setDescTooltip(text)}
            />
          </div>

          {/* Navigation */}
          <div
            style={{
              flexShrink: 0,
              padding: '8px 12px',
              borderTop: '1px solid #3D484D',
              display: 'flex',
              gap: '8px',
            }}
          >
            <button
              onClick={() => onNavigate('back')}
              disabled={sentenceIndex === 0}
              style={navBtnStyle(sentenceIndex === 0, false, 1)}
            >
              ← Back
            </button>

            {allDone ? (
              <button
                onClick={onShowAssessment}
                style={navBtnStyle(false, true, 2)}
              >
                Assessment →
              </button>
            ) : (
              <button
                onClick={() => { if (hasLabel) onNavigate('next'); }}
                disabled={!hasLabel}
                style={navBtnStyle(!hasLabel, hasLabel, 2)}
              >
                Next →
              </button>
            )}
          </div>
        </div>
      </div>

      {descTooltip && (
        <DescTooltip text={descTooltip} onClose={() => setDescTooltip(null)} />
      )}
    </div>
  );
}

function navBtnStyle(disabled: boolean, primary: boolean, flex: number) {
  return {
    flex,
    minHeight: '44px',
    background: disabled ? '#232A2E' : primary ? '#A7C080' : '#343F44',
    border: `1px solid ${disabled ? '#343F44' : primary ? '#A7C080' : '#475258'}`,
    borderRadius: '10px',
    color: disabled ? '#475258' : primary ? '#2D353B' : '#9DA9A0',
    fontSize: '13px',
    fontWeight: 600 as const,
    cursor: disabled ? ('not-allowed' as const) : ('pointer' as const),
    WebkitTapHighlightColor: 'transparent',
    transition: 'all 0.1s',
  };
}
