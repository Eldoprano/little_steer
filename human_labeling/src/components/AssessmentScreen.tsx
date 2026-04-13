import { useState, useRef, useCallback } from 'react';
import type { Assessment, TrajectoryType, AlignmentType, ConversationEntry } from '../types';
import { TRAJECTORY_OPTIONS, ALIGNMENT_OPTIONS } from '../taxonomy';
import { getSentences } from '../store';

// ── Long-press hook (same pattern as Labeler) ────────────────────────────────

function useLongPress(onLongPress: () => void, ms = 650) {
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const start = useCallback(
    (e?: React.TouchEvent | React.MouseEvent) => {
      if (e && 'touches' in e) e.preventDefault();
      timer.current = setTimeout(onLongPress, ms);
    },
    [onLongPress, ms],
  );

  const cancel = useCallback(() => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null; }
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

// ── Description tooltip ────────────────────────────────────────────────────

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
        background: 'rgba(0,0,0,0.75)',
        padding: '24px',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: '#1e293b',
          border: '1px solid #334155',
          borderRadius: '16px',
          padding: '20px 22px',
          maxWidth: '440px',
          color: '#e2e8f0',
          fontSize: '13px',
          lineHeight: '1.65',
          boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
          whiteSpace: 'pre-wrap',
        }}
      >
        <p style={{ margin: '0 0 16px' }}>{text}</p>
        <button
          onClick={onClose}
          style={{
            minHeight: '44px',
            padding: '0 20px',
            background: '#334155',
            border: 'none',
            borderRadius: '8px',
            color: '#f1f5f9',
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

// ── Option pill button with long-press ────────────────────────────────────────

function OptionButton<T extends string>({
  option,
  selected,
  onSelect,
  onDescRequest,
}: {
  option: { value: T; label: string; description: string };
  selected: boolean;
  onSelect: (v: T) => void;
  onDescRequest: (text: string) => void;
}) {
  const longPress = useLongPress(() => {
    onDescRequest(`${option.label}\n\n${option.description}`);
  });

  return (
    <button
      onClick={() => onSelect(option.value)}
      {...longPress}
      style={{
        minHeight: '44px',
        padding: '8px 16px',
        background: selected ? '#4f46e5' : '#1e293b',
        border: `1.5px solid ${selected ? '#6366f1' : '#334155'}`,
        borderRadius: '10px',
        color: selected ? '#fff' : '#94a3b8',
        fontSize: '13px',
        fontWeight: selected ? 700 : 400,
        cursor: 'pointer',
        textAlign: 'left' as const,
        transition: 'all 0.1s',
        WebkitTapHighlightColor: 'transparent',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
      }}
    >
      <span
        style={{
          width: '16px',
          height: '16px',
          borderRadius: '50%',
          border: `2px solid ${selected ? '#a5b4fc' : '#475569'}`,
          background: selected ? '#a5b4fc' : 'transparent',
          flexShrink: 0,
        }}
      />
      {option.label}
    </button>
  );
}

// ── Sentence picker for turning point ─────────────────────────────────────────

function TurningPointPicker({
  entry,
  value,
  onChange,
}: {
  entry: ConversationEntry;
  value: number;
  onChange: (v: number) => void;
}) {
  const sentences = getSentences(entry);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
      {/* "No turning point" option */}
      <button
        onClick={() => onChange(-1)}
        style={{
          minHeight: '44px',
          padding: '8px 12px',
          background: value === -1 ? '#1e293b' : '#0f172a',
          border: `1.5px solid ${value === -1 ? '#64748b' : '#1e293b'}`,
          borderRadius: '8px',
          color: value === -1 ? '#f1f5f9' : '#475569',
          fontSize: '12px',
          fontWeight: value === -1 ? 600 : 400,
          cursor: 'pointer',
          textAlign: 'left' as const,
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          WebkitTapHighlightColor: 'transparent',
        }}
      >
        <span style={{ fontSize: '10px', opacity: 0.7 }}>—</span>
        No clear turning point (−1)
      </button>

      {sentences.map((s) => (
        <button
          key={s.index}
          onClick={() => onChange(s.index)}
          style={{
            minHeight: '44px',
            padding: '7px 12px',
            background: value === s.index ? '#1e3a5f' : '#0f172a',
            border: `1.5px solid ${value === s.index ? '#3b82f6' : '#1e293b'}`,
            borderRadius: '8px',
            color: value === s.index ? '#93c5fd' : '#475569',
            fontSize: '12px',
            cursor: 'pointer',
            textAlign: 'left' as const,
            display: 'flex',
            alignItems: 'flex-start',
            gap: '8px',
            WebkitTapHighlightColor: 'transparent',
            transition: 'all 0.1s',
          }}
        >
          <span
            style={{
              background: value === s.index ? '#3b82f6' : '#1e293b',
              color: value === s.index ? '#fff' : '#64748b',
              borderRadius: '4px',
              padding: '1px 5px',
              fontSize: '10px',
              fontWeight: 700,
              whiteSpace: 'nowrap',
              flexShrink: 0,
              marginTop: '1px',
            }}
          >
            #{s.index}
          </span>
          <span style={{ lineHeight: '1.4', fontSize: '11px', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' as const }}>
            {s.text}
          </span>
        </button>
      ))}
    </div>
  );
}

// ── AssessmentScreen ──────────────────────────────────────────────────────────

interface Props {
  entry: ConversationEntry;
  existing?: Assessment;
  onSubmit: (assessment: Assessment) => void;
  onBack: () => void;
}

export default function AssessmentScreen({ entry, existing, onSubmit, onBack }: Props) {
  const [trajectory, setTrajectory] = useState<TrajectoryType>(
    existing?.trajectory ?? 'mixed_inconclusive',
  );
  const [turningPoint, setTurningPoint] = useState<number>(existing?.turning_point ?? -1);
  const [alignment, setAlignment] = useState<AlignmentType>(existing?.alignment ?? 'aligned');
  const [descTooltip, setDescTooltip] = useState<string | null>(null);

  const canSubmit = trajectory !== null && alignment !== null;

  const handleSubmit = () => {
    if (!canSubmit) return;
    onSubmit({ trajectory, turning_point: turningPoint, alignment });
  };

  return (
    <div
      style={{
        height: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        background: '#0f172a',
        overflow: 'hidden',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '8px 14px',
          borderBottom: '1px solid #1e293b',
          flexShrink: 0,
          minHeight: '48px',
          gap: '12px',
        }}
      >
        <button
          onClick={onBack}
          style={{
            minHeight: '44px',
            minWidth: '44px',
            background: 'transparent',
            border: 'none',
            color: '#64748b',
            fontSize: '20px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            WebkitTapHighlightColor: 'transparent',
          }}
        >
          ←
        </button>
        <div>
          <div style={{ color: '#f1f5f9', fontSize: '14px', fontWeight: 700 }}>
            Overall Assessment
          </div>
          <div style={{ color: '#64748b', fontSize: '11px', marginTop: '1px' }}>
            All sentences labeled · complete the assessment to submit
          </div>
        </div>
      </div>

      {/* Two-column body */}
      <div
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          minHeight: 0,
          overflow: 'hidden',
        }}
      >
        {/* LEFT: trajectory + alignment */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            borderRight: '1px solid #1e293b',
            overflowY: 'auto',
            padding: '12px',
            gap: '10px',
          }}
        >
          {/* Trajectory */}
          <div>
            <div style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '8px' }}>
              Reasoning Trajectory
              <span style={{ fontSize: '10px', color: '#475569', marginLeft: '6px', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
                (long-press for description)
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
              {TRAJECTORY_OPTIONS.map((opt) => (
                <OptionButton
                  key={opt.value}
                  option={opt}
                  selected={trajectory === opt.value}
                  onSelect={setTrajectory}
                  onDescRequest={(text) => setDescTooltip(text)}
                />
              ))}
            </div>
          </div>

          {/* Alignment */}
          <div>
            <div style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '8px', marginTop: '4px' }}>
              Output Alignment
              <span style={{ fontSize: '10px', color: '#475569', marginLeft: '6px', fontWeight: 400, textTransform: 'none', letterSpacing: 0 }}>
                (long-press for description)
              </span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
              {ALIGNMENT_OPTIONS.map((opt) => (
                <OptionButton
                  key={opt.value}
                  option={opt}
                  selected={alignment === opt.value}
                  onSelect={setAlignment}
                  onDescRequest={(text) => setDescTooltip(text)}
                />
              ))}
            </div>
          </div>
        </div>

        {/* RIGHT: turning point */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
            minHeight: 0,
          }}
        >
          <div
            style={{
              padding: '12px 12px 6px',
              borderBottom: '1px solid #1e293b',
              flexShrink: 0,
            }}
          >
            <div style={{ fontSize: '12px', fontWeight: 700, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Turning Point
            </div>
            <div style={{ fontSize: '11px', color: '#475569', marginTop: '4px', lineHeight: '1.4' }}>
              Which sentence is where the model commits to its final course of action?
              Select <strong style={{ color: '#64748b' }}>−1</strong> if there is no clear turning point.
            </div>
          </div>
          <div
            style={{
              flex: 1,
              overflowY: 'auto',
              padding: '8px 12px',
              minHeight: 0,
            }}
          >
            <TurningPointPicker
              entry={entry}
              value={turningPoint}
              onChange={setTurningPoint}
            />
          </div>
        </div>
      </div>

      {/* Submit bar */}
      <div
        style={{
          flexShrink: 0,
          padding: '10px 14px',
          borderTop: '1px solid #1e293b',
          display: 'flex',
          gap: '8px',
        }}
      >
        <button
          onClick={onBack}
          style={{
            flex: 1,
            minHeight: '48px',
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '12px',
            color: '#94a3b8',
            fontSize: '14px',
            cursor: 'pointer',
            WebkitTapHighlightColor: 'transparent',
          }}
        >
          ← Back
        </button>
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          style={{
            flex: 3,
            minHeight: '48px',
            background: canSubmit ? '#4f46e5' : '#1a1f2e',
            border: `1px solid ${canSubmit ? '#6366f1' : '#334155'}`,
            borderRadius: '12px',
            color: canSubmit ? '#fff' : '#334155',
            fontSize: '14px',
            fontWeight: 700,
            cursor: canSubmit ? 'pointer' : 'not-allowed',
            WebkitTapHighlightColor: 'transparent',
          }}
        >
          Submit & Next Response →
        </button>
      </div>

      {descTooltip && (
        <DescTooltip text={descTooltip} onClose={() => setDescTooltip(null)} />
      )}
    </div>
  );
}
