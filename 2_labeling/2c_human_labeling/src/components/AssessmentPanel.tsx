import { useState, useRef, useCallback, useEffect } from 'react';
import type { Assessment, TrajectoryType, AlignmentType, ConversationEntry } from '../types';
import { TRAJECTORY_OPTIONS, ALIGNMENT_OPTIONS } from '../taxonomy';
import { getSentences } from '../store';

// ── Long-press hook ──────────────────────────────────────────────────────────

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

// ── Hover tooltip ────────────────────────────────────────────────────────────

function HoverTooltip({ text, rect }: { text: string; rect: DOMRect }) {
  const [pos, setPos] = useState({ top: 0, left: 0, opacity: 0 });
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    const tt = ref.current.getBoundingClientRect();
    
    let top = rect.bottom + 8;
    let left = rect.left;

    if (left + tt.width > window.innerWidth - 12) {
      left = window.innerWidth - tt.width - 12;
    }
    if (left < 12) left = 12;

    if (top + tt.height > window.innerHeight - 12) {
      top = rect.top - tt.height - 8;
    }

    setPos({ top, left, opacity: 1 });
  }, [rect, text]);

  return (
    <div
      ref={ref}
      style={{
        position: 'fixed',
        top: pos.top,
        left: pos.left,
        opacity: pos.opacity,
        pointerEvents: 'none',
        zIndex: 1000,
        background: '#343F44',
        border: '1px solid #475258',
        borderRadius: '8px',
        padding: '10px 14px',
        maxWidth: '320px',
        color: '#D3C6AA',
        fontSize: '12px',
        lineHeight: '1.5',
        boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
        whiteSpace: 'pre-wrap',
        transition: 'opacity 0.05s',
      }}
    >
      {text}
    </div>
  );
}

// ── Option pill button ────────────────────────────────────────────────────────

function OptionButton<T extends string>({
  option,
  selected,
  onSelect,
  onHover,
}: {
  option: { value: T; label: string; description: string };
  selected: boolean;
  onSelect: (v: T) => void;
  onHover: (text: string | null, rect?: DOMRect) => void;
}) {
  const desc = `${option.label}\n\n${option.description}`;
  const longPress = useLongPress(() => {
    // We can also trigger a formal modal if needed, but for now hover is fine
  });

  return (
    <button
      onClick={() => onSelect(option.value)}
      onMouseEnter={(e) => onHover(desc, e.currentTarget.getBoundingClientRect())}
      {...longPress}
      onMouseLeave={() => { onHover(null); longPress.onMouseLeave(); }}
      style={{
        minHeight: '44px',
        padding: '8px 16px',
        background: selected ? '#A7C080' : '#343F44',
        border: `1.5px solid ${selected ? '#A7C080' : '#475258'}`,
        borderRadius: '10px',
        color: selected ? '#2D353B' : '#9DA9A0',
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
          border: `2px solid ${selected ? '#2D353B' : '#7A8478'}`,
          background: selected ? '#2D353B' : 'transparent',
          flexShrink: 0,
        }}
      />
      {option.label}
    </button>
  );
}

// ── AssessmentPanel ───────────────────────────────────────────────────────────

interface Props {
  entry: ConversationEntry;
  assessment: Assessment;
  step: number;
  onUpdate: (a: Partial<Assessment>) => void;
  onNext: () => void;
  onBack: () => void;
  onSubmit: () => void;
}

export default function AssessmentPanel({
  entry,
  assessment,
  step,
  onUpdate,
  onNext,
  onBack,
  onSubmit,
}: Props) {
  const [hoverTooltip, setHoverTooltip] = useState<{ text: string; rect: DOMRect } | null>(null);
  const sentences = getSentences(entry);
  const selectedSentence = sentences.find(s => s.index === assessment.turning_point);

  const canAdvance = true; // For now all steps have defaults

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Header */}
      <div style={{ padding: '8px 12px', borderBottom: '1px solid #3D484D', background: '#343F44' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '10px', fontWeight: 700, color: '#A7C080', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            Final Assessment · Step {step + 1}/3
          </span>
        </div>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {step === 0 && (
          <>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#D3C6AA' }}>Reasoning Trajectory</div>
            <div style={{ fontSize: '11px', color: '#7A8478', lineHeight: '1.4' }}>
              How does the model deliberate before producing its final output?
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {TRAJECTORY_OPTIONS.map((opt) => (
                <OptionButton
                  key={opt.value}
                  option={opt}
                  selected={assessment.trajectory === opt.value}
                  onSelect={(v) => onUpdate({ trajectory: v })}
                  onHover={(text, rect) => setHoverTooltip(text && rect ? { text, rect } : null)}
                />
              ))}
            </div>
          </>
        )}

        {step === 1 && (
          <>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#D3C6AA' }}>Output Alignment</div>
            <div style={{ fontSize: '11px', color: '#7A8478', lineHeight: '1.4' }}>
              Is the reasoning aligned with the final response?
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {ALIGNMENT_OPTIONS.map((opt) => (
                <OptionButton
                  key={opt.value}
                  option={opt}
                  selected={assessment.alignment === opt.value}
                  onSelect={(v) => onUpdate({ alignment: v })}
                  onHover={(text, rect) => setHoverTooltip(text && rect ? { text, rect } : null)}
                />
              ))}
            </div>
          </>
        )}

        {step === 2 && (
          <>
            <div style={{ fontSize: '13px', fontWeight: 700, color: '#D3C6AA' }}>Turning Point</div>
            <div style={{ fontSize: '11px', color: '#7A8478', lineHeight: '1.4' }}>
              Which sentence is where the model commits to its final course of action?
              <strong> Click a sentence in the reasoning trace</strong> or use the button below.
            </div>

            <div style={{ marginTop: '4px' }}>
              <button
                onClick={() => onUpdate({ turning_point: -1 })}
                style={{
                  width: '100%',
                  minHeight: '44px',
                  padding: '8px 12px',
                  background: assessment.turning_point === -1 ? '#343F44' : '#232A2E',
                  border: `1.5px solid ${assessment.turning_point === -1 ? '#7A8478' : '#343F44'}`,
                  borderRadius: '10px',
                  color: assessment.turning_point === -1 ? '#D3C6AA' : '#7A8478',
                  fontSize: '13px',
                  fontWeight: assessment.turning_point === -1 ? 700 : 400,
                  cursor: 'pointer',
                  textAlign: 'left' as const,
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
                    border: `2px solid ${assessment.turning_point === -1 ? '#D3C6AA' : '#475258'}`,
                    background: assessment.turning_point === -1 ? '#D3C6AA' : 'transparent',
                    flexShrink: 0,
                  }}
                />
                No clear turning point (−1)
              </button>
            </div>

            {assessment.turning_point !== -1 && selectedSentence && (
              <div style={{
                marginTop: '8px',
                padding: '10px',
                background: 'rgba(127,187,179,0.07)',
                border: '1px solid rgba(127,187,179,0.3)',
                borderRadius: '10px',
              }}>
                <div style={{ fontSize: '10px', fontWeight: 700, color: '#7FBBB3', textTransform: 'uppercase', marginBottom: '6px' }}>
                  Selected Sentence #{selectedSentence.index}
                </div>
                <div style={{ fontSize: '12px', color: '#D3C6AA', lineHeight: '1.5', fontStyle: 'italic' }}>
                  "{selectedSentence.text}"
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Navigation */}
      <div style={{ padding: '8px 12px', borderTop: '1px solid #3D484D', display: 'flex', gap: '8px' }}>
        <button
          onClick={onBack}
          style={{
            flex: 1,
            minHeight: '44px',
            background: '#343F44',
            border: '1px solid #475258',
            borderRadius: '10px',
            color: '#9DA9A0',
            fontSize: '13px',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          {step === 0 ? '← Labels' : '← Back'}
        </button>
        {step < 2 ? (
          <button
            onClick={onNext}
            style={{
              flex: 2,
              minHeight: '44px',
              background: '#A7C080',
              border: '1px solid #A7C080',
              borderRadius: '10px',
              color: '#2D353B',
              fontSize: '13px',
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            Next Question →
          </button>
        ) : (
          <button
            onClick={onSubmit}
            style={{
              flex: 2,
              minHeight: '44px',
              background: '#7FBBB3',
              border: '1px solid #7FBBB3',
              borderRadius: '10px',
              color: '#2D353B',
              fontSize: '13px',
              fontWeight: 800,
              cursor: 'pointer',
            }}
          >
            Submit Labeling →
          </button>
        )}
      </div>

      {hoverTooltip && (
        <HoverTooltip text={hoverTooltip.text} rect={hoverTooltip.rect} />
      )}
    </div>
  );
}
