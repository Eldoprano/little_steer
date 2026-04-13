import { useRef, useState, useCallback } from 'react';
import type { ConversationEntry } from '../types';
import { filterLabeledEntries } from '../store';

interface Props {
  onLoad: (entries: ConversationEntry[]) => void;
}

export default function DataLoader({ onLoad }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const processFiles = useCallback(
    async (files: FileList | File[]) => {
      setLoading(true);
      setError(null);
      const fileArr = Array.from(files);
      const all: ConversationEntry[] = [];

      for (const file of fileArr) {
        if (!file.name.endsWith('.jsonl') && !file.name.endsWith('.json')) {
          setError(`"${file.name}" is not a .jsonl or .json file.`);
          setLoading(false);
          return;
        }
        try {
          const text = await file.text();
          const lines = text
            .split('\n')
            .map((l) => l.trim())
            .filter(Boolean);

          for (const line of lines) {
            try {
              const entry = JSON.parse(line) as ConversationEntry;
              if (entry.id && Array.isArray(entry.messages)) {
                all.push(entry);
              }
            } catch {
              // skip malformed lines
            }
          }
        } catch {
          setError(`Could not read "${file.name}".`);
          setLoading(false);
          return;
        }
      }

      const filtered = filterLabeledEntries(all);
      if (filtered.length === 0) {
        setError(
          `Loaded ${all.length} entries but none have LLM-generated sentence annotations yet. ` +
            'Run the automated labeler first, then import the output here.',
        );
        setLoading(false);
        return;
      }

      setLoading(false);
      onLoad(filtered);
    },
    [onLoad],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      processFiles(e.dataTransfer.files);
    },
    [processFiles],
  );

  return (
    <div
      style={{
        minHeight: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0f172a',
        padding: '24px',
        gap: '24px',
      }}
    >
      {/* Title */}
      <div style={{ textAlign: 'center' }}>
        <h1 style={{ color: '#f1f5f9', fontSize: '24px', fontWeight: 700, margin: 0 }}>
          Reasoning Labeler
        </h1>
        <p style={{ color: '#94a3b8', fontSize: '14px', marginTop: '8px' }}>
          Human labeling interface for model reasoning traces
        </p>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          width: '100%',
          maxWidth: '480px',
          minHeight: '180px',
          border: `2px dashed ${dragging ? '#6366f1' : '#334155'}`,
          borderRadius: '16px',
          background: dragging ? '#1e1b4b' : '#1e293b',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '12px',
          cursor: 'pointer',
          transition: 'all 0.15s ease',
          padding: '24px',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".jsonl,.json"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => e.target.files && processFiles(e.target.files)}
        />
        <svg
          width="48"
          height="48"
          viewBox="0 0 24 24"
          fill="none"
          stroke={dragging ? '#818cf8' : '#475569'}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#cbd5e1', fontSize: '15px', fontWeight: 500 }}>
            {loading ? 'Loading…' : 'Drop .jsonl files here'}
          </div>
          <div style={{ color: '#64748b', fontSize: '13px', marginTop: '4px' }}>
            or tap to browse
          </div>
        </div>
      </div>

      {error && (
        <div
          style={{
            maxWidth: '480px',
            background: '#3b1212',
            border: '1px solid #ef4444',
            borderRadius: '10px',
            padding: '12px 16px',
            color: '#fca5a5',
            fontSize: '13px',
            lineHeight: '1.5',
          }}
        >
          {error}
        </div>
      )}

      {/* Info */}
      <div
        style={{
          maxWidth: '480px',
          background: '#1e293b',
          borderRadius: '12px',
          padding: '16px',
          fontSize: '13px',
          color: '#64748b',
          lineHeight: '1.6',
        }}
      >
        <div style={{ color: '#94a3b8', fontWeight: 600, marginBottom: '8px' }}>
          Expected format
        </div>
        <div>JSONL files from the automated sentence labeler — each line is a</div>
        <div>
          <code
            style={{
              background: '#0f172a',
              borderRadius: '4px',
              padding: '1px 5px',
              color: '#7dd3fc',
            }}
          >
            ConversationEntry
          </code>{' '}
          with <code style={{ background: '#0f172a', borderRadius: '4px', padding: '1px 5px', color: '#7dd3fc' }}>annotations</code> already populated by the LLM judge.
        </div>
        <div style={{ marginTop: '8px' }}>
          Only entries with LLM-generated sentence annotations are shown.
        </div>
      </div>
    </div>
  );
}
