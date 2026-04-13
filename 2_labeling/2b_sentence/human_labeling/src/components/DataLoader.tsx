import { useRef, useState, useCallback, useEffect } from 'react';
import type { ConversationEntry } from '../types';
import { filterLabeledEntries } from '../store';

interface Props {
  onLoad: (entries: ConversationEntry[]) => void;
}

/** Parse JSONL text into ConversationEntry objects. */
function parseJsonl(text: string): ConversationEntry[] {
  const entries: ConversationEntry[] = [];
  for (const line of text.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    try {
      const entry = JSON.parse(trimmed) as ConversationEntry;
      if (entry.id && Array.isArray(entry.messages)) entries.push(entry);
    } catch { /* skip malformed */ }
  }
  return entries;
}

interface AvailableFile {
  name: string;
  selected: boolean;
}

export default function DataLoader({ onLoad }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  // Auto-discovered files from the server
  const [availableFiles, setAvailableFiles] = useState<AvailableFile[] | null>(null);
  const [loadingFiles, setLoadingFiles] = useState(false);

  // Fetch available files on mount
  useEffect(() => {
    setLoadingFiles(true);
    fetch('/api/files')
      .then((r) => r.json())
      .then((files: string[]) => {
        if (Array.isArray(files) && files.length > 0) {
          setAvailableFiles(files.map((name) => ({ name, selected: false })));
        }
      })
      .catch(() => { /* server API not available — fall back to manual upload */ })
      .finally(() => setLoadingFiles(false));
  }, []);

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
          all.push(...parseJsonl(text));
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

  const handleLoadSelected = useCallback(async () => {
    if (!availableFiles) return;
    const selected = availableFiles.filter((f) => f.selected);
    if (selected.length === 0) {
      setError('Select at least one file to load.');
      return;
    }

    setLoading(true);
    setError(null);
    const all: ConversationEntry[] = [];

    for (const file of selected) {
      try {
        const resp = await fetch(`/api/data/${encodeURIComponent(file.name)}`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const text = await resp.text();
        all.push(...parseJsonl(text));
      } catch {
        setError(`Could not load "${file.name}" from server.`);
        setLoading(false);
        return;
      }
    }

    const filtered = filterLabeledEntries(all);
    if (filtered.length === 0) {
      setError(
        `Loaded ${all.length} entries but none have sentence annotations. ` +
          'Run the automated labeler first.',
      );
      setLoading(false);
      return;
    }

    setLoading(false);
    onLoad(filtered);
  }, [availableFiles, onLoad]);

  const toggleFile = (idx: number) => {
    setAvailableFiles((prev) =>
      prev?.map((f, i) => (i === idx ? { ...f, selected: !f.selected } : f)) ?? null,
    );
  };

  const toggleAll = () => {
    setAvailableFiles((prev) => {
      if (!prev) return null;
      const allSelected = prev.every((f) => f.selected);
      return prev.map((f) => ({ ...f, selected: !allSelected }));
    });
  };

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      processFiles(e.dataTransfer.files);
    },
    [processFiles],
  );

  const selectedCount = availableFiles?.filter((f) => f.selected).length ?? 0;

  return (
    <div
      style={{
        minHeight: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#2D353B',
        padding: '24px',
        gap: '20px',
      }}
    >
      {/* Title */}
      <div style={{ textAlign: 'center' }}>
        <h1 style={{ color: '#D3C6AA', fontSize: '24px', fontWeight: 700, margin: 0 }}>
          Reasoning Labeler
        </h1>
        <p style={{ color: '#859289', fontSize: '14px', marginTop: '8px' }}>
          Human labeling interface for model reasoning traces
        </p>
      </div>

      {/* Auto-discovered files from data/2b_labeled/ */}
      {availableFiles && availableFiles.length > 0 && (
        <div
          style={{
            width: '100%',
            maxWidth: '520px',
            background: '#343F44',
            borderRadius: '14px',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ color: '#A7C080', fontSize: '12px', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Available labeled files
            </div>
            <button
              onClick={toggleAll}
              style={{
                background: 'transparent',
                border: 'none',
                color: '#7FBBB3',
                fontSize: '11px',
                cursor: 'pointer',
                padding: '2px 6px',
              }}
            >
              {availableFiles.every((f) => f.selected) ? 'Deselect all' : 'Select all'}
            </button>
          </div>

          <div
            style={{
              maxHeight: '200px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '4px',
            }}
          >
            {availableFiles.map((file, i) => (
              <button
                key={file.name}
                onClick={() => toggleFile(i)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '7px 10px',
                  background: file.selected ? '#3D484D' : '#2D353B',
                  border: `1px solid ${file.selected ? '#A7C080' : '#475258'}`,
                  borderRadius: '8px',
                  color: file.selected ? '#D3C6AA' : '#859289',
                  fontSize: '12px',
                  cursor: 'pointer',
                  textAlign: 'left' as const,
                  transition: 'all 0.1s',
                }}
              >
                <span
                  style={{
                    width: '16px',
                    height: '16px',
                    borderRadius: '4px',
                    border: `2px solid ${file.selected ? '#A7C080' : '#7A8478'}`,
                    background: file.selected ? '#A7C080' : 'transparent',
                    flexShrink: 0,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '10px',
                    color: '#2D353B',
                  }}
                >
                  {file.selected ? '✓' : ''}
                </span>
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {file.name}
                </span>
              </button>
            ))}
          </div>

          <button
            onClick={handleLoadSelected}
            disabled={selectedCount === 0 || loading}
            style={{
              minHeight: '44px',
              background: selectedCount > 0 ? '#A7C080' : '#3D484D',
              border: 'none',
              borderRadius: '10px',
              color: selectedCount > 0 ? '#2D353B' : '#7A8478',
              fontSize: '14px',
              fontWeight: 700,
              cursor: selectedCount > 0 ? 'pointer' : 'not-allowed',
              transition: 'all 0.15s',
            }}
          >
            {loading ? 'Loading...' : `Load ${selectedCount} file${selectedCount !== 1 ? 's' : ''}`}
          </button>
        </div>
      )}

      {loadingFiles && (
        <div style={{ color: '#859289', fontSize: '13px' }}>Scanning for labeled files...</div>
      )}

      {/* Divider when both options are available */}
      {availableFiles && availableFiles.length > 0 && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', width: '100%', maxWidth: '520px' }}>
          <div style={{ flex: 1, height: '1px', background: '#475258' }} />
          <span style={{ color: '#7A8478', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
            or upload manually
          </span>
          <div style={{ flex: 1, height: '1px', background: '#475258' }} />
        </div>
      )}

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          width: '100%',
          maxWidth: '520px',
          minHeight: availableFiles ? '100px' : '180px',
          border: `2px dashed ${dragging ? '#A7C080' : '#475258'}`,
          borderRadius: '16px',
          background: dragging ? '#343F44' : '#2D353B',
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
          width="36"
          height="36"
          viewBox="0 0 24 24"
          fill="none"
          stroke={dragging ? '#A7C080' : '#7A8478'}
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        <div style={{ textAlign: 'center' }}>
          <div style={{ color: '#D3C6AA', fontSize: '14px', fontWeight: 500 }}>
            {loading ? 'Loading...' : 'Drop .jsonl files here'}
          </div>
          <div style={{ color: '#7A8478', fontSize: '12px', marginTop: '4px' }}>
            or tap to browse
          </div>
        </div>
      </div>

      {error && (
        <div
          style={{
            maxWidth: '520px',
            background: '#3B2D2E',
            border: '1px solid #E67E80',
            borderRadius: '10px',
            padding: '12px 16px',
            color: '#E67E80',
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
          maxWidth: '520px',
          background: '#343F44',
          borderRadius: '12px',
          padding: '16px',
          fontSize: '13px',
          color: '#7A8478',
          lineHeight: '1.6',
        }}
      >
        <div style={{ color: '#859289', fontWeight: 600, marginBottom: '8px' }}>
          Expected format
        </div>
        <div>JSONL files from the automated sentence labeler — each line is a</div>
        <div>
          <code
            style={{
              background: '#2D353B',
              borderRadius: '4px',
              padding: '1px 5px',
              color: '#7FBBB3',
            }}
          >
            ConversationEntry
          </code>{' '}
          with <code style={{ background: '#2D353B', borderRadius: '4px', padding: '1px 5px', color: '#7FBBB3' }}>annotations</code> already populated by the LLM judge.
        </div>
        <div style={{ marginTop: '8px' }}>
          Only entries with LLM-generated sentence annotations are shown.
        </div>
      </div>
    </div>
  );
}
