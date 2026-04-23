import { useState, useEffect } from 'react';

interface Props {
  onSetHandle: (handle: string) => void;
}

export default function IdentityScreen({ onSetHandle }: Props) {
  const [input, setInput] = useState('');
  const [existingHumans, setExistingHumans] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/humans')
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setExistingHumans(data);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const handle = input.trim().toLowerCase().replace(/[^a-z0-9]/g, '');
    if (handle) {
      onSetHandle(handle);
    }
  };

  return (
    <div
      style={{
        height: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#2D353B',
        padding: '24px',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
    >
      <div style={{ width: '100%', maxWidth: '400px', textAlign: 'center' }}>
        <h1 style={{ color: '#D3C6AA', fontSize: '24px', fontWeight: 800, margin: '0 0 12px' }}>
          Welcome
        </h1>
        <p style={{ color: '#7A8478', fontSize: '14px', margin: '0 0 32px', lineHeight: 1.5 }}>
          Please enter a name or handle to begin labeling.
          Your labels will be tagged with <strong>human_</strong> prefix.
        </p>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <input
            autoFocus
            type="text"
            placeholder="e.g. john_doe"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            style={{
              minHeight: '52px',
              padding: '0 16px',
              background: '#343F44',
              border: '1px solid #475258',
              borderRadius: '12px',
              color: '#D3C6AA',
              fontSize: '16px',
              outline: 'none',
              textAlign: 'center',
            }}
          />
          <button
            disabled={!input.trim()}
            style={{
              minHeight: '52px',
              background: input.trim() ? '#A7C080' : '#343F44',
              border: 'none',
              borderRadius: '12px',
              color: input.trim() ? '#2D353B' : '#475258',
              fontSize: '16px',
              fontWeight: 700,
              cursor: input.trim() ? 'pointer' : 'not-allowed',
              transition: 'all 0.15s',
            }}
          >
            Start Labeling →
          </button>
        </form>

        {existingHumans.length > 0 && (
          <div style={{ marginTop: '48px', textAlign: 'left' }}>
            <div
              style={{
                color: '#A7C080',
                fontSize: '11px',
                fontWeight: 700,
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                marginBottom: '12px',
              }}
            >
              Registered Human Labelers
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {existingHumans.map((h) => {
                const handleValue = h.replace(/^human_/, '');
                return (
                  <button
                    key={h}
                    type="button"
                    onClick={() => setInput(handleValue)}
                    style={{
                      padding: '4px 10px',
                      background: '#343F44',
                      border: '1px solid #475258',
                      borderRadius: '6px',
                      color: '#9DA9A0',
                      fontSize: '12px',
                      cursor: 'pointer',
                      transition: 'all 0.15s'
                    }}
                    onMouseOver={(e) => {
                      e.currentTarget.style.background = '#A7C080';
                      e.currentTarget.style.color = '#2D353B';
                    }}
                    onMouseOut={(e) => {
                      e.currentTarget.style.background = '#343F44';
                      e.currentTarget.style.color = '#9DA9A0';
                    }}
                  >
                    {h}
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {loading && (
          <div style={{ marginTop: '48px', color: '#7A8478', fontSize: '12px' }}>
            Loading existing labelers...
          </div>
        )}
      </div>
    </div>
  );
}
