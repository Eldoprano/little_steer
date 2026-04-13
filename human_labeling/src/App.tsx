import { useState, useCallback, useEffect } from 'react';
import type { ConversationEntry, Assessment, PersistedState } from './types';
import {
  saveEntries,
  loadEntries,
  clearEntries,
  saveProgress,
  loadProgress,
  defaultProgress,
  clearProgress,
  getSentences,
  filterLabeledEntries,
} from './store';
import DataLoader from './components/DataLoader';
import Labeler from './components/Labeler';
import AssessmentScreen from './components/AssessmentScreen';
import StatsView from './components/StatsView';

type Screen = 'labeling' | 'assessment' | 'stats';

export default function App() {
  const [entries, setEntries] = useState<ConversationEntry[]>(() => loadEntries() ?? []);
  const [appState, setAppState] = useState<PersistedState>(() => loadProgress());
  const [screen, setScreen] = useState<Screen>('labeling');

  // Persist on every state change
  useEffect(() => {
    saveProgress(appState);
  }, [appState]);

  // ── Derived ──────────────────────────────────────────────────────────────

  const currentEntry = entries[appState.currentEntryIndex] ?? null;
  const currentProgress = currentEntry
    ? (appState.progress[currentEntry.id] ?? { sentenceLabels: {}, completed: false })
    : null;
  const sentences = currentEntry ? getSentences(currentEntry) : [];

  const completedCount = entries.filter((e) => appState.progress[e.id]?.completed).length;
  const allEntriesDone = entries.length > 0 && completedCount === entries.length;

  // Are all sentences in the current entry labeled (but assessment not yet submitted)?
  const allSentencesLabeled =
    sentences.length > 0 &&
    sentences.every((s) => (currentProgress?.sentenceLabels[s.index]?.length ?? 0) > 0) &&
    !currentProgress?.completed;

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleLoad = useCallback((newEntries: ConversationEntry[]) => {
    setEntries((prev) => {
      const existingIds = new Set(prev.map((e) => e.id));
      const toAdd = newEntries.filter((e) => !existingIds.has(e.id));
      const merged = [...prev, ...toAdd];
      saveEntries(merged);
      return merged;
    });
  }, []);

  const handleLabelSentence = useCallback(
    (labels: string[]) => {
      if (!currentEntry) return;
      const sentIdx = appState.currentSentenceIndex;
      setAppState((prev) => {
        const ep = prev.progress[currentEntry.id] ?? { sentenceLabels: {}, completed: false };
        return {
          ...prev,
          progress: {
            ...prev.progress,
            [currentEntry.id]: {
              ...ep,
              sentenceLabels: { ...ep.sentenceLabels, [sentIdx]: labels },
            },
          },
        };
      });
    },
    [currentEntry, appState.currentSentenceIndex],
  );

  const handleNavigate = useCallback(
    (direction: 'back' | 'next') => {
      setAppState((prev) => {
        const idx = prev.currentSentenceIndex;
        const nextIdx =
          direction === 'back'
            ? Math.max(0, idx - 1)
            : Math.min(sentences.length - 1, idx + 1);
        return { ...prev, currentSentenceIndex: nextIdx };
      });
    },
    [sentences.length],
  );

  const handleJump = useCallback((idx: number) => {
    setAppState((prev) => ({ ...prev, currentSentenceIndex: idx }));
  }, []);

  const handleSubmitAssessment = useCallback(
    (assessment: Assessment) => {
      if (!currentEntry) return;
      setAppState((prev) => {
        const ep = prev.progress[currentEntry.id] ?? { sentenceLabels: {}, completed: false };

        // Find the next unlabeled entry
        let nextIdx = prev.currentEntryIndex + 1;
        while (
          nextIdx < entries.length &&
          prev.progress[entries[nextIdx]?.id]?.completed
        ) {
          nextIdx++;
        }
        if (nextIdx >= entries.length) nextIdx = prev.currentEntryIndex;

        return {
          ...prev,
          currentEntryIndex: nextIdx,
          currentSentenceIndex: 0,
          progress: {
            ...prev.progress,
            [currentEntry.id]: { ...ep, assessment, completed: true },
          },
        };
      });
      setScreen('labeling');
    },
    [currentEntry, entries],
  );

  const handleReset = useCallback(() => {
    clearProgress();
    setAppState(defaultProgress());
    setScreen('labeling');
  }, []);

  const handleClearAll = useCallback(() => {
    clearEntries();
    clearProgress();
    setEntries([]);
    setAppState(defaultProgress());
    setScreen('labeling');
  }, []);

  // ── Routing ───────────────────────────────────────────────────────────────

  // No data loaded
  if (entries.length === 0) {
    return <DataLoader onLoad={handleLoad} />;
  }

  // Stats overlay renders on top of the labeler
  if (screen === 'stats') {
    return (
      <>
        {currentEntry ? (
          <Labeler
            entry={currentEntry}
            entryIndex={appState.currentEntryIndex}
            totalEntries={entries.length}
            completedCount={completedCount}
            sentenceIndex={appState.currentSentenceIndex}
            sentenceLabels={currentProgress?.sentenceLabels ?? {}}
            onLabelSentence={handleLabelSentence}
            onNavigate={handleNavigate}
            onJumpToSentence={handleJump}
            onShowAssessment={() => setScreen('assessment')}
            onShowStats={() => {}}
            allDone={allSentencesLabeled}
          />
        ) : (
          <AllDoneScreen
            completedCount={completedCount}
            total={entries.length}
            onShowStats={() => {}}
            onLoadMore={handleLoad}
          />
        )}
        <StatsView
          entries={entries}
          state={appState}
          onClose={() => setScreen('labeling')}
          onReset={handleReset}
          onClearData={handleClearAll}
        />
      </>
    );
  }

  // Assessment screen
  if (screen === 'assessment' && currentEntry) {
    return (
      <AssessmentScreen
        entry={currentEntry}
        existing={currentProgress?.assessment}
        onSubmit={handleSubmitAssessment}
        onBack={() => setScreen('labeling')}
      />
    );
  }

  // All done
  if (allEntriesDone) {
    return (
      <>
        <AllDoneScreen
          completedCount={completedCount}
          total={entries.length}
          onShowStats={() => setScreen('stats')}
          onLoadMore={handleLoad}
        />
        {(screen as string) === 'stats' && (
          <StatsView
            entries={entries}
            state={appState}
            onClose={() => setScreen('labeling')}
            onReset={handleReset}
            onClearData={handleClearAll}
          />
        )}
      </>
    );
  }

  // Labeling screen (guard: ensure currentEntry exists)
  if (!currentEntry) {
    // Shouldn't happen, but find first incomplete entry
    const firstIncomplete = entries.findIndex((e) => !appState.progress[e.id]?.completed);
    if (firstIncomplete !== -1) {
      setAppState((prev) => ({ ...prev, currentEntryIndex: firstIncomplete, currentSentenceIndex: 0 }));
    }
    return null;
  }

  return (
    <Labeler
      entry={currentEntry}
      entryIndex={appState.currentEntryIndex}
      totalEntries={entries.length}
      completedCount={completedCount}
      sentenceIndex={appState.currentSentenceIndex}
      sentenceLabels={currentProgress?.sentenceLabels ?? {}}
      onLabelSentence={handleLabelSentence}
      onNavigate={handleNavigate}
      onJumpToSentence={handleJump}
      onShowAssessment={() => setScreen('assessment')}
      onShowStats={() => setScreen('stats')}
      allDone={allSentencesLabeled}
    />
  );
}

// ── All done screen ───────────────────────────────────────────────────────────

function AllDoneScreen({
  completedCount,
  total,
  onShowStats,
  onLoadMore,
}: {
  completedCount: number;
  total: number;
  onShowStats: () => void;
  onLoadMore: (entries: ConversationEntry[]) => void;
}) {
  return (
    <div
      style={{
        height: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0f172a',
        gap: '24px',
        padding: '32px',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
    >
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontSize: '52px', marginBottom: '12px' }}>🎉</div>
        <h1 style={{ color: '#f1f5f9', fontSize: '22px', fontWeight: 800, margin: 0 }}>
          All {total} responses labeled!
        </h1>
        <p style={{ color: '#64748b', fontSize: '14px', marginTop: '8px' }}>
          Great work. Export your labels or load more files to continue.
        </p>
      </div>

      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '10px',
          width: '100%',
          maxWidth: '320px',
        }}
      >
        <button
          onClick={onShowStats}
          style={{
            minHeight: '52px',
            background: '#4f46e5',
            border: 'none',
            borderRadius: '12px',
            color: '#fff',
            fontSize: '15px',
            fontWeight: 700,
            cursor: 'pointer',
            WebkitTapHighlightColor: 'transparent',
          }}
        >
          View Stats &amp; Export Labels
        </button>

        <label
          style={{
            minHeight: '52px',
            background: '#1e293b',
            border: '1px solid #334155',
            borderRadius: '12px',
            color: '#94a3b8',
            fontSize: '14px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            WebkitTapHighlightColor: 'transparent',
          }}
        >
          <input
            type="file"
            accept=".jsonl,.json"
            multiple
            style={{ display: 'none' }}
            onChange={async (e) => {
              if (!e.target.files) return;
              const files = Array.from(e.target.files);
              const all: ConversationEntry[] = [];
              for (const file of files) {
                try {
                  const text = await file.text();
                  text
                    .split('\n')
                    .map((l) => l.trim())
                    .filter(Boolean)
                    .forEach((line) => {
                      try { all.push(JSON.parse(line)); } catch { /* skip */ }
                    });
                } catch { /* skip */ }
              }
              const filtered = filterLabeledEntries(all);
              if (filtered.length > 0) onLoadMore(filtered);
            }}
          />
          Load More Files
        </label>
      </div>
    </div>
  );
}
