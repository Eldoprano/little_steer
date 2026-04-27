import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
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
  buildHumanLabeledEntry,
  reconstructProgress,
  computeLlmHints,
} from './store';
import DataLoader from './components/DataLoader';
import Labeler from './components/Labeler';
import StatsView from './components/StatsView';
import IdentityScreen from './components/IdentityScreen';
import GamificationModal from './components/GamificationModal';
import {
  LABEL_DISPLAY_NAMES,
  getLabelGroup,
  LABEL_WEIGHTS,
} from './taxonomy';

type Screen = 'labeling' | 'stats' | 'gamification';

export default function App() {
  const [entries, setEntries] = useState<ConversationEntry[]>(() => loadEntries() ?? []);
  const [appState, setAppState] = useState<PersistedState>(() => loadProgress());
  const [screen, setScreen] = useState<Screen>('labeling');
  const [isLoading, setIsLoading] = useState(false);
  const [loadStatus, setLoadStatus] = useState<string>('');
  const [loadError, setLoadError] = useState<string | null>(null);
  const [totalExpectedCount, setTotalExpectedCount] = useState<number>(0);
  const hasAutoLoaded = useRef(false);

  // Persist on every state change
  useEffect(() => {
    saveProgress(appState);
  }, [appState]);

  // ── Derived ──────────────────────────────────────────────────────────────

  const currentEntry = entries[appState.currentEntryIndex] ?? null;
  const currentProgress = currentEntry
    ? (appState.progress[currentEntry.id] ?? { 
        sentenceLabels: {}, 
        sentenceScores: {}, 
        sentenceConfidences: {},
        completed: false 
      })
    : null;
  const sentences = useMemo(
    () => (currentEntry ? getSentences(currentEntry) : []),
    [currentEntry],
  );

  const llmHints = useMemo(
    () => (currentEntry && sentences.length > 0 ? computeLlmHints(currentEntry, sentences) : null),
    [currentEntry, sentences],
  );

  const completedCount = entries.filter((e) => appState.progress[e.id]?.completed).length;
  // User preference: show X/N as "working on the X-th entry"
  const activeProgressDisplay = completedCount + (currentProgress?.completed ? 0 : 1);

  const allSentencesLabeled =
    sentences.length > 0 &&
    sentences.every(
      (s) =>
        currentProgress?.sentenceLabels[s.index] !== undefined &&
        currentProgress?.sentenceScores[s.index] !== undefined,
    );

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleSetHandle = useCallback((handle: string) => {
    setAppState((prev) => ({ ...prev, handle }));
  }, []);

  const handleToggleLlmHints = useCallback(() => {
    setAppState((prev) => ({ ...prev, llmHintsEnabled: !prev.llmHintsEnabled }));
  }, []);

  const handleLoad = useCallback((newEntries: ConversationEntry[]) => {
    console.log(`[DataLoader] Received ${newEntries.length} entries`);
    setEntries((prev) => {
      const existingIds = new Set(prev.map((e) => e.id));
      const toAdd = newEntries.filter((e) => !existingIds.has(e.id));
      const merged = [...prev, ...toAdd];
      
      if (toAdd.length > 0) {
        console.log(`[DataLoader] Adding ${toAdd.length} new entries to existing ${prev.length}`);
        saveEntries(merged);
      }

      setAppState((prevAppState) => {
        if (!prevAppState.handle) return prevAppState;
        const newProgress = { ...prevAppState.progress };
        let changed = false;
        const judgeName = `human_${prevAppState.handle}`;
        
        for (const entry of merged) {
          if (!newProgress[entry.id]) {
            const reconstructed = reconstructProgress(entry, judgeName);
            if (reconstructed) {
              console.log(`[DataLoader] Reconstructed progress for entry ${entry.id}`);
              newProgress[entry.id] = reconstructed;
              changed = true;
            }
          }
        }
        
        // Auto-advance if we are at the very beginning and current is done
        let nextIdx = prevAppState.currentEntryIndex;
        if (merged.length > 0 && nextIdx < merged.length && newProgress[merged[nextIdx]?.id]?.completed) {
          while (nextIdx < merged.length && newProgress[merged[nextIdx]?.id]?.completed) {
            nextIdx++;
          }
        }

        if (!changed && prevAppState.currentEntryIndex === nextIdx) {
          return prevAppState;
        }

        console.log(`[DataLoader] Updating state. currentIdx: ${nextIdx}, changed: ${changed}`);
        return { 
          ...prevAppState, 
          progress: newProgress, 
          currentEntryIndex: nextIdx, 
          currentSentenceIndex: 0 
        };
      });

      return merged;
    });
  }, []);

  // Optimized auto-load: Fetch from meta and work order directly
  useEffect(() => {
    if (appState.handle && entries.length === 0 && !isLoading && !hasAutoLoaded.current) {
      hasAutoLoaded.current = true;
      setIsLoading(true);
      setLoadError(null);

      (async () => {
        try {
          console.log('[AutoLoad] Starting initial load...');
          // 1. Fetch metadata (small files)
          setLoadStatus('Fetching work order...');
          const [woResp, metaResp] = await Promise.all([
            fetch('/api/work-order'),
            fetch('/api/meta'),
          ]);

          if (!woResp.ok || !metaResp.ok) throw new Error('Failed to fetch initial metadata');

          const workOrder = await woResp.json();
          const meta = await metaResp.json();
          const woMap = new Map<string, number>();
          workOrder.flat_order.forEach((item: any, i: number) => woMap.set(item.id, i));

          // 2. Identify entries with reasoning that are in the work order
          const reasoningIds = new Set(
            meta.entries
              .filter((e: any) => e.has_reasoning && woMap.has(e.id))
              .map((e: any) => e.id)
          );

          if (reasoningIds.size === 0) throw new Error('No reasoning entries found in work order');
          
          const sortedIds = workOrder.flat_order
            .filter((item: any) => reasoningIds.has(item.id))
            .map((item: any) => item.id);

          setTotalExpectedCount(sortedIds.length);
          console.log(`[AutoLoad] Identified ${sortedIds.length} reasoning entries in work order`);

          // 3. Find first incomplete entry
          const judgeName = `human_${appState.handle}`;
          let firstTargetId = sortedIds[0];
          for (const id of sortedIds) {
            const entryMeta = meta.entries.find((e: any) => e.id === id);
            if (entryMeta && !entryMeta.judges.includes(judgeName)) {
              firstTargetId = id;
              break;
            }
          }

          // 4. Load the FIRST entry immediately to UNBLOCK the user
          console.log(`[AutoLoad] Loading first entry: ${firstTargetId}`);
          setLoadStatus('Loading first entry...');
          const firstEntryResp = await fetch(`/api/entry/${firstTargetId}`);
          if (!firstEntryResp.ok) throw new Error(`Failed to load entry ${firstTargetId}`);
          
          const firstEntry = await firstEntryResp.json();
          handleLoad([{ ...firstEntry, id: `dataset.jsonl:${firstEntry.id}` }]);

          // !!! UNBLOCK THE UI HERE !!!
          setIsLoading(false);

          // 5. Load the next 100 entries in parallel (much faster than fetching whole dataset.jsonl)
          const remainingIncompleteIds = sortedIds.filter(id => id !== firstTargetId).slice(0, 100);
          console.log(`[AutoLoad] Background sync: fetching ${remainingIncompleteIds.length} entries...`);
          setLoadStatus(`Loading queue (${remainingIncompleteIds.length} entries)...`);

          // Batch the requests to avoid overwhelming the server
          const BATCH_SIZE = 5;
          for (let i = 0; i < remainingIncompleteIds.length; i += BATCH_SIZE) {
            const batch = remainingIncompleteIds.slice(i, i + BATCH_SIZE);
            const batchResults = await Promise.all(
              batch.map(async id => {
                try {
                  const r = await fetch(`/api/entry/${id}`);
                  if (!r.ok) return null;
                  const e = await r.json();
                  return { ...e, id: `dataset.jsonl:${e.id}` };
                } catch { return null; }
              })
            );
            const valid = batchResults.filter(Boolean) as ConversationEntry[];
            if (valid.length > 0) handleLoad(valid);
            setLoadStatus(`Syncing queue... ${Math.min(i + BATCH_SIZE, remainingIncompleteIds.length)}/${remainingIncompleteIds.length}`);
          }
          setLoadStatus('');

        } catch (err) {
          console.error('[AutoLoad] Failed:', err);
          setLoadError(err instanceof Error ? err.message : String(err));
          setIsLoading(false); // Make sure to unblock even on error so user sees the error UI
        }
      })();
    }
  }, [appState.handle, entries.length, isLoading, handleLoad]);


  const handleLabelSentence = useCallback(
    (labels: string[]) => {
      if (!currentEntry) return;
      const sentIdx = appState.currentSentenceIndex;
      setAppState((prev) => {
        const ep = prev.progress[currentEntry.id] ?? { 
          sentenceLabels: {}, 
          sentenceScores: {}, 
          sentenceConfidences: {},
          completed: false 
        };
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

  const handleToggleConfidence = useCallback(
    (label: string) => {
      if (!currentEntry) return;
      const sentIdx = appState.currentSentenceIndex;
      setAppState((prev) => {
        const ep = prev.progress[currentEntry.id] ?? { 
          sentenceLabels: {}, 
          sentenceScores: {}, 
          sentenceConfidences: {},
          completed: false 
        };
        const currentConfMap = ep.sentenceConfidences?.[sentIdx] ?? {};
        const currentVal = currentConfMap[label] ?? 1;
        const newVal = currentVal === 1 ? 0 : 1;
        
        return {
          ...prev,
          progress: {
            ...prev.progress,
            [currentEntry.id]: {
              ...ep,
              sentenceConfidences: {
                ...ep.sentenceConfidences,
                [sentIdx]: { ...currentConfMap, [label]: newVal }
              }
            }
          }
        };
      });
    },
    [currentEntry, appState.currentSentenceIndex]
  );

  const handleScoreSentence = useCallback(
    (score: number) => {
      if (!currentEntry) return;
      const sentIdx = appState.currentSentenceIndex;
      setAppState((prev) => {
        const ep = prev.progress[currentEntry.id] ?? { 
          sentenceLabels: {}, 
          sentenceScores: {}, 
          sentenceConfidences: {},
          completed: false 
        };
        return {
          ...prev,
          progress: {
            ...prev.progress,
            [currentEntry.id]: {
              ...ep,
              sentenceScores: { ...(ep.sentenceScores ?? {}), [sentIdx]: score },
            },
          },
        };
      });
    },
    [currentEntry, appState.currentSentenceIndex],
  );

  const handleNavigate = useCallback(
    (direction: 'back' | 'next') => {
      if (!currentEntry) return;
      setAppState((prev) => {
        const idx = prev.currentSentenceIndex;
        const nextIdx =
          direction === 'back'
            ? Math.max(0, idx - 1)
            : Math.min(sentences.length - 1, idx + 1);
        if (direction === 'next') {
          const ep = prev.progress[currentEntry.id] ?? { 
            sentenceLabels: {}, 
            sentenceScores: {}, 
            sentenceConfidences: {},
            completed: false 
          };
          let currentLabels = ep.sentenceLabels[idx];
          let currentScore = ep.sentenceScores[idx];
          
          if (currentLabels === undefined || currentScore === undefined) {
            if (currentLabels === undefined && prev.llmHintsEnabled && llmHints) {
              const votes = llmHints.labelVotes[idx] ?? {};
              const pointsMap = llmHints.labelPoints[idx] ?? {};
              const preSelected = Object.entries(votes)
                .filter(([, c]) => (c as number) > 0)
                .sort(([labelA], [labelB]) => {
                  const ptsA = pointsMap[labelA] ?? 0;
                  const ptsB = pointsMap[labelB] ?? 0;
                  // Weighted score: 1000 points per point + taxonomy weight tie-breaker
                  const scoreA = ptsA * 1000 + (LABEL_WEIGHTS[labelA] ?? 0);
                  const scoreB = ptsB * 1000 + (LABEL_WEIGHTS[labelB] ?? 0);
                  return scoreB - scoreA;
                })
                .slice(0, 3)
                .map(([l]) => l);
              if (preSelected.length > 0) {
                currentLabels = preSelected;
              }
            }
            return {
              ...prev,
              currentSentenceIndex: nextIdx,
              progress: {
                ...prev.progress,
                [currentEntry.id]: {
                  ...ep,
                  sentenceLabels: { ...ep.sentenceLabels, [idx]: currentLabels ?? [] },
                  sentenceScores: { ...ep.sentenceScores, [idx]: currentScore ?? 0 },
                },
              },
            };
          }
        }
        return { ...prev, currentSentenceIndex: nextIdx };
      });
    },
    [sentences.length, currentEntry, llmHints],
  );

  const handleJump = useCallback((idx: number) => {
    setAppState((prev) => ({ ...prev, currentSentenceIndex: idx }));
  }, []);

  const handleAdvanceEntry = useCallback(() => {
    setAppState((prev) => {
      // Find the next incomplete entry, starting from the current position
      let nextIdx = (prev.currentEntryIndex + 1) % entries.length;
      let checkedCount = 0;
      while (
        checkedCount < entries.length && 
        prev.progress[entries[nextIdx].id]?.completed
      ) {
        nextIdx = (nextIdx + 1) % entries.length;
        checkedCount++;
      }

      if (checkedCount === entries.length) {
        console.log('[Submit] All entries completed!');
        nextIdx = entries.length; // Signal all done
      } else {
        console.log(`[Submit] Moving to next entry index: ${nextIdx}`);
      }

      return {
        ...prev,
        currentEntryIndex: nextIdx,
        currentSentenceIndex: 0,
      };
    });
    setScreen('labeling');
  }, [entries]);

  const handleSubmitAssessment = useCallback(
    async (assessment: Assessment) => {
      if (!currentEntry || !appState.handle) return;
      console.log(`[Submit] Submitting assessment for ${currentEntry.id}`);

      const ep = appState.progress[currentEntry.id] ?? { 
        sentenceLabels: {}, 
        sentenceScores: {}, 
        sentenceConfidences: {},
        completed: false 
      };
      
      const labeledEntry = buildHumanLabeledEntry(
        currentEntry,
        ep.sentenceLabels,
        ep.sentenceScores,
        ep.sentenceConfidences ?? {},
        assessment,
        appState.handle,
        appState.llmHintsEnabled,
      );

      try {
        const resp = await fetch('/api/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(labeledEntry),
        });
        if (!resp.ok) console.error('Failed to save to backend');
      } catch (err) {
        console.error('Error saving to backend:', err);
      }

      setAppState((prev) => {
        const currentEp = prev.progress[currentEntry.id] ?? { sentenceLabels: {}, sentenceScores: {}, completed: false };
        const newProgress = {
          ...prev.progress,
          [currentEntry.id]: { ...currentEp, assessment, completed: true },
        };

        return {
          ...prev,
          progress: newProgress,
        };
      });

      if (appState.showGamification) {
        setScreen('gamification');
      } else {
        handleAdvanceEntry();
      }
    },
    [currentEntry, appState.handle, appState.progress, appState.showGamification, appState.llmHintsEnabled, handleAdvanceEntry],
  );

  const handleMarkBroken = useCallback(
    async () => {
      if (!currentEntry || !appState.handle) return;
      if (!window.confirm('Are you sure you want to mark this entry as broken?')) return;
      
      console.log(`[Submit] Marking assessment for ${currentEntry.id} as broken`);

      const ep = appState.progress[currentEntry.id] ?? { 
        sentenceLabels: {}, 
        sentenceScores: {}, 
        sentenceConfidences: {},
        completed: false 
      };
      
      // Pass a default assessment to pass frontend checks, but status = 'error' is the main signal
      const mockAssessment = { trajectory: 'ambiguous', turning_point: -1, alignment: 'ambiguous' } as any;

      const labeledEntry = buildHumanLabeledEntry(
        currentEntry,
        ep.sentenceLabels,
        ep.sentenceScores,
        ep.sentenceConfidences ?? {},
        mockAssessment,
        appState.handle,
        appState.llmHintsEnabled,
        'error',
      );

      try {
        const resp = await fetch('/api/save', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(labeledEntry),
        });
        if (!resp.ok) console.error('Failed to save to backend');
      } catch (err) {
        console.error('Error saving to backend:', err);
      }

      setAppState((prev) => {
        const currentEp = prev.progress[currentEntry.id] ?? { sentenceLabels: {}, sentenceScores: {}, completed: false };
        const newProgress = {
          ...prev.progress,
          [currentEntry.id]: { ...currentEp, assessment: mockAssessment, completed: true, status: 'error' },
        };

        return {
          ...prev,
          progress: newProgress,
        };
      });

      handleAdvanceEntry();
    },
    [currentEntry, appState.handle, appState.progress, appState.llmHintsEnabled, handleAdvanceEntry],
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

  const handleSelectEntry = useCallback((idx: number) => {
    setAppState((prev) => ({ ...prev, currentEntryIndex: idx, currentSentenceIndex: 0 }));
    setScreen('labeling');
  }, []);

  // ── Routing ───────────────────────────────────────────────────────────────

  if (!appState.handle) {
    return <IdentityScreen onSetHandle={handleSetHandle} />;
  }

  if (isLoading || (appState.handle && entries.length === 0)) {
    return (
      <div style={{
        height: '100dvh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#2D353B',
        gap: '20px',
        padding: '24px',
        textAlign: 'center'
      }}>
        {loadError ? (
          <>
            <div style={{ color: '#E67E80', fontSize: '14px', maxWidth: '400px', lineHeight: 1.5 }}>
              <strong>Error loading data:</strong><br />
              {loadError}
            </div>
            <button
              onClick={() => {
                hasAutoLoaded.current = false;
                setAppState(prev => ({ ...prev, handle: null }));
              }}
              style={{
                background: '#A7C080',
                border: 'none',
                borderRadius: '8px',
                padding: '10px 20px',
                color: '#2D353B',
                fontWeight: 700,
                cursor: 'pointer'
              }}
            >
              ← Back to Name Selection
            </button>
          </>
        ) : (
          <>
            <div className="spinner" style={{
              width: '24px',
              height: '24px',
              border: '3px solid #475258',
              borderTopColor: '#A7C080',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }} />
            <div style={{ color: '#7A8478', fontSize: '14px' }}>
              {loadStatus || 'Preparing labeling queue...'}
            </div>
            <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
          </>
        )}
      </div>
    );
  }

  return (
    <>
      {currentEntry ? (
        <Labeler
          handle={appState.handle || 'Unknown'}
          entry={currentEntry}
          entryIndex={appState.currentEntryIndex}
          totalEntries={totalExpectedCount || entries.length}
          completedCount={activeProgressDisplay}
          sentenceIndex={appState.currentSentenceIndex}
          sentenceLabels={currentProgress?.sentenceLabels ?? {}}
          sentenceScores={currentProgress?.sentenceScores ?? {}}
          sentenceConfidences={currentProgress?.sentenceConfidences ?? {}}
          currentProgress={currentProgress}
          llmHintsEnabled={appState.llmHintsEnabled}
          llmLabelVotes={llmHints?.labelVotes[appState.currentSentenceIndex] ?? {}}
          llmLabelPoints={llmHints?.labelPoints[appState.currentSentenceIndex] ?? {}}
          llmScoreVotes={llmHints?.scoreVotes[appState.currentSentenceIndex] ?? {}}
          onLabelSentence={handleLabelSentence}
          onToggleConfidence={handleToggleConfidence}
          onScoreSentence={handleScoreSentence}
          onNavigate={handleNavigate}
          onJumpToSentence={handleJump}
          onSubmitAssessment={handleSubmitAssessment}
          onShowStats={() => setScreen('stats')}
          onToggleLlmHints={handleToggleLlmHints}
          onMarkBroken={handleMarkBroken}
          allDone={allSentencesLabeled}
        />
      ) : (
        <AllDoneScreen
          completedCount={completedCount}
          total={totalExpectedCount || entries.length}
          onShowStats={() => setScreen('stats')}
          onLoadMore={handleLoad}
        />
      )}

      {screen === 'stats' && (
        <StatsView
          entries={entries}
          state={appState}
          onClose={() => setScreen('labeling')}
          onReset={handleReset}
          onClearData={handleClearAll}
          onSelectEntry={handleSelectEntry}
        />
      )}

      {screen === 'gamification' && currentEntry && (
        <GamificationModal
          entry={currentEntry}
          progress={appState.progress[currentEntry.id]}
          onNext={handleAdvanceEntry}
          onDisable={() => {
            setAppState(prev => ({ ...prev, showGamification: false }));
            handleAdvanceEntry();
          }}
        />
      )}

      {/* Optional: Background sync progress indicator */}
      {loadStatus && (
        <div style={{
          position: 'fixed',
          bottom: '20px',
          right: '20px',
          background: '#343F44',
          padding: '8px 12px',
          borderRadius: '8px',
          color: '#7A8478',
          fontSize: '11px',
          border: '1px solid #475258',
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <div className="spinner-small" style={{
            width: '10px',
            height: '10px',
            border: '2px solid #475258',
            borderTopColor: '#A7C080',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <span>{loadStatus}</span>
          <style>{`
            @keyframes spin { to { transform: rotate(360deg); } }
          `}</style>
        </div>
      )}
    </>
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
        background: '#2D353B',
        gap: '24px',
        padding: '32px',
        userSelect: 'none',
        WebkitUserSelect: 'none',
      }}
    >
      <div style={{ textAlign: 'center' }}>
        <h1 style={{ color: '#D3C6AA', fontSize: '22px', fontWeight: 800, margin: 0 }}>
          All {total} responses labeled!
        </h1>
        <p style={{ color: '#7A8478', fontSize: '14px', marginTop: '8px' }}>
          Great work. Your labels were saved to dataset.jsonl.
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
            background: '#A7C080',
            border: 'none',
            borderRadius: '12px',
            color: '#2D353B',
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
            background: '#343F44',
            border: '1px solid #475258',
            borderRadius: '12px',
            color: '#9DA9A0',
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
