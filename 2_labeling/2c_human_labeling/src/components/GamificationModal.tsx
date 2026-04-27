import React, { useMemo } from 'react';
import type { ConversationEntry, EntryProgress } from '../types';
import { getSentences, codePointToCodeUnitOffset } from '../store';

export default function GamificationModal({
  entry,
  progress,
  onNext,
  onDisable,
}: {
  entry: ConversationEntry;
  progress: EntryProgress;
  onNext: () => void;
  onDisable: () => void;
}) {
  const stats = useMemo(() => {
    const sentences = getSentences(entry);
    const reasoningIdx = entry.messages.findIndex((m) => m.role === 'reasoning');
    const reasoningText = reasoningIdx >= 0 ? entry.messages[reasoningIdx].content : '';

    const llmRuns = (entry.label_runs ?? []).filter(
      (r) => !r.judge_name.startsWith('human_') && r.status !== 'error' && !r.error
    );

    const matchCounts: Record<string, { matches: number; total: number }> = {};
    for (const run of llmRuns) {
      matchCounts[run.judge_name] = { matches: 0, total: 0 };
    }

    for (const sent of sentences) {
      const userLabels = progress.sentenceLabels[sent.index] ?? [];
      
      for (const run of llmRuns) {
        const overlapping = (run.spans ?? []).filter((sp) => {
          const spStart = codePointToCodeUnitOffset(reasoningText, sp.char_start);
          const spEnd = codePointToCodeUnitOffset(reasoningText, sp.char_end);
          return spStart < sent.char_end && spEnd > sent.char_start;
        });
        
        const llmLabels = new Set<string>();
        for (const sp of overlapping) {
          for (const lbl of sp.labels ?? []) {
            if (lbl && lbl !== 'none') llmLabels.add(lbl);
          }
        }
        
        const userSet = new Set(userLabels);
        let intersection = 0;
        const union = new Set([...llmLabels, ...userSet]).size;
        
        for (const lbl of userSet) {
          if (llmLabels.has(lbl)) intersection++;
        }
        
        if (union > 0) {
          matchCounts[run.judge_name].matches += intersection;
          matchCounts[run.judge_name].total += union;
        }
      }
    }

    return Object.entries(matchCounts)
      .map(([name, counts]) => ({
        name,
        percentage: counts.total > 0 ? (counts.matches / counts.total) * 100 : 0,
      }))
      .sort((a, b) => b.percentage - a.percentage);
  }, [entry, progress]);

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(35,42,46,0.95)',
      display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      zIndex: 9999, padding: '20px'
    }}>
      <div className="card" style={{ maxWidth: '500px', width: '100%', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        <h2 style={{ margin: 0, textAlign: 'center', color: '#A7C080' }}>Labeling Agreement</h2>
        <p style={{ margin: 0, textAlign: 'center', fontSize: '14px', color: '#D3C6AA' }}>
          Here is how your labels for this entry compared to the LLMs!
        </p>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '10px' }}>
          {stats.length === 0 && (
            <div style={{ textAlign: 'center', color: '#7A8478', fontSize: '14px' }}>
              No LLM labels available for comparison.
            </div>
          )}
          {stats.map(stat => (
            <div key={stat.name} style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                <span style={{ fontWeight: 600, color: '#D3C6AA' }}>{stat.name}</span>
                <span style={{ color: '#A7C080', fontWeight: 600 }}>{stat.percentage.toFixed(0)}% Match</span>
              </div>
              <div style={{ width: '100%', height: '8px', background: '#3D484D', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ 
                  height: '100%', 
                  width: `${stat.percentage}%`, 
                  background: 'linear-gradient(90deg, #83C092, #A7C080)',
                  transition: 'width 1s ease-out'
                }} />
              </div>
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: '12px', marginTop: '24px' }}>
          <button 
            onClick={onDisable}
            style={{ 
              flex: 1, padding: '12px', background: 'transparent', border: '1px solid #475258',
              borderRadius: '8px', color: '#7A8478', cursor: 'pointer', fontWeight: 600
            }}
          >
            Don't show this again
          </button>
          <button 
            onClick={onNext}
            style={{ 
              flex: 2, padding: '12px', background: '#A7C080', border: 'none',
              borderRadius: '8px', color: '#232A2E', cursor: 'pointer', fontWeight: 700
            }}
          >
            Next Entry ➔
          </button>
        </div>
      </div>
    </div>
  );
}
