import { useState } from "react";
import { api } from "../api/client";
import type { AnalysisItem } from "../types";

interface Props {
  score: number;
  total: number;
  analyses: AnalysisItem[];
  analyzing: boolean;
  wrongAttemptIds: number[];
  onRetry: () => void;
}

export default function QuizResults({
  score, total, analyses, analyzing, wrongAttemptIds, onRetry,
}: Props) {
  const pct = total > 0 ? Math.round((score / total) * 100) : 0;
  const wrong = total - score;
  const [explanations, setExplanations] = useState<Record<number, string>>({});
  const [generating, setGenerating] = useState<Record<number, boolean>>({});

  const handleGenerate = async (index: number) => {
    const attemptId = wrongAttemptIds[index];
    if (attemptId == null) return;
    setGenerating(g => ({ ...g, [index]: true }));
    try {
      const res = await api.explainAttempt(attemptId);
      setExplanations(e => ({ ...e, [index]: res.explanation }));
    } catch {/* ignore */} finally {
      setGenerating(g => ({ ...g, [index]: false }));
    }
  };

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Score card */}
      <div className="bg-ivory border border-cream rounded-2xl p-6 text-center animate-pop-in shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
        <p className="text-5xl font-black text-brand-500">{pct}%</p>
        <p className="text-sm text-ash mt-1">
          {score} correct · {wrong} wrong · {total} total
        </p>
        <div className="h-2 bg-sand rounded-full overflow-hidden mt-4">
          <div
            className="h-full bg-brand-500 rounded-full transition-all duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Wrong answers with LLM analysis */}
      {wrong > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-medium text-ash uppercase tracking-wide">
            {analyzing ? "Analyzing wrong answers…" : `${wrong} wrong answer${wrong > 1 ? "s" : ""}`}
          </p>

          {analyzing ? (
            <div className="space-y-2">
              {Array.from({ length: wrong }).map((_, i) => (
                <div key={i} className="h-16 bg-sand rounded-2xl animate-timer-pulse" />
              ))}
            </div>
          ) : analyses.length === 0 ? (
            wrongAttemptIds.length > 0 ? (
              <div className="space-y-2">
                {wrongAttemptIds.map((_id, i) => (
                  <div key={i} className="bg-ivory border border-cream rounded-2xl p-3.5 animate-slide-in"
                    style={{ animationDelay: `${i * 60}ms` }}>
                    <button
                      onClick={() => handleGenerate(i)}
                      disabled={generating[i]}
                      className="text-xs text-brand-500 font-semibold hover:underline disabled:opacity-50"
                    >
                      {generating[i] ? "Generating…" : "Generate explanation"}
                    </button>
                    {explanations[i] && (
                      <p className="text-xs text-bark leading-relaxed mt-2">{explanations[i]}</p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-ash italic">Analysis unavailable</p>
            )
          ) : (
            analyses.map((item, i) => {
              const explanation = explanations[i] || item.explanation;
              return (
                <div
                  key={i}
                  className="bg-ivory border border-cream rounded-2xl p-4 space-y-2 animate-slide-in"
                  style={{ animationDelay: `${i * 60}ms` }}
                >
                  <p className="text-sm text-ink leading-snug">{item.question}</p>
                  <div className="flex gap-3 text-xs">
                    <span className="text-red-600 font-semibold">You: {item.user_answer}</span>
                    <span className="text-green-700 font-semibold">Correct: {item.correct_answer}</span>
                  </div>
                  {explanation ? (
                    <p className="text-xs text-bark leading-relaxed border-t border-cream pt-2">
                      {explanation}
                    </p>
                  ) : wrongAttemptIds[i] ? (
                    <div className="border-t border-cream pt-2">
                      <button
                        onClick={() => handleGenerate(i)}
                        disabled={generating[i]}
                        className="text-xs text-brand-500 font-semibold hover:underline disabled:opacity-50"
                      >
                        {generating[i] ? "Generating…" : "Generate explanation"}
                      </button>
                    </div>
                  ) : null}
                </div>
              );
            })
          )}
        </div>
      )}

      <button
        onClick={onRetry}
        className="w-full border border-cream bg-sand hover:bg-sand/80 active:scale-[0.98]
          text-charcoal py-2.5 rounded-xl text-sm font-medium transition-all"
      >
        ← Back to Setup
      </button>
    </div>
  );
}
