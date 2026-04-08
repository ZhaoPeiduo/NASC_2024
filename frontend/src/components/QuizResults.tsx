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
    if (!attemptId) return;
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
      <div className="bg-white border border-slate-200 rounded-xl p-5 text-center animate-pop-in">
        <p className="text-5xl font-black text-brand-500">{pct}%</p>
        <p className="text-sm text-slate-500 mt-1">
          {score} correct · {wrong} wrong · {total} total
        </p>
        <div className="h-2 bg-slate-100 rounded-full overflow-hidden mt-3">
          <div
            className="h-full bg-brand-500 rounded-full transition-all duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Wrong answers with LLM analysis */}
      {wrong > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            {analyzing ? "Analyzing wrong answers…" : `${wrong} wrong answer${wrong > 1 ? "s" : ""}`}
          </p>

          {analyzing ? (
            <div className="space-y-2">
              {Array.from({ length: wrong }).map((_, i) => (
                <div key={i} className="h-16 bg-slate-100 rounded-xl animate-timer-pulse" />
              ))}
            </div>
          ) : analyses.length === 0 ? (
            <p className="text-xs text-slate-400 italic">Analysis unavailable</p>
          ) : (
            analyses.map((item, i) => {
              const explanation = explanations[i] || item.explanation;
              return (
                <div
                  key={i}
                  className="bg-white border border-red-100 rounded-xl p-3 space-y-2 animate-slide-in"
                  style={{ animationDelay: `${i * 60}ms` }}
                >
                  <p className="text-sm text-slate-800 leading-snug">{item.question}</p>
                  <div className="flex gap-3 text-xs">
                    <span className="text-red-500 font-semibold">You: {item.user_answer}</span>
                    <span className="text-green-600 font-semibold">Correct: {item.correct_answer}</span>
                  </div>
                  {explanation ? (
                    <p className="text-xs text-slate-600 leading-relaxed border-t border-slate-100 pt-2">
                      {explanation}
                    </p>
                  ) : wrongAttemptIds[i] ? (
                    <div className="border-t border-slate-100 pt-2">
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
        className="w-full border border-slate-200 hover:bg-slate-50 active:scale-[0.98]
          text-slate-600 py-2.5 rounded-xl text-sm font-medium transition-all"
      >
        ← Back to Setup
      </button>
    </div>
  );
}
