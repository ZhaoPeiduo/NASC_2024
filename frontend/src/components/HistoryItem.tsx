import { useState } from "react";
import { api } from "../api/client";
import type { AttemptResponse } from "../types";

interface Props {
  attempt: AttemptResponse;
  index?: number;
}

export default function HistoryItem({ attempt, index = 0 }: Props) {
  const date = new Date(attempt.created_at).toLocaleDateString();
  const [expanded, setExpanded] = useState(false);
  const [explanation, setExplanation] = useState(attempt.explanation || "");
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState("");

  const handleGenerate = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setGenerating(true);
    setGenError("");
    try {
      const res = await api.explainAttempt(attempt.id);
      setExplanation(res.explanation);
    } catch {
      setGenError("Failed to generate. Try again.");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div
      className="bg-white border border-slate-200 rounded-xl p-3 space-y-1.5 animate-fade-in
        hover:shadow-sm transition-shadow cursor-pointer"
      style={{ animationDelay: `${index * 80}ms` }}
      onClick={() => setExpanded(v => !v)}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm text-slate-800 flex-1 leading-snug">{attempt.question_text}</p>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full
            ${attempt.user_marked_correct ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"}`}>
            {attempt.user_marked_correct ? "✓" : "✗"}
          </span>
          <span className="text-slate-300 text-xs">{expanded ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* Summary row */}
      <div className="flex items-center gap-3 text-xs text-slate-400">
        <span>Answer: <span className="font-semibold text-slate-600">{attempt.correct_answer}</span></span>
        <span>{date}</span>
      </div>

      {/* Concepts */}
      {attempt.concepts.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {attempt.concepts.map(c => (
            <span key={c} className="bg-slate-100 text-slate-500 text-xs px-1.5 py-0.5 rounded-full">{c}</span>
          ))}
        </div>
      )}

      {/* Expanded section */}
      {expanded && (
        <div className="border-t border-slate-100 pt-2 space-y-2 animate-fade-in">
          {/* Options grid */}
          {attempt.options.length > 0 && (
            <div className="space-y-1">
              {attempt.options.map((opt, i) => {
                const label = opt.split(":")[0]?.trim() ?? String.fromCharCode(65 + i);
                const isCorrect = attempt.correct_answer === label;
                const isUserAnswer = attempt.llm_answer === label;
                return (
                  <div
                    key={i}
                    className={`text-xs px-2 py-1 rounded-lg
                      ${isCorrect
                        ? "bg-green-50 text-green-700 font-medium border border-green-200"
                        : isUserAnswer && !attempt.user_marked_correct
                          ? "bg-red-50 text-red-600 border border-red-200"
                          : "bg-slate-50 text-slate-600"
                      }`}
                  >
                    {opt}
                  </div>
                );
              })}
            </div>
          )}

          {/* Explanation */}
          {explanation ? (
            <p className="text-xs text-slate-600 leading-relaxed">{explanation}</p>
          ) : (
            <div>
              {genError && <p className="text-xs text-red-500 mb-1">{genError}</p>}
              <button
                onClick={handleGenerate}
                disabled={generating}
                className="text-xs text-brand-500 font-semibold hover:underline disabled:opacity-50"
              >
                {generating ? "Generating…" : "Generate explanation"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
