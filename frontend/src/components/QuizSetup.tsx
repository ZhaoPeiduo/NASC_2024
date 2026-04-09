import { useRef, useState } from "react";
import { api } from "../api/client";
import type { QuizQuestion } from "../types";

interface Props {
  onStart: (questions: QuizQuestion[], timeSec: number) => void;
}

export default function QuizSetup({ onStart }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState("");
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [minutes, setMinutes] = useState(10);
  const [includeHistory, setIncludeHistory] = useState(false);
  const [historyCount, setHistoryCount] = useState(5);
  const [maxQuestions, setMaxQuestions] = useState(0); // 0 = all
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = async (file: File) => {
    setLoading(true); setError(""); setQuestions([]);
    setFileName(file.name);
    try {
      const res = await api.uploadPracticeCSV(file, includeHistory, historyCount);
      setQuestions(res.questions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to parse CSV");
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="space-y-4 animate-fade-in">
      <div>
        <h1 className="text-lg font-bold text-stone-800">Timed Quiz</h1>
        <p className="text-xs text-stone-500 mt-0.5">Upload a grammar CSV to start a timed practice session</p>
      </div>

      {/* CSV Upload */}
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed border-stone-200 rounded-2xl p-6 text-center cursor-pointer
          hover:border-brand-500 transition-colors active:scale-[0.99]"
      >
        <input ref={inputRef} type="file" accept=".csv" className="hidden"
          onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
        {loading ? (
          <p className="text-sm text-brand-500 animate-timer-pulse">Parsing CSV…</p>
        ) : fileName ? (
          <div>
            <p className="text-sm font-medium text-stone-700">{fileName}</p>
            <p className="text-xs text-green-600 mt-1">{questions.length} questions loaded</p>
          </div>
        ) : (
          <div>
            <p className="text-sm text-stone-400">Drop CSV here or click to upload</p>
            <p className="text-xs text-stone-300 mt-1">Format: Question, Options, Answer</p>
          </div>
        )}
      </div>
      {error && <p className="text-red-500 text-xs">{error}</p>}

      {/* Config */}
      <div className="bg-white border border-stone-200 rounded-2xl p-4 space-y-3">
        {/* Timer */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-stone-700">Time limit</p>
            <p className="text-xs text-stone-400">Set 0 for unlimited</p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setMinutes(m => Math.max(0, m - 5))}
              aria-label="Decrease time limit"
              className="w-7 h-7 rounded-lg border border-stone-200 text-stone-500 hover:bg-stone-50
                active:scale-95 transition-all text-sm flex items-center justify-center">−</button>
            <span className="text-sm font-semibold text-stone-800 w-12 text-center">
              {minutes === 0 ? "∞" : `${minutes}m`}
            </span>
            <button onClick={() => setMinutes(m => m + 5)}
              aria-label="Increase time limit"
              className="w-7 h-7 rounded-lg border border-stone-200 text-stone-500 hover:bg-stone-50
                active:scale-95 transition-all text-sm flex items-center justify-center">+</button>
          </div>
        </div>

        {/* Include wrong history */}
        <div className="flex items-center justify-between border-t border-stone-100 pt-3">
          <div>
            <p className="text-sm font-medium text-stone-700">Mix in wrong history</p>
            <p className="text-xs text-stone-400">Add past wrong answers to practice</p>
          </div>
          <button
            onClick={() => setIncludeHistory(v => !v)}
            role="switch"
            aria-checked={includeHistory}
            aria-label="Mix in wrong history from past attempts"
            className={`relative w-10 h-5 rounded-full transition-colors ${includeHistory ? "bg-brand-500" : "bg-stone-200"}`}
          >
            <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform
              ${includeHistory ? "translate-x-5" : "translate-x-0.5"}`} />
          </button>
        </div>

        {includeHistory && (
          <div className="flex items-center justify-between animate-fade-in">
            <p className="text-xs text-stone-500">How many wrong questions to add</p>
            <div className="flex items-center gap-2">
              <button onClick={() => setHistoryCount(n => Math.max(1, n - 1))}
                aria-label="Decrease history question count"
                className="w-6 h-6 rounded border border-stone-200 text-xs text-stone-500
                  hover:bg-stone-50 active:scale-95 transition-all">−</button>
              <span className="text-sm font-semibold text-stone-800 w-6 text-center">{historyCount}</span>
              <button onClick={() => setHistoryCount(n => Math.min(20, n + 1))}
                aria-label="Increase history question count"
                className="w-6 h-6 rounded border border-stone-200 text-xs text-stone-500
                  hover:bg-stone-50 active:scale-95 transition-all">+</button>
            </div>
          </div>
        )}

        {/* Question count */}
        <div className="flex items-center justify-between border-t border-stone-100 pt-3">
          <div>
            <p className="text-sm font-medium text-stone-700">Max questions</p>
            <p className="text-xs text-stone-400">Set 0 for all</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setMaxQuestions(n => Math.max(0, n - 5))}
              aria-label="Decrease max questions"
              className="w-7 h-7 rounded-lg border border-stone-200 text-stone-500 hover:bg-stone-50
                active:scale-95 transition-all text-sm flex items-center justify-center"
            >−</button>
            <span className="text-sm font-semibold text-stone-800 w-12 text-center">
              {maxQuestions === 0 ? "All" : `${maxQuestions}`}
            </span>
            <button
              onClick={() => setMaxQuestions(n => n + 5)}
              aria-label="Increase max questions"
              className="w-7 h-7 rounded-lg border border-stone-200 text-stone-500 hover:bg-stone-50
                active:scale-95 transition-all text-sm flex items-center justify-center"
            >+</button>
          </div>
        </div>
      </div>

      <button
        disabled={questions.length === 0}
        onClick={() => {
          const qs = maxQuestions > 0 ? questions.slice(0, maxQuestions) : questions;
          onStart(qs, minutes * 60);
        }}
        className="w-full bg-brand-500 hover:bg-brand-600 active:scale-[0.98] text-white py-2.5 rounded-xl
          font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {questions.length > 0
          ? `Start Quiz — ${maxQuestions > 0 ? Math.min(maxQuestions, questions.length) : questions.length} questions`
          : "Upload CSV to start"}
      </button>
    </div>
  );
}
