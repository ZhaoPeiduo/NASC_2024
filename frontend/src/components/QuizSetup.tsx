import { useRef, useState } from "react";
import { api } from "../api/client";
import type { QuizQuestion } from "../types";

type Mode = "csv" | "screenshots" | "generate";
type Level = "N5" | "N4" | "N3" | "N2" | "N1";

interface Props {
  onStart: (questions: QuizQuestion[], timeSec: number) => void;
}

export default function QuizSetup({ onStart }: Props) {
  const [mode, setMode] = useState<Mode>("csv");

  // Shared
  const [questions, setQuestions]   = useState<QuizQuestion[]>([]);
  const [minutes, setMinutes]       = useState(10);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState("");
  const [maxQuestions, setMaxQuestions] = useState(0);

  // CSV mode
  const inputRef   = useRef<HTMLInputElement>(null);
  const [fileName, setFileName]     = useState("");
  const [includeHistory, setIncludeHistory] = useState(false);
  const [historyCount, setHistoryCount]     = useState(5);

  // Screenshots mode
  const screenshotRef = useRef<HTMLInputElement>(null);
  const [screenshotFiles, setScreenshotFiles] = useState<File[]>([]);

  // Generate mode
  const [level, setLevel]           = useState<Level>("N4");
  const [concept, setConcept]       = useState("");
  const [genCount, setGenCount]     = useState(5);
  const [genProgress, setGenProgress] = useState(0);

  // --- CSV handlers ---
  const handleCSVFile = async (file: File) => {
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

  const handleCSVDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleCSVFile(file);
  };

  // --- Screenshots handlers ---
  const handleScreenshots = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const arr = Array.from(files);
    setScreenshotFiles(arr);
    setLoading(true); setError(""); setQuestions([]);
    try {
      const res = await api.uploadOcrImages(arr);
      setQuestions(res.questions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "OCR failed");
    } finally {
      setLoading(false);
    }
  };

  const handleScreenshotDrop = (e: React.DragEvent) => {
    e.preventDefault();
    handleScreenshots(e.dataTransfer.files);
  };

  // --- Generate handlers ---
  const handleGenerate = async () => {
    setLoading(true); setError(""); setQuestions([]); setGenProgress(0);
    const generated: QuizQuestion[] = [];
    try {
      await Promise.all(
        Array.from({ length: genCount }).map(async (_) => {
          const q = await api.generateQuestion(concept || "grammar", level);
          generated.push({
            question: q.question,
            options: q.options,
            correct_answer: q.correct_answer,
          });
          setGenProgress(p => p + 1);
        })
      );
      setQuestions(generated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  const handleStart = () => {
    const qs = maxQuestions > 0 ? questions.slice(0, maxQuestions) : questions;
    onStart(qs, minutes * 60);
  };

  const MODES: { id: Mode; label: string }[] = [
    { id: "csv",         label: "Upload CSV" },
    { id: "screenshots", label: "Screenshots" },
    { id: "generate",    label: "Generate with AI" },
  ];

  const LEVELS: Level[] = ["N5", "N4", "N3", "N2", "N1"];

  return (
    <div className="space-y-4 animate-fade-in">
      <div>
        <h1 className="font-serif text-xl font-medium text-ink">Timed Quiz</h1>
        <p className="text-xs text-ash mt-1">
          Build a question set, then test yourself under timed conditions. AI reviews every mistake after.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex bg-sand rounded-xl p-1 gap-1">
        {MODES.map(m => (
          <button
            key={m.id}
            onClick={() => { setMode(m.id); setQuestions([]); setError(""); }}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-all ${
              mode === m.id
                ? "bg-ivory text-ink shadow-sm border border-cream"
                : "text-bark hover:text-ink"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* --- CSV mode --- */}
      {mode === "csv" && (
        <>
          <div
            onDrop={handleCSVDrop}
            onDragOver={e => e.preventDefault()}
            onClick={() => inputRef.current?.click()}
            className="border-2 border-dashed border-cream rounded-2xl p-8 text-center cursor-pointer
              hover:border-brand-500 transition-colors active:scale-[0.99] bg-ivory"
          >
            <input ref={inputRef} type="file" accept=".csv" className="hidden"
              onChange={e => e.target.files?.[0] && handleCSVFile(e.target.files[0])} />
            {loading ? (
              <p className="text-sm text-brand-500 animate-timer-pulse">Parsing CSV…</p>
            ) : fileName ? (
              <div>
                <p className="text-sm font-medium text-ink">{fileName}</p>
                <p className="text-xs text-green-700 mt-1">{questions.length} questions loaded</p>
              </div>
            ) : (
              <div>
                <p className="text-sm text-ash">Drop CSV here or click to upload</p>
                <p className="text-xs text-ash/60 mt-1">Format: Question, Options, Answer</p>
              </div>
            )}
          </div>

          {/* CSV config */}
          <div className="bg-ivory border border-cream rounded-2xl p-5 space-y-4 shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-ink">Mix in wrong history</p>
                <p className="text-xs text-ash mt-0.5">Add past wrong answers to practice</p>
              </div>
              <button
                onClick={() => setIncludeHistory(v => !v)}
                role="switch"
                aria-checked={includeHistory}
                className={`relative w-10 h-5 rounded-full transition-colors ${includeHistory ? "bg-brand-500" : "bg-sand"}`}
              >
                <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform
                  ${includeHistory ? "translate-x-5" : "translate-x-0.5"}`} />
              </button>
            </div>
            {includeHistory && (
              <div className="flex items-center justify-between animate-fade-in">
                <p className="text-xs text-bark">Wrong questions to add</p>
                <div className="flex items-center gap-2">
                  <button onClick={() => setHistoryCount(n => Math.max(1, n - 1))}
                    className="w-6 h-6 rounded border border-cream bg-sand text-xs text-charcoal hover:bg-sand/80 active:scale-95 transition-all">−</button>
                  <span className="text-sm font-semibold text-ink w-6 text-center">{historyCount}</span>
                  <button onClick={() => setHistoryCount(n => Math.min(20, n + 1))}
                    className="w-6 h-6 rounded border border-cream bg-sand text-xs text-charcoal hover:bg-sand/80 active:scale-95 transition-all">+</button>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* --- Screenshots mode --- */}
      {mode === "screenshots" && (
        <div
          onDrop={handleScreenshotDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => screenshotRef.current?.click()}
          className="border-2 border-dashed border-cream rounded-2xl p-8 text-center cursor-pointer
            hover:border-brand-500 transition-colors active:scale-[0.99] bg-ivory"
        >
          <input ref={screenshotRef} type="file" accept="image/*" multiple className="hidden"
            onChange={e => handleScreenshots(e.target.files)} />
          {loading ? (
            <p className="text-sm text-brand-500 animate-timer-pulse">Running OCR…</p>
          ) : screenshotFiles.length > 0 ? (
            <div>
              <p className="text-sm font-medium text-ink">{screenshotFiles.length} image{screenshotFiles.length > 1 ? "s" : ""} uploaded</p>
              <p className="text-xs text-green-700 mt-1">{questions.length} questions extracted</p>
            </div>
          ) : (
            <div>
              <p className="text-sm text-ash">Drop screenshots here or click to upload</p>
              <p className="text-xs text-ash/60 mt-1">PNG or JPG — one question per image</p>
            </div>
          )}
        </div>
      )}

      {/* --- Generate with AI mode --- */}
      {mode === "generate" && (
        <div className="bg-ivory border border-cream rounded-2xl p-5 space-y-4 shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
          {/* Level */}
          <div>
            <p className="text-sm font-medium text-ink mb-2">JLPT Level</p>
            <div className="flex gap-2">
              {LEVELS.map(l => (
                <button
                  key={l}
                  onClick={() => setLevel(l)}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                    level === l
                      ? "bg-brand-500 text-ivory border-brand-500"
                      : "bg-sand text-bark border-cream hover:border-brand-500"
                  }`}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>

          {/* Concept */}
          <div>
            <p className="text-sm font-medium text-ink mb-1">Grammar concept <span className="text-ash font-normal">(optional)</span></p>
            <input
              type="text"
              value={concept}
              onChange={e => setConcept(e.target.value)}
              placeholder="e.g. て-form, は vs が, conditional"
              className="w-full px-3 py-2 rounded-xl border border-cream bg-parchment text-sm text-ink
                placeholder:text-ash/60 focus:outline-none focus:border-brand-500 transition-colors"
            />
          </div>

          {/* Count */}
          <div className="flex items-center justify-between border-t border-cream pt-4">
            <p className="text-sm font-medium text-ink">Questions to generate</p>
            <div className="flex items-center gap-2">
              <button onClick={() => setGenCount(n => Math.max(1, n - 1))}
                className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">−</button>
              <span className="text-sm font-semibold text-ink w-8 text-center">{genCount}</span>
              <button onClick={() => setGenCount(n => Math.min(20, n + 1))}
                className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">+</button>
            </div>
          </div>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full bg-brand-500 hover:bg-brand-600 active:scale-[0.98] text-ivory py-2 rounded-xl
              font-semibold text-sm transition-all disabled:opacity-40"
          >
            {loading
              ? `Generating ${genProgress} of ${genCount}…`
              : questions.length > 0
              ? `Regenerate (${questions.length} ready)`
              : "Generate Questions"}
          </button>

          {questions.length > 0 && !loading && (
            <p className="text-xs text-green-700 text-center">{questions.length} questions ready</p>
          )}
        </div>
      )}

      {error && <p className="text-red-700 text-xs">{error}</p>}

      {/* Shared: timer + max questions */}
      <div className="bg-ivory border border-cream rounded-2xl p-5 space-y-4 shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-ink">Time limit</p>
            <p className="text-xs text-ash mt-0.5">Set 0 for unlimited</p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setMinutes(m => Math.max(0, m - 5))}
              aria-label="Decrease time limit"
              className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">−</button>
            <span className="text-sm font-semibold text-ink w-12 text-center">
              {minutes === 0 ? "∞" : `${minutes}m`}
            </span>
            <button onClick={() => setMinutes(m => m + 5)}
              aria-label="Increase time limit"
              className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">+</button>
          </div>
        </div>

        <div className="flex items-center justify-between border-t border-cream pt-4">
          <div>
            <p className="text-sm font-medium text-ink">Max questions</p>
            <p className="text-xs text-ash mt-0.5">Set 0 for all</p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setMaxQuestions(n => Math.max(0, n - 5))}
              aria-label="Decrease max questions"
              className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">−</button>
            <span className="text-sm font-semibold text-ink w-12 text-center">
              {maxQuestions === 0 ? "All" : `${maxQuestions}`}
            </span>
            <button onClick={() => setMaxQuestions(n => n + 5)}
              aria-label="Increase max questions"
              className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">+</button>
          </div>
        </div>
      </div>

      <button
        disabled={questions.length === 0}
        onClick={handleStart}
        className="w-full bg-brand-500 hover:bg-brand-600 active:scale-[0.98] text-ivory py-2.5 rounded-xl
          font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {questions.length > 0
          ? `Start Quiz — ${maxQuestions > 0 ? Math.min(maxQuestions, questions.length) : questions.length} questions`
          : "Load questions to start"}
      </button>
    </div>
  );
}
