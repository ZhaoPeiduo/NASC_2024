import type { QuizQuestion } from "../types";

interface Props {
  question: QuizQuestion;
  questionNumber: number;
  totalQuestions: number;
  selected: string | null;
  onSelect: (answer: string) => void;
  onConfirm: () => void;
  timeLeft: number;
  timeLimitSec: number;
  timerWarning: boolean;
  timerCritical: boolean;
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function ActiveQuiz({
  question, questionNumber, totalQuestions, selected, onSelect, onConfirm,
  timeLeft, timeLimitSec, timerWarning, timerCritical,
}: Props) {
  const progress = ((questionNumber - 1) / totalQuestions) * 100;

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400">
          {questionNumber} / {totalQuestions}
        </span>
        {timeLimitSec > 0 && (
          <span className={`text-sm font-bold tabular-nums transition-colors
            ${timerCritical ? "text-red-500 animate-timer-pulse"
            : timerWarning  ? "text-amber-500"
            : "text-slate-500"}`}>
            ⏱ {formatTime(timeLeft)}
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-brand-500 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Question */}
      <div className="bg-white border border-slate-200 rounded-xl p-4">
        <p className="text-base text-slate-800 leading-relaxed font-medium">
          {question.question}
        </p>
        {question.from_history && (
          <span className="inline-block mt-2 text-xs bg-amber-50 text-amber-600 px-2 py-0.5 rounded-full">
            from history
          </span>
        )}
      </div>

      {/* Options */}
      <div className="space-y-2">
        {question.options.map((opt) => {
          const letter = opt.split(":")[0].trim();
          const isSelected = selected === letter;
          return (
            <button
              key={letter}
              onClick={() => onSelect(letter)}
              className={`w-full text-left p-3 rounded-xl border text-sm transition-all
                active:scale-99 font-medium
                ${isSelected
                  ? "border-brand-500 bg-brand-50 text-brand-700 shadow-sm"
                  : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"
                }`}
            >
              {opt}
            </button>
          );
        })}
      </div>

      <button
        onClick={onConfirm}
        disabled={!selected}
        className="w-full bg-brand-500 hover:bg-brand-600 active:scale-98 text-white py-2.5
          rounded-xl font-semibold text-sm transition-all disabled:opacity-40"
      >
        {questionNumber === totalQuestions ? "Finish Quiz" : "Next →"}
      </button>
    </div>
  );
}
