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
        <span className="text-xs font-medium text-ash">
          {questionNumber} / {totalQuestions}
        </span>
        {timeLimitSec > 0 && (
          <span className={`text-sm font-bold tabular-nums transition-colors
            ${timerCritical ? "text-red-600 animate-timer-pulse"
            : timerWarning  ? "text-amber-600"
            : "text-bark"}`}>
            ⏱ {formatTime(timeLeft)}
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-sand rounded-full overflow-hidden">
        <div
          className="h-full bg-brand-500 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Question */}
      <div className="bg-ivory border border-cream rounded-2xl p-5 shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
        <p className="text-base text-ink leading-relaxed font-medium">
          {question.question}
        </p>
        {question.from_history && (
          <span className="inline-block mt-2 text-xs bg-amber-50 text-amber-700 px-2 py-0.5 rounded-full border border-amber-200">
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
              className={`w-full text-left p-3.5 rounded-xl border text-sm transition-all
                active:scale-[0.99] font-medium
                ${isSelected
                  ? "border-brand-500 bg-brand-50 text-brand-700 shadow-[0px_0px_0px_1px_#c96442]"
                  : "border-cream bg-ivory text-bark hover:border-sand hover:text-ink"
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
        className="w-full bg-brand-500 hover:bg-brand-600 active:scale-[0.98] text-ivory py-2.5
          rounded-xl font-semibold text-sm transition-all disabled:opacity-40"
      >
        {questionNumber === totalQuestions ? "Finish Quiz" : "Next →"}
      </button>
    </div>
  );
}
