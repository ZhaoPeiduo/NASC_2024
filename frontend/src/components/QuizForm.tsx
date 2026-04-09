import type { KeyboardEvent } from "react";

interface Props {
  question: string;
  options: string[];
  onQuestionChange: (v: string) => void;
  onOptionChange: (i: number, v: string) => void;
  onSubmit: () => void;
  onReset: () => void;
  disabled: boolean;
  error: string;
}

export default function QuizForm({
  question, options, onQuestionChange, onOptionChange,
  onSubmit, onReset, disabled, error
}: Props) {
  const handleKey = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onSubmit();
  };
  const canSubmit = question.trim() && options.filter(o => o.trim()).length >= 2;

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-stone-200 p-4 animate-fade-in">
      <label className="block text-xs font-semibold text-stone-500 uppercase tracking-wide mb-1">Question</label>
      <textarea
        value={question}
        onChange={e => onQuestionChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder="彼女は毎日日本語を＿＿＿います。"
        disabled={disabled}
        rows={2}
        className="w-full border border-stone-200 rounded-xl px-3 py-2 text-sm mb-3
          focus:outline-none focus:ring-2 focus:ring-brand-500 resize-none
          disabled:bg-stone-50 transition-colors"
      />

      <p className="text-xs font-semibold text-stone-500 uppercase tracking-wide mb-2">Options</p>
      <div className="grid grid-cols-2 gap-2 mb-4">
        {options.map((opt, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <span className="text-xs font-bold text-stone-400 w-3">{String.fromCharCode(65 + i)}</span>
            <input
              value={opt}
              onChange={e => onOptionChange(i, e.target.value)}
              onKeyDown={handleKey}
              placeholder={`Option ${String.fromCharCode(65 + i)}`}
              disabled={disabled}
              className="flex-1 border border-stone-200 rounded-xl px-2.5 py-1.5 text-sm
                focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:bg-stone-50
                transition-colors hover:border-stone-300"
            />
          </div>
        ))}
      </div>

      {error && <p className="text-red-500 text-xs mb-2">{error}</p>}

      <div className="flex gap-2">
        <button onClick={onSubmit} disabled={disabled || !canSubmit}
          className="flex-1 bg-brand-500 hover:bg-brand-600 active:scale-95 text-white py-2 rounded-lg
            font-medium text-sm transition-all disabled:opacity-40"
        >
          {disabled ? "Analyzing…" : "Get Answer ⌘↵"}
        </button>
        <button onClick={onReset} disabled={disabled}
          className="px-3 py-2 border border-stone-200 rounded-xl text-sm text-stone-500
            hover:bg-stone-50 active:scale-95 transition-all disabled:opacity-40"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
