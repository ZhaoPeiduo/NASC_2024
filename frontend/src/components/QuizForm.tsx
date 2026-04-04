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
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
      <label className="block text-sm font-medium text-slate-700 mb-1">Question</label>
      <textarea
        value={question}
        onChange={e => onQuestionChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder="彼女は毎日日本語を＿＿＿います。"
        disabled={disabled}
        rows={3}
        className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm mb-4
          focus:outline-none focus:ring-2 focus:ring-brand-500 resize-none disabled:bg-slate-50"
      />

      <p className="text-sm font-medium text-slate-700 mb-2">Options</p>
      <div className="grid grid-cols-2 gap-3 mb-5">
        {options.map((opt, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-sm font-bold text-slate-500 w-4">{String.fromCharCode(65 + i)}</span>
            <input
              value={opt}
              onChange={e => onOptionChange(i, e.target.value)}
              onKeyDown={handleKey}
              placeholder={`Option ${String.fromCharCode(65 + i)}`}
              disabled={disabled}
              className="flex-1 border border-slate-200 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:bg-slate-50"
            />
          </div>
        ))}
      </div>

      {error && <p className="text-red-600 text-sm mb-3">{error}</p>}

      <div className="flex gap-3">
        <button onClick={onSubmit} disabled={disabled || !canSubmit}
          className="flex-1 bg-brand-500 hover:bg-brand-700 text-white py-2.5 rounded-lg
            font-medium text-sm transition-colors disabled:opacity-40"
        >
          {disabled ? "Analyzing…" : "Get Answer ⌘↵"}
        </button>
        <button onClick={onReset} disabled={disabled}
          className="px-4 py-2.5 border border-slate-200 rounded-lg text-sm text-slate-600
            hover:bg-slate-50 transition-colors disabled:opacity-40"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
