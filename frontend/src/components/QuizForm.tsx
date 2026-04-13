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
    <div className="bg-ivory rounded-2xl border border-cream shadow-[rgba(0,0,0,0.05)_0px_4px_24px] p-5 animate-fade-in">
      <label className="block text-xs font-medium text-ash uppercase tracking-wide mb-1.5">Question</label>
      <textarea
        value={question}
        onChange={e => onQuestionChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder="彼女は毎日日本語を＿＿＿います。"
        disabled={disabled}
        rows={2}
        className="w-full border border-cream bg-white rounded-xl px-3 py-2.5 text-sm text-ink mb-4
          focus:outline-none focus:ring-2 focus:ring-[#3898ec] focus:border-[#3898ec] resize-none
          disabled:bg-ivory placeholder:text-ash transition-colors"
      />

      <p className="text-xs font-medium text-ash uppercase tracking-wide mb-2">Options</p>
      <div className="grid grid-cols-2 gap-2 mb-4">
        {options.map((opt, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <span className="text-xs font-medium text-ash w-3">{String.fromCharCode(65 + i)}</span>
            <input
              value={opt}
              onChange={e => onOptionChange(i, e.target.value)}
              onKeyDown={handleKey}
              placeholder={`Option ${String.fromCharCode(65 + i)}`}
              disabled={disabled}
              className="flex-1 border border-cream bg-white rounded-lg px-2.5 py-1.5 text-sm text-ink
                focus:outline-none focus:ring-2 focus:ring-[#3898ec] focus:border-[#3898ec]
                disabled:bg-ivory placeholder:text-ash transition-colors hover:border-sand"
            />
          </div>
        ))}
      </div>

      {error && <p className="text-red-700 text-xs mb-2">{error}</p>}

      <div className="flex gap-2">
        <button onClick={onSubmit} disabled={disabled || !canSubmit}
          className="flex-1 bg-brand-500 hover:bg-brand-600 active:scale-95 text-ivory py-2 rounded-lg
            font-medium text-sm transition-all disabled:opacity-40"
        >
          {disabled ? "Analyzing…" : "Get Answer ⌘↵"}
        </button>
        <button onClick={onReset} disabled={disabled}
          className="px-3 py-2 border border-cream bg-sand rounded-xl text-sm text-charcoal
            hover:bg-sand/80 active:scale-95 transition-all disabled:opacity-40"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
