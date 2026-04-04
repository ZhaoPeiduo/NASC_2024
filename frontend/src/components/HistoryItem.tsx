import type { AttemptResponse } from "../types";

export default function HistoryItem({ attempt }: { attempt: AttemptResponse }) {
  const date = new Date(attempt.created_at).toLocaleDateString();
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-2">
      <div className="flex items-start justify-between gap-4">
        <p className="text-sm text-slate-800 flex-1">{attempt.question_text}</p>
        <span className={`text-xs font-bold px-2 py-1 rounded-full shrink-0
          ${attempt.user_marked_correct ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
          {attempt.user_marked_correct ? "Correct" : "Wrong"}
        </span>
      </div>
      <div className="flex items-center gap-3 text-xs text-slate-400">
        <span>Answer: <span className="font-semibold text-slate-600">{attempt.correct_answer}</span></span>
        <span>{date}</span>
      </div>
      {attempt.concepts.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {attempt.concepts.map(c => (
            <span key={c} className="bg-slate-100 text-slate-500 text-xs px-2 py-0.5 rounded-full">{c}</span>
          ))}
        </div>
      )}
    </div>
  );
}
