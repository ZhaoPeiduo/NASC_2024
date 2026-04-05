import type { Phase } from "../types";

const PHASES: { key: Phase; label: string }[] = [
  { key: "answering",  label: "Selecting" },
  { key: "explaining", label: "Explaining" },
  { key: "done",       label: "Done" },
];

const ORDER: Record<string, number> = { answering: 0, explaining: 1, done: 2 };

export default function StreamingProgress({ phase }: { phase: Phase }) {
  if (phase === "idle") return null;
  const cur = ORDER[phase] ?? -1;
  return (
    <div className="flex items-center gap-2 my-3 animate-slide-in">
      {PHASES.map(({ key, label }, i) => {
        const idx = ORDER[key];
        const done   = idx < cur;
        const active = idx === cur;
        return (
          <div key={key} className="flex items-center gap-2">
            {i > 0 && (
              <div className={`h-px w-5 transition-all duration-500 ${done ? "bg-brand-500" : "bg-slate-200"}`} />
            )}
            <span className={`text-xs font-semibold px-2 py-0.5 rounded-full transition-all duration-300
              ${done   ? "bg-brand-100 text-brand-700"
              : active ? "bg-brand-500 text-white animate-timer-pulse"
              : "bg-slate-100 text-slate-400"}`}>
              {label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
