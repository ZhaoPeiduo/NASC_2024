import type { Phase } from "../types";

const PHASES: { key: Phase; label: string }[] = [
  { key: "answering", label: "Selecting answer" },
  { key: "explaining", label: "Generating explanation" },
  { key: "done", label: "Done" },
];

const PHASE_ORDER: Record<string, number> = { answering: 0, explaining: 1, done: 2 };

export default function StreamingProgress({ phase }: { phase: Phase }) {
  if (phase === "idle") return null;
  const currentOrder = PHASE_ORDER[phase] ?? -1;
  return (
    <div className="flex items-center gap-3 mb-4">
      {PHASES.map(({ key, label }, i) => {
        const itemOrder = PHASE_ORDER[key];
        const isDone = itemOrder < currentOrder;
        const isActive = itemOrder === currentOrder;
        return (
          <div key={key} className="flex items-center gap-2">
            {i > 0 && <div className={`h-px w-6 ${isDone ? "bg-brand-500" : "bg-slate-200"}`} />}
            <span className={`text-xs font-medium px-2 py-1 rounded-full
              ${isDone ? "bg-brand-50 text-brand-700"
              : isActive ? "bg-brand-500 text-white animate-pulse"
              : "bg-slate-100 text-slate-400"}`}>
              {label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
