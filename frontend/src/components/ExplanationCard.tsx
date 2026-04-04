import type { Phase, SolveResult } from "../types";

export default function ExplanationCard({
  streamText, result, phase
}: {
  streamText: string; result: SolveResult | null; phase: Phase;
}) {
  if (phase === "idle") return null;

  return (
    <div className="mt-6 space-y-4">
      {result ? (
        <>
          <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
            <span className="text-2xl font-bold text-green-700">{result.answer}</span>
            <p className="text-sm text-slate-700">{result.explanation}</p>
          </div>

          {Object.entries(result.wrong_options).length > 0 && (
            <div className="p-4 bg-slate-50 rounded-xl space-y-2">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Why the others are wrong</p>
              {Object.entries(result.wrong_options).map(([opt, reason]) => (
                <div key={opt} className="flex gap-2 text-sm">
                  <span className="font-bold text-red-500 w-4 shrink-0">{opt}</span>
                  <span className="text-slate-600">{reason}</span>
                </div>
              ))}
            </div>
          )}

          {result.concepts.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {result.concepts.map(c => (
                <span key={c} className="bg-brand-50 text-brand-700 text-xs px-2 py-1 rounded-full">{c}</span>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="p-4 bg-white border border-slate-200 rounded-xl min-h-24">
          <p className="text-sm text-slate-600 whitespace-pre-wrap">{streamText}
            <span className="animate-pulse">▌</span>
          </p>
        </div>
      )}
    </div>
  );
}
