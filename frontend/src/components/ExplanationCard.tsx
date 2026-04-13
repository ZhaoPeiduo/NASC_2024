import type { Phase, SolveResult } from "../types";

export default function ExplanationCard({
  streamText, result, phase
}: {
  streamText: string; result: SolveResult | null; phase: Phase;
}) {
  if (phase === "idle") return null;

  return (
    <div className="mt-4 space-y-3 animate-fade-in">
      {result ? (
        <>
          <div className="flex items-start gap-3 p-4 bg-green-50 border border-green-200 rounded-2xl animate-pop-in">
            <span className="text-xl font-black text-green-700 shrink-0">{result.answer}</span>
            <p className="text-sm text-bark leading-relaxed">{result.explanation}</p>
          </div>

          {Object.entries(result.wrong_options).length > 0 && (
            <div className="p-4 bg-ivory border border-cream rounded-2xl space-y-2">
              <p className="text-xs font-medium text-ash uppercase tracking-wide">Why others are wrong</p>
              {Object.entries(result.wrong_options).map(([opt, reason]) => (
                <div key={opt} className="flex gap-2 text-sm animate-slide-in">
                  <span className="font-bold text-red-500 w-4 shrink-0">{opt}</span>
                  <span className="text-bark text-xs leading-relaxed">{reason}</span>
                </div>
              ))}
            </div>
          )}

          {result.concepts.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {result.concepts.map(c => (
                <span key={c} className="bg-brand-50 text-brand-600 text-xs px-2.5 py-0.5 rounded-full font-medium border border-brand-100">
                  {c}
                </span>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="p-4 bg-ivory border border-cream rounded-2xl min-h-16">
          <p className="text-sm text-bark whitespace-pre-wrap leading-relaxed">
            {streamText}<span className="animate-pulse">▌</span>
          </p>
        </div>
      )}
    </div>
  );
}
