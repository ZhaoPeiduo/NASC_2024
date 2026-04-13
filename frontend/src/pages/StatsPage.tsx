import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { StatsResponse, GeneratedQuestion } from "../types";
import { BarStat } from "../components/StatsChart";

export default function StatsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [generating, setGenerating] = useState(false);
  const [generated, setGenerated] = useState<GeneratedQuestion | null>(null);
  const [genError, setGenError] = useState("");

  useEffect(() => { api.getStats().then(setStats); }, []);

  const generateForConcept = async (concept: string) => {
    setGenerating(true); setGenError(""); setGenerated(null);
    try {
      const q = await api.generateQuestion(concept, "N3");
      setGenerated(q);
    } catch {
      setGenError("Failed to generate question.");
    } finally {
      setGenerating(false);
    }
  };

  if (!stats) return <p className="text-ash text-center py-12">Loading…</p>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-serif text-2xl font-medium text-ink">Your Progress</h1>
        <p className="text-sm text-bark mt-1 leading-relaxed">
          Track accuracy, study days, and the grammar points that need more work.
        </p>
      </div>

      {stats.total_attempts === 0 ? (
        <div className="text-center py-16 space-y-3 bg-ivory rounded-2xl border border-cream shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
          <p className="text-2xl">📊</p>
          <p className="text-bark font-medium">Your journey starts today</p>
          <p className="text-ash text-sm max-w-xs mx-auto">
            Study at least once a day to build your streak and uncover your weak grammar points.
          </p>
          <a href="/ask"
            className="inline-block mt-1 text-sm text-brand-500 font-semibold hover:underline"
          >
            Start practicing →
          </a>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: "Total Attempts", value: stats.total_attempts },
              { label: "Accuracy", value: `${Math.round(stats.correct_rate * 100)}%` },
              { label: "Study Days", value: stats.study_days },
            ].map(({ label, value }) => (
              <div key={label} className="bg-ivory border border-cream rounded-2xl p-5 text-center shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
                <p className="text-3xl font-black text-ink">{value}</p>
                <p className="text-xs text-ash mt-1.5">{label}</p>
              </div>
            ))}
          </div>

          <div className="bg-ivory border border-cream rounded-2xl p-6 shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
            <BarStat label="Accuracy" value={stats.correct_rate} color="bg-brand-500" />
          </div>

          {stats.weak_concepts.length > 0 && (
            <div className="bg-ivory border border-cream rounded-2xl p-6 shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
              <p className="text-sm font-medium text-ink mb-3">Weak Concepts to Review</p>
              <div className="flex flex-wrap gap-2">
                {stats.weak_concepts.map((c, i) => (
                  <button key={c}
                    onClick={() => generateForConcept(c)}
                    className="bg-brand-50 text-brand-600 border border-brand-100 text-sm px-3 py-1 rounded-full
                      hover:bg-brand-100 transition-colors animate-pop-in"
                    style={{ animationDelay: `${i * 50}ms` }}
                  >
                    {c} → Practice
                  </button>
                ))}
              </div>

              {generating && <p className="text-sm text-ash mt-3">Generating question…</p>}
              {genError && <p className="text-red-700 text-sm mt-3">{genError}</p>}
              {generated && (
                <div className="mt-4 p-4 bg-parchment rounded-2xl border border-cream space-y-3">
                  <p className="text-sm font-medium text-ink">Practice Question</p>
                  <p className="text-sm text-bark">{generated.question}</p>
                  <div className="grid grid-cols-2 gap-2">
                    {generated.options.map((opt, i) => (
                      <div key={i} className={`text-xs p-2.5 rounded-lg border
                        ${opt.startsWith(generated.correct_answer)
                          ? "bg-green-50 border-green-200 text-green-800"
                          : "bg-ivory border-cream text-bark"}`}>
                        {opt}
                      </div>
                    ))}
                  </div>
                  <p className="text-xs text-ash">{generated.explanation}</p>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
