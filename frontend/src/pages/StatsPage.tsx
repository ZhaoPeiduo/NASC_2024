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

  if (!stats) return <p className="text-slate-400 text-center py-12">Loading…</p>;

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold text-slate-800">Your Progress</h1>

      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Total Attempts", value: stats.total_attempts },
          { label: "Accuracy", value: `${Math.round(stats.correct_rate * 100)}%` },
          { label: "Study Days", value: stats.study_days },
        ].map(({ label, value }) => (
          <div key={label} className="bg-white border border-slate-200 rounded-xl p-4 text-center">
            <p className="text-3xl font-bold text-slate-800">{value}</p>
            <p className="text-xs text-slate-400 mt-1">{label}</p>
          </div>
        ))}
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6">
        <BarStat label="Accuracy" value={stats.correct_rate} color="bg-brand-500" />
      </div>

      {stats.weak_concepts.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <p className="text-sm font-semibold text-slate-700 mb-3">Weak Concepts to Review</p>
          <div className="flex flex-wrap gap-2">
            {stats.weak_concepts.map(c => (
              <button key={c}
                onClick={() => generateForConcept(c)}
                className="bg-red-50 text-red-700 text-sm px-3 py-1 rounded-full hover:bg-red-100 transition-colors"
              >
                {c} → Practice
              </button>
            ))}
          </div>

          {generating && <p className="text-sm text-slate-400 mt-3">Generating question…</p>}
          {genError && <p className="text-red-600 text-sm mt-3">{genError}</p>}
          {generated && (
            <div className="mt-4 p-4 bg-slate-50 rounded-xl space-y-3">
              <p className="text-sm font-semibold text-slate-700">Practice Question</p>
              <p className="text-sm text-slate-800">{generated.question}</p>
              <div className="grid grid-cols-2 gap-2">
                {generated.options.map((opt, i) => (
                  <div key={i} className={`text-xs p-2 rounded-lg border
                    ${opt.startsWith(generated.correct_answer)
                      ? "bg-green-50 border-green-200 text-green-800"
                      : "bg-white border-slate-200 text-slate-600"}`}>
                    {opt}
                  </div>
                ))}
              </div>
              <p className="text-xs text-slate-500">{generated.explanation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
