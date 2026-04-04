import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { StatsResponse } from "../types";
import { BarStat } from "../components/StatsChart";

export default function StatsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);

  useEffect(() => { api.getStats().then(setStats); }, []);

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
              <span key={c} className="bg-red-50 text-red-700 text-sm px-3 py-1 rounded-full">{c}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
