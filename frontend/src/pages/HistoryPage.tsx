import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AttemptResponse } from "../types";
import HistoryItem from "../components/HistoryItem";

export default function HistoryPage() {
  const [attempts, setAttempts] = useState<AttemptResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "wrong">("all");
  const [weakConcepts, setWeakConcepts] = useState<string[]>([]);
  const [conceptsOpen, setConceptsOpen] = useState(true);

  useEffect(() => {
    api.getHistory().then(setAttempts).finally(() => setLoading(false));
    api.getWeakConcepts(10).then(res => setWeakConcepts(res.concepts)).catch(() => {});
  }, []);

  const shown = filter === "wrong" ? attempts.filter(a => !a.user_marked_correct) : attempts;

  return (
    <div className="space-y-5">
      {/* Weak concepts panel */}
      {weakConcepts.length > 0 && (
        <div className="bg-white border border-stone-200 rounded-2xl overflow-hidden animate-fade-in">
          <button
            onClick={() => setConceptsOpen(v => !v)}
            className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-stone-50 transition-colors"
          >
            <span className="text-sm font-semibold text-stone-700">
              Top {weakConcepts.length} Weak Points
            </span>
            <span className="text-stone-400 text-xs">{conceptsOpen ? "▲" : "▼"}</span>
          </button>
          {conceptsOpen && (
            <div className="px-4 pb-3 flex flex-wrap gap-1.5 animate-fade-in">
              {weakConcepts.map((c, i) => (
                <span
                  key={c}
                  className="bg-red-50 text-red-600 text-xs px-2 py-1 rounded-full border border-red-100 animate-pop-in"
                  style={{ animationDelay: `${i * 40}ms` }}
                >
                  {i + 1}. {c}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Filter + list */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-xl font-bold text-stone-800">History</h1>
            <p className="text-sm text-stone-500 mt-0.5">
              Every question you've answered — expand any item to review options and explanations.
            </p>
          </div>
          <div className="flex gap-2">
            {(["all", "wrong"] as const).map(f => (
              <button key={f} onClick={() => setFilter(f)}
                className={`text-sm px-3 py-1.5 rounded-lg font-medium transition-colors
                  ${filter === f ? "bg-brand-500 text-white" : "bg-white border border-stone-200 text-stone-600"}`}
              >
                {f === "all" ? "All" : "Wrong only"}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <p className="text-stone-400 text-center py-12">Loading…</p>
        ) : shown.length === 0 ? (
          <div className="text-center py-16 space-y-3">
            <p className="text-2xl">📖</p>
            <p className="text-stone-500 font-medium">No attempts yet</p>
            <p className="text-stone-400 text-sm">
              Answer your first question to start building your review log.
            </p>
            <a href="/ask"
              className="inline-block mt-1 text-sm text-brand-600 font-semibold hover:underline"
            >
              Try your first question →
            </a>
          </div>
        ) : (
          <div className="space-y-3">
            {shown.map((a, i) => <HistoryItem key={a.id} attempt={a} index={i} />)}
          </div>
        )}
      </div>
    </div>
  );
}
