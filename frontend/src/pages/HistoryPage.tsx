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
    <div className="space-y-6">
      {/* Weak concepts panel */}
      {weakConcepts.length > 0 && (
        <div className="bg-ivory border border-cream rounded-2xl overflow-hidden animate-fade-in shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
          <button
            onClick={() => setConceptsOpen(v => !v)}
            className="w-full flex items-center justify-between px-5 py-3.5 text-left hover:bg-parchment transition-colors"
          >
            <span className="text-sm font-medium text-ink">
              Top {weakConcepts.length} Weak Points
            </span>
            <span className="text-ash text-xs">{conceptsOpen ? "▲" : "▼"}</span>
          </button>
          {conceptsOpen && (
            <div className="px-5 pb-4 flex flex-wrap gap-1.5 animate-fade-in border-t border-cream">
              {weakConcepts.map((c, i) => (
                <span
                  key={c}
                  className="bg-brand-50 text-brand-600 text-xs px-2.5 py-1 rounded-full border border-brand-100 animate-pop-in"
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
        <div className="flex items-center justify-between mb-5">
          <div>
            <h1 className="font-serif text-2xl font-medium text-ink">History</h1>
            <p className="text-sm text-bark mt-0.5">
              Every question you've answered — expand any item to review options and explanations.
            </p>
          </div>
          <div className="flex gap-2">
            {(["all", "wrong"] as const).map(f => (
              <button key={f} onClick={() => setFilter(f)}
                className={`text-sm px-3 py-1.5 rounded-lg font-medium transition-colors
                  ${filter === f ? "bg-brand-500 text-ivory" : "bg-sand border border-cream text-charcoal hover:bg-sand/80"}`}
              >
                {f === "all" ? "All" : "Wrong only"}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <p className="text-ash text-center py-12">Loading…</p>
        ) : shown.length === 0 ? (
          <div className="text-center py-16 space-y-3 bg-ivory rounded-2xl border border-cream">
            <p className="text-2xl">📖</p>
            <p className="text-bark font-medium">No attempts yet</p>
            <p className="text-ash text-sm">
              Answer your first question to start building your review log.
            </p>
            <a href="/ask"
              className="inline-block mt-1 text-sm text-brand-500 font-semibold hover:underline"
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
