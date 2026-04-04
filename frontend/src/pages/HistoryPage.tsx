import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AttemptResponse } from "../types";
import HistoryItem from "../components/HistoryItem";

export default function HistoryPage() {
  const [attempts, setAttempts] = useState<AttemptResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "wrong">("all");

  useEffect(() => {
    api.getHistory().then(setAttempts).finally(() => setLoading(false));
  }, []);

  const shown = filter === "wrong" ? attempts.filter(a => !a.user_marked_correct) : attempts;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-xl font-bold text-slate-800">History</h1>
        <div className="flex gap-2">
          {(["all", "wrong"] as const).map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`text-sm px-3 py-1.5 rounded-lg font-medium transition-colors
                ${filter === f ? "bg-brand-500 text-white" : "bg-white border border-slate-200 text-slate-600"}`}
            >
              {f === "all" ? "All" : "Wrong only"}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <p className="text-slate-400 text-center py-12">Loading…</p>
      ) : shown.length === 0 ? (
        <p className="text-slate-400 text-center py-12">No attempts yet. Start practicing!</p>
      ) : (
        <div className="space-y-3">
          {shown.map(a => <HistoryItem key={a.id} attempt={a} />)}
        </div>
      )}
    </div>
  );
}
