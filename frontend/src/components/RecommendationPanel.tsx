import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { MediaRecommendResponse } from "../types";

const PLATFORMS = [
  { name: "TikTok", bg: "bg-black", text: "text-white",
    url: (q: string) => `https://www.tiktok.com/search?q=${encodeURIComponent(q)}` },
  { name: "Lemon8", bg: "bg-yellow-400", text: "text-black",
    url: (q: string) => `https://www.lemon8-app.com/search/result?keyword=${encodeURIComponent(q)}` },
  { name: "Spotify", bg: "bg-green-500", text: "text-white",
    url: (q: string) => `https://open.spotify.com/search/${encodeURIComponent(q)}/podcasts` },
];

interface MediaCardProps {
  label: string;
  query: string;
  sub?: string;
  delay: number;
}

function MediaCard({ label, query, sub, delay }: MediaCardProps) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-3 space-y-2 animate-slide-in hover:shadow-sm transition-shadow"
      style={{ animationDelay: `${delay}ms` }}>
      <p className="text-xs font-semibold text-slate-700 leading-snug">{label}</p>
      {sub && <p className="text-xs text-slate-400">{sub}</p>}
      <div className="flex flex-wrap gap-1.5">
        {PLATFORMS.map(p => (
          <a key={p.name} href={p.url(query)} target="_blank" rel="noreferrer"
            className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full
              font-medium transition-opacity hover:opacity-75 ${p.bg} ${p.text}`}>
            {p.name}
          </a>
        ))}
      </div>
    </div>
  );
}

export default function RecommendationPanel({ concepts }: { concepts: string[] }) {
  const [data, setData] = useState<MediaRecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (concepts.length === 0) return;
    setLoading(true); setData(null);
    api.getMediaRecommendations(concepts[0])
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [concepts.join(",")]);

  if (concepts.length === 0) return null;

  return (
    <div className="mt-4 space-y-3 animate-fade-in">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
        Study resources for: <span className="text-brand-600 normal-case">{concepts[0]}</span>
      </p>

      {loading && (
        <div className="space-y-2">
          {[0, 1, 2].map(i => (
            <div key={i} className="h-16 bg-slate-100 rounded-xl animate-timer-pulse"
              style={{ animationDelay: `${i * 100}ms` }} />
          ))}
        </div>
      )}

      {data && !loading && (
        <>
          {data.songs.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-slate-400">🎵 Songs</p>
              {data.songs.map((s, i) => (
                <MediaCard key={i} label={s.title} sub={s.artist}
                  query={`${s.title} ${s.artist} Japanese`} delay={i * 60} />
              ))}
            </div>
          )}

          {data.anime.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-slate-400">🎌 Anime</p>
              {data.anime.map((a, i) => (
                <MediaCard key={i} label={a.title} sub={a.scene}
                  query={`${a.title} ${a.scene}`} delay={i * 60} />
              ))}
            </div>
          )}

          {data.articles.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-slate-400">📖 Articles</p>
              {data.articles.map((ar, i) => (
                <MediaCard key={i} label={ar.title} query={ar.keywords} delay={i * 60} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
