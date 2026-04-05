import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { VideoRecommendation } from "../types";

export default function RecommendationPanel({ concepts }: { concepts: string[] }) {
  const [videos, setVideos] = useState<VideoRecommendation[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (concepts.length === 0) return;
    setLoading(true);
    api.getRecommendations(concepts).then(setVideos).finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [concepts.join(",")]);

  if (concepts.length === 0 || (videos.length === 0 && !loading)) return null;

  return (
    <div className="mt-6">
      <p className="text-sm font-semibold text-slate-700 mb-3">Related videos for: {concepts[0]}</p>
      {loading ? (
        <p className="text-sm text-slate-400">Finding videos…</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {videos.map(v => (
            <a
              key={v.video_id}
              href={`https://www.youtube.com/watch?v=${v.video_id}`}
              target="_blank" rel="noreferrer"
              className="bg-white border border-slate-200 rounded-xl overflow-hidden hover:shadow-md transition-shadow"
            >
              <img src={v.thumbnail_url} alt={v.title}
                className="w-full h-24 object-cover" />
              <div className="p-3">
                <p className="text-xs font-medium text-slate-800 line-clamp-2">{v.title}</p>
                <p className="text-xs text-slate-400 mt-1">{v.channel_title}</p>
              </div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
