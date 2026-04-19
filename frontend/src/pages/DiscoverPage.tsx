import { useState } from "react";
import { api } from "../api/client";
import type { VideoRecommendation } from "../types";
import YouTubeCard from "../components/YouTubeCard";

export default function DiscoverPage() {
  const [video, setVideo] = useState<VideoRecommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hitCount, setHitCount] = useState(0);

  const hitMe = async () => {
    setLoading(true);
    setVideo(null);
    setError("");
    try {
      const v = await api.getRandomSong();
      setVideo(v);
      setHitCount(n => n + 1);
    } catch {
      setError("Couldn't find a video right now. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-serif text-2xl font-medium text-ink">Discover</h1>
        <p className="text-sm text-bark mt-1 leading-relaxed">
          Warm up with a random Japanese song before you study.
        </p>
      </div>

      <div className="flex flex-col items-center gap-6">
        <button
          onClick={hitMe}
          disabled={loading}
          className="bg-brand-500 hover:bg-brand-600 active:scale-95 text-ivory
            text-base font-semibold px-10 py-4 rounded-2xl transition-all
            disabled:opacity-50 shadow-[rgba(0,0,0,0.08)_0px_8px_32px]"
        >
          {loading ? "Finding…" : hitCount === 0 ? "🎵 Hit Me" : "🎵 Another One"}
        </button>

        {error && (
          <p className="text-sm text-red-700 text-center">{error}</p>
        )}

        {video && !loading && (
          <div className="w-full max-w-lg animate-fade-in">
            <YouTubeCard video={video} autoPlay />
          </div>
        )}

        {!video && !loading && hitCount === 0 && (
          <div className="text-center space-y-2 py-8">
            <p className="text-3xl">🎌</p>
            <p className="text-bark text-sm">Hit the button to discover a Japanese song.</p>
            <p className="text-ash text-xs max-w-xs">
              Songs are picked from J-pop, anime, city pop, and traditional music — sometimes with grammar relevance.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
