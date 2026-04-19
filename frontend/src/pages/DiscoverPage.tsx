import { useState } from "react";
import { api } from "../api/client";
import type { VideoRecommendation, TikTokVideoMeta } from "../types";
import YouTubeCard from "../components/YouTubeCard";
import TikTokCard from "../components/TikTokCard";

type Source = "youtube" | "tiktok";

const YOUTUBE_TOPICS = ["Random", "J-Pop", "Anime", "Grammar"] as const;
const TIKTOK_TOPICS  = ["Trending", "日本語", "Anime", "Grammar"] as const;

const YOUTUBE_TOPIC_MAP: Record<string, string | undefined> = {
  "Random": undefined,
  "J-Pop": "J-Pop",
  "Anime": "anime opening",
  "Grammar": "JLPT grammar",
};

const TIKTOK_TOPIC_MAP: Record<string, string> = {
  "Trending": "trending",
  "日本語": "日本語",
  "Anime": "anime",
  "Grammar": "grammar",
};

export default function DiscoverPage() {
  const [source, setSource] = useState<Source>("youtube");
  const [ytTopic, setYtTopic]   = useState<string>("Random");
  const [ttTopic, setTtTopic]   = useState<string>("Trending");
  const [ytVideo, setYtVideo]   = useState<VideoRecommendation | null>(null);
  const [ttVideo, setTtVideo]   = useState<TikTokVideoMeta | null>(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");
  const [hitCount, setHitCount] = useState(0);

  const hitMe = async () => {
    setLoading(true);
    setYtVideo(null);
    setTtVideo(null);
    setError("");
    try {
      if (source === "youtube") {
        const concept = YOUTUBE_TOPIC_MAP[ytTopic];
        const v = await api.getRandomSong(concept);
        setYtVideo(v);
      } else {
        const topic = TIKTOK_TOPIC_MAP[ttTopic];
        const v = await api.getRandomTikTok(topic);
        setTtVideo(v);
      }
      setHitCount(n => n + 1);
    } catch {
      setError("Couldn't find a video right now. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const activeVideo = source === "youtube" ? ytVideo : ttVideo;
  const activeTopic = source === "youtube" ? ytTopic : ttTopic;
  const topics = source === "youtube" ? YOUTUBE_TOPICS : TIKTOK_TOPICS;
  const setTopic = source === "youtube" ? setYtTopic : setTtTopic;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-serif text-2xl font-medium text-ink">Discover</h1>
        <p className="text-sm text-bark mt-1 leading-relaxed">
          Warm up with a random Japanese video before you study.
        </p>
      </div>

      {/* Source toggle */}
      <div className="flex bg-sand rounded-xl p-1 gap-1 w-fit">
        {(["youtube", "tiktok"] as Source[]).map(s => (
          <button
            key={s}
            onClick={() => { setSource(s); setError(""); }}
            className={`px-5 py-1.5 rounded-lg text-sm font-medium transition-all ${
              source === s
                ? "bg-ivory text-ink shadow-sm border border-cream"
                : "text-bark hover:text-ink"
            }`}
          >
            {s === "youtube" ? "YouTube" : "TikTok"}
          </button>
        ))}
      </div>

      {/* Topic chips */}
      <div className="flex flex-wrap gap-2">
        {topics.map(t => (
          <button
            key={t}
            onClick={() => setTopic(t)}
            className={`px-3 py-1 rounded-full text-xs font-medium border transition-all ${
              activeTopic === t
                ? "bg-brand-500 text-ivory border-brand-500"
                : "bg-ivory text-bark border-cream hover:border-brand-500"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Hit Me button */}
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

        {error && <p className="text-sm text-red-700 text-center">{error}</p>}

        {ytVideo && source === "youtube" && !loading && (
          <div className="w-full max-w-lg animate-fade-in">
            <YouTubeCard video={ytVideo} autoPlay />
          </div>
        )}

        {ttVideo && source === "tiktok" && !loading && (
          <div className="w-full max-w-lg animate-fade-in">
            <TikTokCard video={ttVideo} autoPlay />
          </div>
        )}

        {!activeVideo && !loading && hitCount === 0 && (
          <div className="text-center space-y-2 py-8">
            <p className="text-3xl">{source === "youtube" ? "🎌" : "🎵"}</p>
            <p className="text-bark text-sm">
              Hit the button to discover a Japanese {source === "youtube" ? "song" : "TikTok"}.
            </p>
            <p className="text-ash text-xs max-w-xs">
              {source === "youtube"
                ? "Songs are picked from J-pop, anime, city pop, and traditional music."
                : "Videos are picked from trending and topic-based TikTok searches."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
