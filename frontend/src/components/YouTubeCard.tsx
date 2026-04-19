import { useState } from "react";
import type { VideoRecommendation } from "../types";

interface Props {
  video: VideoRecommendation;
  autoPlay?: boolean;
  delay?: number;
}

export default function YouTubeCard({ video, autoPlay = false, delay = 0 }: Props) {
  const [playing, setPlaying] = useState(autoPlay);

  return (
    <div
      className="rounded-2xl overflow-hidden border border-cream bg-ivory animate-pop-in"
      style={{ animationDelay: `${delay}ms` }}
    >
      {playing ? (
        <iframe
          src={`https://www.youtube.com/embed/${video.video_id}?autoplay=1&rel=0`}
          className="w-full aspect-video"
          allow="autoplay; encrypted-media; picture-in-picture"
          allowFullScreen
          title={video.title}
        />
      ) : (
        <button
          onClick={() => setPlaying(true)}
          className="w-full relative group focus:outline-none"
          aria-label={`Play ${video.title}`}
        >
          {video.thumbnail_url ? (
            <img
              src={video.thumbnail_url}
              alt={video.title}
              className="w-full aspect-video object-cover"
            />
          ) : (
            <div className="w-full aspect-video bg-sand flex items-center justify-center">
              <span className="text-ash text-sm">No thumbnail</span>
            </div>
          )}
          {/* Play overlay */}
          <div className="absolute inset-0 flex items-center justify-center
            bg-ink/10 group-hover:bg-ink/20 transition-colors">
            <span className="w-12 h-12 rounded-full bg-red-600 flex items-center justify-center
              shadow-lg group-hover:scale-110 transition-transform">
              <svg viewBox="0 0 24 24" fill="white" className="w-5 h-5 ml-0.5">
                <path d="M8 5v14l11-7z" />
              </svg>
            </span>
          </div>
        </button>
      )}

      <div className="px-3 py-2.5 space-y-0.5">
        <p className="text-xs font-medium text-ink leading-snug line-clamp-2">{video.title}</p>
        <p className="text-xs text-ash">{video.channel_title}</p>
      </div>
    </div>
  );
}
