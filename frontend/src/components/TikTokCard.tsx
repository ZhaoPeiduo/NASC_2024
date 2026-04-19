import { useEffect, useRef, useState } from "react";
import type { TikTokVideoMeta } from "../types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

interface Props {
  video: TikTokVideoMeta;
  autoPlay?: boolean;
  delay?: number;
}

export default function TikTokCard({ video, autoPlay = false, delay = 0 }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(false);
    const token = localStorage.getItem("token");
    let objectUrl = "";

    fetch(`${BASE}/api/v1/tiktok/video-stream/${video.video_id}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
      .then(r => {
        if (!r.ok) throw new Error("stream failed");
        return r.blob();
      })
      .then(blob => {
        objectUrl = URL.createObjectURL(blob);
        if (videoRef.current) {
          videoRef.current.src = objectUrl;
          if (autoPlay) videoRef.current.play().catch(() => {});
        }
        setLoading(false);
      })
      .catch(() => {
        setError(true);
        setLoading(false);
      });

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [video.video_id, autoPlay]);

  return (
    <div
      className="rounded-2xl overflow-hidden border border-cream bg-ivory animate-pop-in"
      style={{ animationDelay: `${delay}ms` }}
    >
      {loading && (
        <div className="w-full aspect-video bg-sand flex items-center justify-center animate-timer-pulse">
          <span className="text-ash text-xs">Loading video…</span>
        </div>
      )}
      {error && (
        <div className="w-full aspect-video bg-sand flex items-center justify-center">
          <span className="text-ash text-xs">Video unavailable</span>
        </div>
      )}
      <video
        ref={videoRef}
        controls
        className={`w-full aspect-video ${loading || error ? "hidden" : ""}`}
        onLoadedData={() => setLoading(false)}
      />
      <div className="px-3 py-2.5 space-y-0.5">
        <p className="text-xs font-medium text-ink leading-snug line-clamp-2">{video.title || "Untitled"}</p>
        <p className="text-xs text-ash">@{video.author}</p>
      </div>
    </div>
  );
}
