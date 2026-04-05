interface Platform {
  name: string;
  bg: string;
  text: string;
  buildUrl: (query: string) => string;
}

const PLATFORMS: Platform[] = [
  {
    name: "TikTok",
    bg: "bg-black",
    text: "text-white",
    buildUrl: (q) => `https://www.tiktok.com/search?q=${encodeURIComponent(q)}`,
  },
  {
    name: "Lemon8",
    bg: "bg-yellow-400",
    text: "text-black",
    buildUrl: (q) =>
      `https://www.lemon8-app.com/search/result?keyword=${encodeURIComponent(q)}`,
  },
  {
    name: "Spotify",
    bg: "bg-green-500",
    text: "text-white",
    buildUrl: (q) => `https://open.spotify.com/search/${encodeURIComponent(q)}/podcasts`,
  },
];

export default function RecommendationPanel({ concepts }: { concepts: string[] }) {
  if (concepts.length === 0) return null;

  const query = `JLPT ${concepts[0]} grammar Japanese`;

  return (
    <div className="mt-6">
      <p className="text-sm font-semibold text-slate-700 mb-3">
        Find content for:{" "}
        <span className="text-brand-600">{concepts[0]}</span>
      </p>
      <div className="flex flex-wrap gap-3">
        {PLATFORMS.map(({ name, bg, text, buildUrl }) => (
          <a
            key={name}
            href={buildUrl(query)}
            target="_blank"
            rel="noreferrer"
            className={`inline-flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium transition-opacity hover:opacity-80 ${bg} ${text}`}
          >
            {name === "TikTok" ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M19.59 6.69a4.83 4.83 0 0 1-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 0 1-2.88 2.5 2.89 2.89 0 0 1-2.89-2.89 2.89 2.89 0 0 1 2.89-2.89c.28 0 .54.04.79.1V9.01a6.32 6.32 0 0 0-.79-.05 6.34 6.34 0 0 0-6.34 6.34 6.34 6.34 0 0 0 6.34 6.34 6.34 6.34 0 0 0 6.33-6.34V8.69a8.18 8.18 0 0 0 4.78 1.52V6.74a4.85 4.85 0 0 1-1.01-.05z"/>
              </svg>
            ) : name === "Spotify" ? (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z"/>
              </svg>
            ) : (
              <span>🍋</span>
            )}
            Search {name}
          </a>
        ))}
      </div>
      {concepts.length > 1 && (
        <p className="text-xs text-slate-400 mt-2">
          Also weak in: {concepts.slice(1).join(", ")}
        </p>
      )}
    </div>
  );
}
