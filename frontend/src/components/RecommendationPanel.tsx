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
