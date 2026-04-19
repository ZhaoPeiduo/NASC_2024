import { type ReactNode } from "react";
import { NavLink, Navigate } from "react-router-dom";
import { useAuthContext } from "../contexts/AuthContext";

const NAV_ITEMS = [
  { to: "/ask",      label: "Ask" },
  { to: "/quiz",     label: "Quiz" },
  { to: "/history",  label: "History" },
  { to: "/stats",    label: "Stats" },
  { to: "/discover", label: "Discover" },
];

export default function Layout({ children }: { children: ReactNode }) {
  const { user, loading, logout } = useAuthContext();

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center text-ash text-sm">
      Loading…
    </div>
  );
  if (!user) return <Navigate to="/login" replace />;

  return (
    <div className="min-h-screen bg-parchment">
      <nav className="bg-ivory border-b border-cream sticky top-0 z-40 px-6 flex items-center justify-between h-14">
        {/* Logo */}
        <span className="text-base font-semibold tracking-tight select-none text-ink">
          JLPT <span className="text-brand-500">Sensei</span>
        </span>

        {/* Nav links */}
        <div className="flex items-center gap-7">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink key={to} to={to}
              className={({ isActive }) =>
                `text-sm transition-colors pb-0.5 border-b-2 ${
                  isActive
                    ? "text-ink border-brand-500 font-medium"
                    : "text-bark border-transparent hover:text-ink"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
          <button
            onClick={logout}
            className="text-xs text-ash hover:text-bark transition-colors"
          >
            Sign out
          </button>
        </div>
      </nav>
      <main className="max-w-3xl mx-auto px-4 py-8">{children}</main>
    </div>
  );
}
