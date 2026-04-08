import { type ReactNode } from "react";
import { NavLink, Navigate } from "react-router-dom";
import { useAuthContext } from "../contexts/AuthContext";

const NAV_ITEMS = [
  { to: "/ask",      label: "Ask" },
  { to: "/quiz",     label: "Quiz" },
  { to: "/history",  label: "History" },
  { to: "/stats",    label: "Stats" },
];

export default function Layout({ children }: { children: ReactNode }) {
  const { user, loading, logout } = useAuthContext();

  if (loading) return (
    <div className="min-h-screen flex items-center justify-center text-stone-400 text-sm">
      Loading…
    </div>
  );
  if (!user) return <Navigate to="/login" replace />;

  return (
    <div className="min-h-screen bg-[#f8f7f4]">
      <nav className="bg-white shadow-sm sticky top-0 z-40 px-4 flex items-center justify-between h-14">
        {/* Logo */}
        <span className="text-base font-bold tracking-tight select-none">
          <span className="text-brand-600">JLPT</span>
          <span className="text-stone-800"> Sensei</span>
        </span>

        {/* Nav links */}
        <div className="flex items-center gap-6">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink key={to} to={to}
              className={({ isActive }) =>
                `text-xs font-semibold transition-colors pb-0.5 border-b-2 ${
                  isActive
                    ? "text-brand-600 border-brand-500"
                    : "text-stone-400 border-transparent hover:text-stone-700"
                }`
              }
            >
              {label}
            </NavLink>
          ))}
          <button
            onClick={logout}
            className="text-xs text-stone-400 hover:text-stone-600 transition-colors"
          >
            Sign out
          </button>
        </div>
      </nav>
      <main className="max-w-3xl mx-auto px-4 py-6">{children}</main>
    </div>
  );
}
