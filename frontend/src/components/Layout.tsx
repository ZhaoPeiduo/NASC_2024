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
    <div className="min-h-screen flex items-center justify-center text-slate-400 text-sm">
      Loading…
    </div>
  );
  if (!user) return <Navigate to="/login" replace />;

  return (
    <div className="min-h-screen bg-slate-50">
      <nav className="bg-white border-b border-slate-200 px-4 py-2 flex items-center justify-between">
        <span className="font-bold text-slate-800 text-sm tracking-tight">JLPT Sensei</span>
        <div className="flex items-center gap-5">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink key={to} to={to}
              className={({ isActive }) =>
                `text-xs font-semibold transition-colors ${isActive ? "text-brand-500" : "text-slate-500 hover:text-slate-800"}`
              }
            >
              {label}
            </NavLink>
          ))}
          <button onClick={logout} className="text-xs text-slate-400 hover:text-slate-600 transition-colors">
            Sign out
          </button>
        </div>
      </nav>
      <main className="max-w-3xl mx-auto px-4 py-5">{children}</main>
    </div>
  );
}
