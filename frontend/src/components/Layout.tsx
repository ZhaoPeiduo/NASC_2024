import { type ReactNode } from "react";
import { NavLink, Navigate } from "react-router-dom";
import { useAuthContext } from "../contexts/AuthContext";

const NAV_ITEMS = [
  { to: "/practice", label: "Practice" },
  { to: "/history", label: "History" },
  { to: "/stats", label: "Stats" },
];

export default function Layout({ children }: { children: ReactNode }) {
  const { user, loading, logout } = useAuthContext();

  if (loading) return <div className="min-h-screen flex items-center justify-center text-slate-400">Loading…</div>;
  if (!user) return <Navigate to="/login" replace />;

  return (
    <div className="min-h-screen bg-slate-50">
      <nav className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between">
        <span className="font-bold text-slate-800">JLPT Sensei</span>
        <div className="flex items-center gap-6">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink key={to} to={to}
              className={({ isActive }) =>
                `text-sm font-medium transition-colors ${isActive ? "text-brand-500" : "text-slate-600 hover:text-slate-900"}`
              }
            >
              {label}
            </NavLink>
          ))}
          <button onClick={logout} className="text-sm text-slate-400 hover:text-slate-600">Sign out</button>
        </div>
      </nav>
      <main className="max-w-4xl mx-auto px-6 py-8">{children}</main>
    </div>
  );
}
