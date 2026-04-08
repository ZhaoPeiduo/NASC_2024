import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthContext } from "../contexts/AuthContext";

const VALUE_PROPS = [
  "Understand every wrong answer, not just the correct one",
  "Track which grammar patterns trip you up most",
  "Generate new practice questions on your weak spots",
];

export default function LoginPage() {
  const { login, register } = useAuthContext();
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const switchMode = (m: "login" | "register") => {
    setMode(m); setError(""); setEmail(""); setPassword("");
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(""); setSubmitting(true);
    try {
      if (mode === "login") await login(email, password);
      else await register(email, password);
      navigate("/ask");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* ── Left hero panel (desktop only) ── */}
      <div className="hidden lg:flex lg:w-[45%] bg-gradient-to-br from-brand-700 via-brand-600 to-brand-500 flex-col items-center justify-center px-12">
        <div className="max-w-sm w-full space-y-7">
          {/* Brand */}
          <div>
            <h1 className="text-4xl font-bold text-white tracking-tight">JLPT Sensei</h1>
            <p className="text-brand-100 mt-2 text-lg font-medium">
              Master the grammar. Pass the exam.
            </p>
          </div>

          {/* Value props */}
          <ul className="space-y-3">
            {VALUE_PROPS.map(item => (
              <li key={item} className="flex items-start gap-3 text-brand-100 text-sm leading-relaxed">
                <span className="mt-0.5 font-bold text-brand-200 shrink-0">✓</span>
                {item}
              </li>
            ))}
          </ul>

          {/* Tags */}
          <div className="flex gap-2 flex-wrap">
            {["N5 → N1", "AI Tutor", "Free"].map(tag => (
              <span key={tag} className="bg-white/10 text-white/90 text-xs px-3 py-1 rounded-full">
                {tag}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Right form panel ── */}
      <div className="flex-1 flex items-center justify-center bg-[#f8f7f4] p-6">
        <div className="w-full max-w-sm">
          {/* Mobile brand header (hidden on desktop where left panel shows) */}
          <div className="lg:hidden text-center mb-8">
            <h1 className="text-2xl font-bold text-brand-600">JLPT Sensei</h1>
            <p className="text-stone-500 text-sm mt-1">Master the grammar. Pass the exam.</p>
          </div>

          <div className="bg-white rounded-2xl shadow-md p-8">
            {/* Mode toggle */}
            <div className="flex gap-2 mb-6">
              {(["login", "register"] as const).map(m => (
                <button key={m} onClick={() => switchMode(m)}
                  className={`flex-1 py-2 rounded-xl text-sm font-medium transition-colors
                    ${mode === m ? "bg-brand-500 text-white" : "bg-stone-100 text-stone-600"}`}
                >
                  {m === "login" ? "Sign In" : "Sign Up"}
                </button>
              ))}
            </div>

            <form onSubmit={submit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-stone-700 mb-1">Email</label>
                <input type="email" value={email} onChange={e => setEmail(e.target.value)} required
                  className="w-full border border-stone-200 rounded-xl px-3 py-2 text-sm
                    focus:outline-none focus:ring-2 focus:ring-brand-500 transition-colors"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-stone-700 mb-1">Password</label>
                <input type="password" value={password} onChange={e => setPassword(e.target.value)} required
                  className="w-full border border-stone-200 rounded-xl px-3 py-2 text-sm
                    focus:outline-none focus:ring-2 focus:ring-brand-500 transition-colors"
                />
              </div>
              {error && <p className="text-red-600 text-sm">{error}</p>}
              <button type="submit" disabled={submitting}
                className="w-full bg-brand-500 hover:bg-brand-600 text-white py-2.5 rounded-xl
                  font-medium transition-colors disabled:opacity-50"
              >
                {submitting ? "Please wait…" : mode === "login" ? "Sign In" : "Create Account"}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
