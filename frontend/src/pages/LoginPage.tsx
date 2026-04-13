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
      {/* ── Left hero panel — Near Black dark section (desktop only) ── */}
      <div className="hidden lg:flex lg:w-[45%] bg-ink flex-col items-center justify-center px-12">
        <div className="max-w-sm w-full space-y-8">
          {/* Brand */}
          <div>
            <h1 className="font-serif text-4xl font-medium text-ivory tracking-tight leading-tight">
              JLPT Sensei
            </h1>
            <p className="text-silver mt-3 text-base leading-relaxed">
              Master the grammar. Pass the exam.
            </p>
          </div>

          {/* Value props */}
          <ul className="space-y-4">
            {VALUE_PROPS.map(item => (
              <li key={item} className="flex items-start gap-3 text-silver text-sm leading-relaxed">
                <span className="mt-0.5 text-brand-500 shrink-0 font-medium">✓</span>
                {item}
              </li>
            ))}
          </ul>

          {/* Tags */}
          <div className="flex gap-2 flex-wrap">
            {["N5 → N1", "AI Tutor", "Free"].map(tag => (
              <span key={tag}
                className="bg-coal text-silver text-xs px-3 py-1 rounded-full border border-coal"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Right form panel — Parchment ── */}
      <div className="flex-1 flex items-center justify-center bg-parchment p-6">
        <div className="w-full max-w-sm">
          {/* Mobile brand header */}
          <div className="lg:hidden text-center mb-8">
            <h1 className="font-serif text-2xl font-medium text-ink">JLPT Sensei</h1>
            <p className="text-bark text-sm mt-1">Master the grammar. Pass the exam.</p>
          </div>

          <div className="bg-ivory rounded-2xl border border-cream shadow-[rgba(0,0,0,0.05)_0px_4px_24px] p-8">
            {/* Mode toggle */}
            <div className="flex gap-2 mb-6">
              {(["login", "register"] as const).map(m => (
                <button key={m} onClick={() => switchMode(m)}
                  className={`flex-1 py-2 rounded-xl text-sm font-medium transition-colors
                    ${mode === m ? "bg-brand-500 text-ivory" : "bg-sand text-charcoal"}`}
                >
                  {m === "login" ? "Sign In" : "Sign Up"}
                </button>
              ))}
            </div>

            <form onSubmit={submit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-bark mb-1">Email</label>
                <input type="email" value={email} onChange={e => setEmail(e.target.value)} required
                  className="w-full border border-cream bg-white rounded-xl px-3 py-2.5 text-sm text-ink
                    focus:outline-none focus:ring-2 focus:ring-[#3898ec] focus:border-[#3898ec] transition-colors"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-bark mb-1">Password</label>
                <input type="password" value={password} onChange={e => setPassword(e.target.value)} required
                  className="w-full border border-cream bg-white rounded-xl px-3 py-2.5 text-sm text-ink
                    focus:outline-none focus:ring-2 focus:ring-[#3898ec] focus:border-[#3898ec] transition-colors"
                />
              </div>
              {error && <p className="text-red-700 text-sm">{error}</p>}
              <button type="submit" disabled={submitting}
                className="w-full bg-brand-500 hover:bg-brand-600 text-ivory py-2.5 rounded-xl
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
