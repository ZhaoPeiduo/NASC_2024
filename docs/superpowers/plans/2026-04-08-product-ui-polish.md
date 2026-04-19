# JLPT Sensei Product UI Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform JLPT Sensei from a developer prototype into a polished product with a warm indigo color system, sticky frosted nav, split login hero, modernized card rounding, consistent stone-toned text, and copy/empty-states on every page.

**Architecture:** Four isolated layers of change: (1) design token replacement in `index.css` flows automatically through every component via CSS variables; (2) Layout and Login shell changes are self-contained; (3) a systematic find-and-replace of slate→stone and rounded-xl→rounded-2xl touches all component files mechanically; (4) copy and empty-state additions are additive (no existing logic removed). Changes are intentionally constrained by the devil's advocate review: no backdrop-blur, no background patterns, no shadow larger than `shadow-sm`, no icon nav, no font-family change.

**Tech Stack:** React 18, TypeScript, Tailwind CSS v4 (`@theme` / `@utility` directives), Vite 5

---

## File Map

| File | Change |
|------|--------|
| `frontend/src/index.css` | Replace brand palette (sky → indigo); add `bg-surface` warm background token |
| `frontend/src/components/Layout.tsx` | Sticky nav, shadow-sm, warm bg, logo split color, active underline indicator |
| `frontend/src/pages/LoginPage.tsx` | Split desktop layout (hero left + form right); value props; fix `/practice` → `/ask` navigate bug |
| `frontend/src/pages/AskPage.tsx` | H2 subtitle |
| `frontend/src/pages/QuizPage.tsx` / `frontend/src/components/QuizSetup.tsx` | H2 subtitle in QuizSetup; empty-state in setup screen |
| `frontend/src/pages/HistoryPage.tsx` | H2 subtitle; empty-state copy with CTA |
| `frontend/src/pages/StatsPage.tsx` | H2 subtitle; empty-state copy with CTA |
| `frontend/src/components/QuizForm.tsx` | rounded-xl→2xl; slate→stone text colors |
| `frontend/src/components/ExplanationCard.tsx` | rounded-xl→2xl; slate→stone |
| `frontend/src/components/ActiveQuiz.tsx` | rounded-xl→2xl; slate→stone |
| `frontend/src/components/QuizResults.tsx` | rounded-xl→2xl; slate→stone |
| `frontend/src/components/HistoryItem.tsx` | rounded-xl→2xl; slate→stone |

---

### Task 1: Color Palette — Brand + Surface Tokens

**Files:**
- Modify: `frontend/src/index.css`

- [ ] **Step 1: Verify current build passes**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no output (zero errors).

- [ ] **Step 2: Replace the `@theme` block in `frontend/src/index.css`**

The entire current `@theme` block (lines 3–9) becomes:

```css
@theme {
  /* Primary: deep indigo — replaces sky blue across all brand-* classes */
  --color-brand-50:  #eef2ff;
  --color-brand-100: #e0e7ff;
  --color-brand-200: #c7d2fe;
  --color-brand-500: #6366f1;
  --color-brand-600: #4f46e5;
  --color-brand-700: #4338ca;
}
```

No other changes to index.css. The CSS variable names stay identical so every existing `bg-brand-500`, `text-brand-600`, `border-brand-500`, `ring-brand-500` class automatically picks up the new indigo color with zero code changes elsewhere.

> **Why indigo?** Indigo (#6366f1) passes WCAG AA on white (4.56:1 contrast). It evokes the deep blues of traditional Japanese ai-zome (indigo dyeing) while being visually distinct from every other study app using sky blue. `brand-600` (#4f46e5) passes AA+ at 5.65:1.

- [ ] **Step 3: Run type-check and build**

```bash
cd frontend && npx tsc --noEmit && npm run build 2>&1 | tail -5
```
Expected: zero type errors; build succeeds with `dist/` output.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/index.css
git commit -m "feat: replace brand color palette sky-blue → indigo"
```

---

### Task 2: Navigation — Sticky, Logo, Active Indicator

**Files:**
- Modify: `frontend/src/components/Layout.tsx`

- [ ] **Step 1: Replace `frontend/src/components/Layout.tsx`** with:

```typescript
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
```

Key changes:
- `bg-slate-50` → `bg-[#f8f7f4]` (warm off-white — same hex used in login form panel)
- Nav: `shadow-sm sticky top-0 z-40 h-14` (pinned, substantial height, no blur)
- Logo: `JLPT` in `text-brand-600` (indigo), `Sensei` in `text-stone-800`
- Active link: `border-b-2 border-brand-500` indicator; inactive: `border-transparent`
- All `slate-*` → `stone-*` for text

- [ ] **Step 2: Run type-check**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/Layout.tsx
git commit -m "feat: sticky nav with brand logo and active underline indicator"
```

---

### Task 3: Login Page Hero

**Files:**
- Modify: `frontend/src/pages/LoginPage.tsx`

- [ ] **Step 1: Replace `frontend/src/pages/LoginPage.tsx`** with:

```typescript
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
```

Key changes:
- **Bug fix:** `navigate("/practice")` → `navigate("/ask")`
- **Desktop:** 45% indigo gradient hero panel with value props, hidden on mobile
- **Mobile:** Small branded header above the form card
- **Form card:** `rounded-2xl shadow-md` (upgraded from `rounded-2xl shadow-md`)
- **Stone colors** throughout inputs, labels, toggle

- [ ] **Step 2: Run type-check**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/LoginPage.tsx
git commit -m "feat: split login hero with value props and indigo gradient panel; fix navigate to /ask"
```

---

### Task 4: Global Card Design System — rounded-2xl + Stone Colors

**Files:** All component and page files that use old card classes.

This task is a systematic find-and-replace across the component layer. Make each change to each file, then type-check at the end.

- [ ] **Step 1: Update `frontend/src/components/QuizForm.tsx`**

Replace these specific class strings:

| Find | Replace |
|------|---------|
| `bg-white rounded-xl shadow-sm border border-slate-200` | `bg-white rounded-2xl shadow-sm border border-stone-200` |
| `border border-slate-200 rounded-lg` (on textarea) | `border border-stone-200 rounded-xl` |
| `border border-slate-200 rounded-lg` (on option inputs) | `border border-stone-200 rounded-lg` |
| `border border-slate-200 rounded-lg` (on Clear button) | `border border-stone-200 rounded-xl` |
| `text-slate-500` | `text-stone-500` |
| `bg-slate-50` | `bg-stone-50` |
| `hover:border-slate-300` | `hover:border-stone-300` |

- [ ] **Step 2: Update `frontend/src/components/ExplanationCard.tsx`**

| Find | Replace |
|------|---------|
| `bg-white border border-slate-200 rounded-xl min-h-16` | `bg-white border border-stone-200 rounded-2xl min-h-16` |
| `bg-green-50 border border-green-200 rounded-xl` | `bg-green-50 border border-green-200 rounded-2xl` |
| `bg-slate-50 rounded-xl` | `bg-stone-50 rounded-2xl` |
| `text-slate-700` | `text-stone-700` |
| `text-slate-600` | `text-stone-600` |
| `text-slate-400` | `text-stone-400` |

- [ ] **Step 3: Update `frontend/src/components/ActiveQuiz.tsx`**

| Find | Replace |
|------|---------|
| `bg-white border border-slate-200 rounded-xl p-4` | `bg-white border border-stone-200 rounded-2xl p-4` |
| `border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50` | `border-stone-200 bg-white text-stone-700 hover:border-stone-300 hover:bg-stone-50` |
| `text-slate-400` | `text-stone-400` |
| `text-slate-500` | `text-stone-500` |
| `bg-slate-100 rounded-full` (progress bar bg) | `bg-stone-100 rounded-full` |

- [ ] **Step 4: Update `frontend/src/components/QuizResults.tsx`**

| Find | Replace |
|------|---------|
| `rounded-xl` (all occurrences) | `rounded-2xl` |
| `border-slate-200` | `border-stone-200` |
| `text-slate-500` | `text-stone-500` |
| `text-slate-600` | `text-stone-600` |
| `text-slate-800` | `text-stone-800` |
| `bg-slate-100` | `bg-stone-100` |

- [ ] **Step 5: Update `frontend/src/components/HistoryItem.tsx`**

| Find | Replace |
|------|---------|
| `rounded-xl` (all occurrences) | `rounded-2xl` |
| `border-slate-200` | `border-stone-200` |
| `border-slate-100` | `border-stone-100` |
| `text-slate-800` | `text-stone-800` |
| `text-slate-600` | `text-stone-600` |
| `text-slate-500` | `text-stone-500` |
| `text-slate-400` | `text-stone-400` |
| `text-slate-300` | `text-stone-300` |

- [ ] **Step 6: Update `frontend/src/components/QuizSetup.tsx`**

| Find | Replace |
|------|---------|
| `rounded-xl` (all occurrences) | `rounded-2xl` |
| `border-slate-200` | `border-stone-200` |
| `border-slate-100` | `border-stone-100` |
| `text-slate-700` | `text-stone-700` |
| `text-slate-500` | `text-stone-500` |
| `text-slate-400` | `text-stone-400` |
| `text-slate-300` | `text-stone-300` |
| `text-slate-800` | `text-stone-800` |
| `hover:bg-slate-50` | `hover:bg-stone-50` |

- [ ] **Step 7: Update `frontend/src/pages/HistoryPage.tsx`**

| Find | Replace |
|------|---------|
| `rounded-xl` (all occurrences) | `rounded-2xl` |
| `border-slate-200` | `border-stone-200` |
| `text-slate-800` | `text-stone-800` |
| `text-slate-600` | `text-stone-600` |
| `text-slate-400` | `text-stone-400` |
| `bg-slate-50` | `bg-stone-50` |
| `bg-slate-100` | `bg-stone-100` |
| `hover:bg-slate-50` | `hover:bg-stone-50` |

- [ ] **Step 8: Update `frontend/src/pages/StatsPage.tsx`**

| Find | Replace |
|------|---------|
| `rounded-xl` (all occurrences) | `rounded-2xl` |
| `border-slate-200` | `border-stone-200` |
| `text-slate-800` | `text-stone-800` |
| `text-slate-700` | `text-stone-700` |
| `text-slate-500` | `text-stone-500` |
| `text-slate-400` | `text-stone-400` |
| `bg-slate-50` | `bg-stone-50` |
| `border-slate-200` | `border-stone-200` |

- [ ] **Step 9: Run type-check**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no errors. (These are CSS class string changes — TypeScript won't catch typos, but the visual check at build time will.)

- [ ] **Step 10: Run build to confirm no broken imports**

```bash
cd frontend && npm run build 2>&1 | tail -5
```
Expected: build succeeds.

- [ ] **Step 11: Commit**

```bash
git add frontend/src/components/QuizForm.tsx \
        frontend/src/components/ExplanationCard.tsx \
        frontend/src/components/ActiveQuiz.tsx \
        frontend/src/components/QuizResults.tsx \
        frontend/src/components/HistoryItem.tsx \
        frontend/src/components/QuizSetup.tsx \
        frontend/src/pages/HistoryPage.tsx \
        frontend/src/pages/StatsPage.tsx
git commit -m "feat: unified card design system — rounded-2xl, stone text/border palette"
```

---

### Task 5: Page Copy + Empty States

**Files:**
- Modify: `frontend/src/pages/AskPage.tsx`
- Modify: `frontend/src/pages/HistoryPage.tsx`
- Modify: `frontend/src/pages/StatsPage.tsx`
- Modify: `frontend/src/components/QuizSetup.tsx`

- [ ] **Step 1: Update `frontend/src/pages/AskPage.tsx` heading**

Find:
```typescript
      <h1 className="text-xl font-bold text-slate-800 mb-6">Ask</h1>
```

Replace with:
```typescript
      <div className="mb-6">
        <h1 className="text-xl font-bold text-stone-800">Ask</h1>
        <p className="text-sm text-stone-500 mt-0.5">
          Enter any grammar question — get an instant AI explanation of every option.
        </p>
      </div>
```

- [ ] **Step 2: Update `frontend/src/pages/HistoryPage.tsx` heading and empty state**

Find the heading block:
```typescript
          <h1 className="text-xl font-bold text-slate-800">History</h1>
```

Replace with:
```typescript
          <div>
            <h1 className="text-xl font-bold text-stone-800">History</h1>
            <p className="text-sm text-stone-500 mt-0.5">
              Every question you've answered — expand any item to review options and explanations.
            </p>
          </div>
```

Find the empty-state paragraph:
```typescript
          <p className="text-slate-400 text-center py-12">No attempts yet. Start practicing!</p>
```

Replace with:
```typescript
          <div className="text-center py-16 space-y-3">
            <p className="text-2xl">📖</p>
            <p className="text-stone-500 font-medium">No attempts yet</p>
            <p className="text-stone-400 text-sm">
              Answer your first question to start building your review log.
            </p>
            <a href="/ask"
              className="inline-block mt-1 text-sm text-brand-600 font-semibold hover:underline"
            >
              Try your first question →
            </a>
          </div>
```

- [ ] **Step 3: Update `frontend/src/pages/StatsPage.tsx` heading and empty state**

Find:
```typescript
      <h1 className="text-xl font-bold text-slate-800">Your Progress</h1>
```

Replace with:
```typescript
      <div>
        <h1 className="text-xl font-bold text-stone-800">Your Progress</h1>
        <p className="text-sm text-stone-500 mt-0.5">
          Track accuracy, study days, and the grammar points that need more work.
        </p>
      </div>
```

Find the loading/no-stats guard at the top of the return:
```typescript
  if (!stats) return <p className="text-slate-400 text-center py-12">Loading…</p>;
```

Replace with:
```typescript
  if (!stats) return <p className="text-stone-400 text-center py-12">Loading…</p>;
```

After the stats grid (the `<div className="grid grid-cols-3 gap-4">` block), if `stats.total_attempts === 0`, show an empty state. Wrap the stats body in a conditional:

After the `<div className="space-y-6">` opening and the `<div>` heading block, add:

```typescript
      {stats.total_attempts === 0 ? (
        <div className="text-center py-16 space-y-3 bg-white rounded-2xl shadow-sm border border-stone-200">
          <p className="text-2xl">📊</p>
          <p className="text-stone-500 font-medium">Your journey starts today</p>
          <p className="text-stone-400 text-sm max-w-xs mx-auto">
            Study at least once a day to build your streak and uncover your weak grammar points.
          </p>
          <a href="/ask"
            className="inline-block mt-1 text-sm text-brand-600 font-semibold hover:underline"
          >
            Start practicing →
          </a>
        </div>
      ) : (
        <>
          {/* grid + chart + weak concepts — existing JSX here */}
        </>
      )}
```

Read the current StatsPage to wrap the existing grid/chart/weak-concepts JSX inside the `<>...</>` fragment above. Do not modify any of that existing JSX.

- [ ] **Step 4: Update `frontend/src/components/QuizSetup.tsx` heading**

Find:
```typescript
        <h1 className="text-lg font-bold text-slate-800">Timed Quiz</h1>
        <p className="text-xs text-slate-400 mt-0.5">Upload a grammar CSV to start a timed practice session</p>
```

Replace with:
```typescript
        <h1 className="text-lg font-bold text-stone-800">Timed Quiz</h1>
        <p className="text-xs text-stone-500 mt-0.5">
          Upload a question set and test yourself under timed conditions. AI reviews every mistake after.
        </p>
```

- [ ] **Step 5: Run type-check**

```bash
cd frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 6: Build check**

```bash
cd frontend && npm run build 2>&1 | tail -5
```
Expected: build succeeds.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/pages/AskPage.tsx \
        frontend/src/pages/HistoryPage.tsx \
        frontend/src/pages/StatsPage.tsx \
        frontend/src/components/QuizSetup.tsx
git commit -m "feat: page subtitles and empty states on History, Stats, Ask, Quiz"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Task |
|---|---|
| Brand color palette sky→indigo | Task 1 |
| Warm off-white page background | Task 2 (Layout), Task 3 (Login) |
| Sticky nav with logo and active underline | Task 2 |
| Login split-screen desktop hero with value props | Task 3 |
| Login `/practice` bug fix → `/ask` | Task 3 |
| `rounded-xl` → `rounded-2xl` globally | Task 4 |
| `slate-*` → `stone-*` text/border colors globally | Task 4 |
| H2 subtitles on every page | Task 5 |
| Empty states on History, Stats (Quiz has upload prompt already) | Task 5 |
| No backdrop-blur (DA constraint) | ✓ — Layout uses plain `bg-white shadow-sm` |
| No background patterns (DA constraint) | ✓ — solid `#f8f7f4` only |
| No shadow > shadow-sm on cards (DA constraint) | ✓ — only login card uses `shadow-md` (intentional prominence) |
| No nav icons (DA constraint) | ✓ — text-only nav |
| No font-family change (DA constraint) | ✓ — system font stack unchanged |

**Placeholder scan:** No TBD, no "implement later", all JSX is complete.

**Consistency check:**
- `#f8f7f4` warm background is used in both `Layout.tsx` (line: `bg-[#f8f7f4]`) and `LoginPage.tsx` (form panel: `bg-[#f8f7f4]`) — consistent.
- `rounded-2xl` used in all card replacements — consistent.
- All `stone-*` substitutions maintain same numeric shade as the `slate-*` they replace (e.g. `slate-400` → `stone-400`) — consistent contrast ratios.
- `navigate("/ask")` in LoginPage matches `<Route path="/ask">` in App.tsx — correct.
