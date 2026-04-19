# History, Quiz, Ask Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrich history items with options/explanations; add quiz question-count control and results persistence with per-question on-demand explanations; rename Practice→Ask with auto-recording; display top-10 weak concepts on History page; fix animation stagger.

**Architecture:** Backend gains two new endpoints (weak-concepts for history, per-attempt explain) and enriches existing responses with `options`/`explanation` fields. Frontend types, API client, and components are updated in layers: schema→client→components, so each task is self-contained.

**Tech Stack:** FastAPI + SQLAlchemy 2 async, React 18 + TypeScript, Tailwind CSS v4 (`@utility` pattern), Vite 5

---

## File Map

| File | Change |
|------|--------|
| `backend/schemas/api.py` | Add `options`/`explanation` to `AttemptResponse`; add `explanation` to `AttemptRecord`; add `WeakConceptsResponse`; change `record-batch` response type |
| `backend/routers/history.py` | Serialize `options`/`explanation` in GET; pass `explanation` in POST/record; add `GET /history/weak-concepts`; add `POST /history/{attempt_id}/explain` |
| `backend/routers/practice.py` | Return IDs from `record-batch` |
| `frontend/src/types/index.ts` | Add `options`/`explanation` to `AttemptResponse` |
| `frontend/src/api/client.ts` | Add `explainAttempt`, `getWeakConcepts`; update `recordAttempt` body; type `recordBatch` return |
| `frontend/src/components/HistoryItem.tsx` | Expandable card with options, explanation, Generate button; accept `index` prop for stagger |
| `frontend/src/pages/HistoryPage.tsx` | Weak-concepts panel at top; pass index to HistoryItem |
| `frontend/src/pages/PracticePage.tsx` | Rename to `AskPage.tsx`, update heading |
| `frontend/src/App.tsx` | `/ask` route + `/practice` redirect |
| `frontend/src/components/Layout.tsx` | Nav: `/ask` label "Ask" |
| `frontend/src/hooks/useQuizSession.ts` | Auto-record attempt when `phase === "done"` |
| `frontend/src/components/QuizSetup.tsx` | `maxQuestions` stepper |
| `frontend/src/hooks/useQuizMode.ts` | Collect attempt IDs from record-batch; sessionStorage persistence; expose `wrongAttemptIds` |
| `frontend/src/components/QuizResults.tsx` | Per-wrong-answer Generate button; "View Last Results" restored from sessionStorage |
| `frontend/src/pages/QuizPage.tsx` | Restore results from sessionStorage on setup screen |

---

### Task A: Backend — Enrich responses + new endpoints

**Files:**
- Modify: `backend/schemas/api.py`
- Modify: `backend/routers/history.py`
- Modify: `backend/routers/practice.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_history_enhancements.py
import json
import pytest
from httpx import AsyncClient
from backend.main import app
from backend.services.analytics import record_attempt
from backend.database import get_db

@pytest.mark.asyncio
async def test_history_returns_options_and_explanation(auth_client, db_session, test_user):
    await record_attempt(
        db=db_session,
        user_id=test_user.id,
        question_text="Q?",
        options=["A: foo", "B: bar"],
        correct_answer="A",
        llm_answer="A",
        explanation="Because foo.",
        concepts=["grammar"],
        user_marked_correct=True,
    )
    resp = await auth_client.get("/api/v1/history")
    assert resp.status_code == 200
    item = resp.json()[0]
    assert item["options"] == ["A: foo", "B: bar"]
    assert item["explanation"] == "Because foo."


@pytest.mark.asyncio
async def test_record_saves_explanation(auth_client):
    resp = await auth_client.post("/api/v1/history/record", json={
        "question_text": "Q?",
        "options": ["A: x", "B: y"],
        "correct_answer": "A",
        "llm_answer": "A",
        "user_marked_correct": True,
        "concepts": [],
        "explanation": "Saved explanation.",
    })
    assert resp.status_code == 201

    history = await auth_client.get("/api/v1/history")
    assert history.json()[0]["explanation"] == "Saved explanation."


@pytest.mark.asyncio
async def test_weak_concepts_limit(auth_client, db_session, test_user):
    for i in range(12):
        await record_attempt(
            db=db_session, user_id=test_user.id,
            question_text=f"Q{i}", options=[], correct_answer="A",
            llm_answer="B", explanation="", concepts=[f"concept_{i}"],
            user_marked_correct=False,
        )
    resp = await auth_client.get("/api/v1/history/weak-concepts?limit=10")
    assert resp.status_code == 200
    assert len(resp.json()["concepts"]) == 10


@pytest.mark.asyncio
async def test_record_batch_returns_ids(auth_client):
    resp = await auth_client.post("/api/v1/practice/record-batch", json={
        "attempts": [
            {"question_text": "Q?", "options": ["A: x"], "correct_answer": "A",
             "user_answer": "A", "user_marked_correct": True}
        ]
    })
    assert resp.status_code == 201
    body = resp.json()
    assert "ids" in body
    assert len(body["ids"]) == 1
    assert isinstance(body["ids"][0], int)
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
python -m pytest tests/test_history_enhancements.py -v
```
Expected: 4 failures (missing fields/endpoints)

- [ ] **Step 3: Update `backend/schemas/api.py`**

Replace the existing `AttemptResponse` and `AttemptRecord` and add `WeakConceptsResponse`:

```python
class AttemptRecord(BaseModel):
    question_text: str
    options: list[str]
    correct_answer: str
    llm_answer: str
    user_marked_correct: bool
    concepts: list[str]
    explanation: str = ""


class AttemptResponse(BaseModel):
    id: int
    question_text: str
    correct_answer: str
    llm_answer: str
    user_marked_correct: bool
    concepts: list[str]
    created_at: str
    options: list[str] = []
    explanation: str = ""


class WeakConceptsResponse(BaseModel):
    concepts: list[str]
```

- [ ] **Step 4: Update `backend/routers/history.py`**

Full replacement of the file:

```python
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db
from backend.models.tables import Attempt, User
from backend.auth.dependencies import get_current_user
from backend.schemas.api import (
    AttemptResponse, AttemptRecord, StatsResponse, WeakConceptsResponse,
)
from backend.services.analytics import get_stats, get_weak_concepts, record_attempt
from backend.services.llm.factory import get_llm_provider

router = APIRouter(prefix="/api/v1")


@router.get("/history", response_model=list[AttemptResponse])
async def get_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Attempt)
        .where(Attempt.user_id == current_user.id)
        .order_by(Attempt.created_at.desc())
        .limit(limit)
    )
    attempts = result.scalars().all()
    return [
        AttemptResponse(
            id=a.id,
            question_text=a.question_text,
            correct_answer=a.correct_answer,
            llm_answer=a.llm_answer,
            user_marked_correct=a.user_marked_correct,
            concepts=json.loads(a.concepts or "[]"),
            created_at=a.created_at.isoformat(),
            options=json.loads(a.options or "[]"),
            explanation=a.explanation or "",
        )
        for a in attempts
    ]


@router.post("/history/record", status_code=201)
async def record(
    body: AttemptRecord,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await record_attempt(
        db=db,
        user_id=current_user.id,
        question_text=body.question_text,
        options=body.options,
        correct_answer=body.correct_answer,
        llm_answer=body.llm_answer,
        explanation=body.explanation,
        concepts=body.concepts,
        user_marked_correct=body.user_marked_correct,
    )
    return {"status": "recorded"}


@router.get("/history/weak-concepts", response_model=WeakConceptsResponse)
async def history_weak_concepts(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    concepts = await get_weak_concepts(db, current_user.id, limit=limit)
    return WeakConceptsResponse(concepts=concepts)


@router.post("/history/{attempt_id}/explain")
async def explain_attempt(
    attempt_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Attempt).where(
            Attempt.id == attempt_id,
            Attempt.user_id == current_user.id,
        )
    )
    attempt = result.scalar_one_or_none()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    if attempt.explanation:
        return {"explanation": attempt.explanation}

    provider = get_llm_provider()
    options_text = "\n".join(json.loads(attempt.options or "[]"))
    prompt = (
        f"Japanese grammar question:\n{attempt.question_text}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Correct answer: {attempt.correct_answer}\n"
        f"User's answer: {attempt.llm_answer}\n\n"
        "In 2-3 sentences, explain why the correct answer is right and why the user's answer is wrong. "
        "Be concise and educational."
    )
    explanation = await provider.complete(prompt)

    attempt.explanation = explanation
    await db.commit()
    return {"explanation": explanation}


@router.get("/stats", response_model=StatsResponse)
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stats = await get_stats(db, current_user.id)
    weak = await get_weak_concepts(db, current_user.id)
    return StatsResponse(weak_concepts=weak, **stats)
```

- [ ] **Step 5: Update `backend/routers/practice.py` record-batch to return IDs**

Replace the `record_batch` function (lines 61–79):

```python
@router.post("/record-batch", status_code=201)
async def record_batch(
    body: BatchRecordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    ids = []
    for item in body.attempts:
        attempt = await record_attempt(
            db=db,
            user_id=current_user.id,
            question_text=item.question_text,
            options=item.options,
            correct_answer=item.correct_answer,
            llm_answer=item.user_answer,
            explanation="",
            concepts=[],
            user_marked_correct=item.user_marked_correct,
        )
        ids.append(attempt.id)
    return {"recorded": len(ids), "ids": ids}
```

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_history_enhancements.py -v
```
Expected: 4 tests pass

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py
```
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add backend/schemas/api.py backend/routers/history.py backend/routers/practice.py tests/test_history_enhancements.py
git commit -m "feat: enrich AttemptResponse with options/explanation, add weak-concepts and explain endpoints, record-batch returns IDs"
```

---

### Task B: Frontend types + API client

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Update `frontend/src/types/index.ts`**

Replace the `AttemptResponse` interface:

```typescript
export interface AttemptResponse {
  id: number;
  question_text: string;
  correct_answer: string;
  llm_answer: string;
  user_marked_correct: boolean;
  concepts: string[];
  created_at: string;
  options: string[];
  explanation: string;
}
```

- [ ] **Step 2: Run type-check to confirm failure**

```bash
cd frontend && npm run type-check
```
Expected: errors about `options`/`explanation` used but not in type (or no errors yet — proceed to Step 3)

- [ ] **Step 3: Update `frontend/src/api/client.ts`**

Make these targeted changes:

1. Update `recordAttempt` signature (add `explanation`):
```typescript
  recordAttempt: (body: {
    question_text: string;
    options: string[];
    correct_answer: string;
    llm_answer: string;
    user_marked_correct: boolean;
    concepts: string[];
    explanation?: string;
  }) => post("/api/v1/history/record", body, true),
```

2. Update `recordBatch` return type:
```typescript
  recordBatch: (attempts: {
    question_text: string; options: string[]; correct_answer: string;
    user_answer: string; user_marked_correct: boolean;
  }[]) =>
    post<{ recorded: number; ids: number[] }>("/api/v1/practice/record-batch", { attempts }, true),
```

3. Add `getWeakConcepts` after `getStats`:
```typescript
  getWeakConcepts: (limit = 10) =>
    get<{ concepts: string[] }>(`/api/v1/history/weak-concepts?limit=${limit}`),
```

4. Add `explainAttempt` after `getWeakConcepts`:
```typescript
  explainAttempt: (attemptId: number) =>
    post<{ explanation: string }>(`/api/v1/history/${attemptId}/explain`, {}, true),
```

- [ ] **Step 4: Run type-check**

```bash
cd frontend && npm run type-check
```
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/types/index.ts frontend/src/api/client.ts
git commit -m "feat: add options/explanation to AttemptResponse type, add explainAttempt and getWeakConcepts API methods"
```

---

### Task C: HistoryItem expandable card + stagger

**Files:**
- Modify: `frontend/src/components/HistoryItem.tsx`
- Modify: `frontend/src/pages/HistoryPage.tsx`

- [ ] **Step 1: Rewrite `frontend/src/components/HistoryItem.tsx`**

Full file replacement:

```typescript
import { useState } from "react";
import { api } from "../api/client";
import type { AttemptResponse } from "../types";

interface Props {
  attempt: AttemptResponse;
  index?: number;
}

export default function HistoryItem({ attempt, index = 0 }: Props) {
  const date = new Date(attempt.created_at).toLocaleDateString();
  const [expanded, setExpanded] = useState(false);
  const [explanation, setExplanation] = useState(attempt.explanation || "");
  const [generating, setGenerating] = useState(false);
  const [genError, setGenError] = useState("");

  const handleGenerate = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setGenerating(true);
    setGenError("");
    try {
      const res = await api.explainAttempt(attempt.id);
      setExplanation(res.explanation);
    } catch {
      setGenError("Failed to generate. Try again.");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div
      className="bg-white border border-slate-200 rounded-xl p-3 space-y-1.5 animate-fade-in
        hover:shadow-sm transition-shadow cursor-pointer"
      style={{ animationDelay: `${index * 80}ms` }}
      onClick={() => setExpanded(v => !v)}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm text-slate-800 flex-1 leading-snug">{attempt.question_text}</p>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className={`text-xs font-bold px-2 py-0.5 rounded-full
            ${attempt.user_marked_correct ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"}`}>
            {attempt.user_marked_correct ? "✓" : "✗"}
          </span>
          <span className="text-slate-300 text-xs">{expanded ? "▲" : "▼"}</span>
        </div>
      </div>

      {/* Summary row */}
      <div className="flex items-center gap-3 text-xs text-slate-400">
        <span>Answer: <span className="font-semibold text-slate-600">{attempt.correct_answer}</span></span>
        <span>{date}</span>
      </div>

      {/* Concepts */}
      {attempt.concepts.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {attempt.concepts.map(c => (
            <span key={c} className="bg-slate-100 text-slate-500 text-xs px-1.5 py-0.5 rounded-full">{c}</span>
          ))}
        </div>
      )}

      {/* Expanded section */}
      {expanded && (
        <div className="border-t border-slate-100 pt-2 space-y-2 animate-fade-in">
          {/* Options grid */}
          {attempt.options.length > 0 && (
            <div className="space-y-1">
              {attempt.options.map((opt, i) => {
                const label = opt.split(":")[0]?.trim() ?? String.fromCharCode(65 + i);
                const isCorrect = attempt.correct_answer === label;
                const isUserAnswer = attempt.llm_answer === label;
                return (
                  <div
                    key={i}
                    className={`text-xs px-2 py-1 rounded-lg
                      ${isCorrect
                        ? "bg-green-50 text-green-700 font-medium border border-green-200"
                        : isUserAnswer && !attempt.user_marked_correct
                          ? "bg-red-50 text-red-600 border border-red-200"
                          : "bg-slate-50 text-slate-600"
                      }`}
                  >
                    {opt}
                  </div>
                );
              })}
            </div>
          )}

          {/* Explanation */}
          {explanation ? (
            <p className="text-xs text-slate-600 leading-relaxed">{explanation}</p>
          ) : (
            <div>
              {genError && <p className="text-xs text-red-500 mb-1">{genError}</p>}
              <button
                onClick={handleGenerate}
                disabled={generating}
                className="text-xs text-brand-500 font-semibold hover:underline disabled:opacity-50"
              >
                {generating ? "Generating…" : "Generate explanation"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Update `frontend/src/pages/HistoryPage.tsx` to pass index**

Change the list render (currently line 39):
```typescript
{shown.map((a, i) => <HistoryItem key={a.id} attempt={a} index={i} />)}
```

- [ ] **Step 3: Run type-check**

```bash
cd frontend && npm run type-check
```
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/HistoryItem.tsx frontend/src/pages/HistoryPage.tsx
git commit -m "feat: expandable history items with options/explanation and generate button; stagger animation"
```

---

### Task D: History page — top-10 weak concepts panel

**Files:**
- Modify: `frontend/src/pages/HistoryPage.tsx`

- [ ] **Step 1: Update `frontend/src/pages/HistoryPage.tsx`**

Full file replacement:

```typescript
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AttemptResponse } from "../types";
import HistoryItem from "../components/HistoryItem";

export default function HistoryPage() {
  const [attempts, setAttempts] = useState<AttemptResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "wrong">("all");
  const [weakConcepts, setWeakConcepts] = useState<string[]>([]);
  const [conceptsOpen, setConceptsOpen] = useState(true);

  useEffect(() => {
    api.getHistory().then(setAttempts).finally(() => setLoading(false));
    api.getWeakConcepts(10).then(res => setWeakConcepts(res.concepts)).catch(() => {});
  }, []);

  const shown = filter === "wrong" ? attempts.filter(a => !a.user_marked_correct) : attempts;

  return (
    <div className="space-y-5">
      {/* Weak concepts panel */}
      {weakConcepts.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden animate-fade-in">
          <button
            onClick={() => setConceptsOpen(v => !v)}
            className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-slate-50 transition-colors"
          >
            <span className="text-sm font-semibold text-slate-700">
              Top {weakConcepts.length} Weak Points
            </span>
            <span className="text-slate-400 text-xs">{conceptsOpen ? "▲" : "▼"}</span>
          </button>
          {conceptsOpen && (
            <div className="px-4 pb-3 flex flex-wrap gap-1.5 animate-fade-in">
              {weakConcepts.map((c, i) => (
                <span
                  key={c}
                  className="bg-red-50 text-red-600 text-xs px-2 py-1 rounded-full border border-red-100 animate-pop-in"
                  style={{ animationDelay: `${i * 40}ms` }}
                >
                  {i + 1}. {c}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Filter + list */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-xl font-bold text-slate-800">History</h1>
          <div className="flex gap-2">
            {(["all", "wrong"] as const).map(f => (
              <button key={f} onClick={() => setFilter(f)}
                className={`text-sm px-3 py-1.5 rounded-lg font-medium transition-colors
                  ${filter === f ? "bg-brand-500 text-white" : "bg-white border border-slate-200 text-slate-600"}`}
              >
                {f === "all" ? "All" : "Wrong only"}
              </button>
            ))}
          </div>
        </div>

        {loading ? (
          <p className="text-slate-400 text-center py-12">Loading…</p>
        ) : shown.length === 0 ? (
          <p className="text-slate-400 text-center py-12">No attempts yet. Start practicing!</p>
        ) : (
          <div className="space-y-3">
            {shown.map((a, i) => <HistoryItem key={a.id} attempt={a} index={i} />)}
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Run type-check**

```bash
cd frontend && npm run type-check
```
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/HistoryPage.tsx
git commit -m "feat: top-10 weak concepts panel on history page"
```

---

### Task E: Rename Practice → Ask

**Files:**
- Rename: `frontend/src/pages/PracticePage.tsx` → `frontend/src/pages/AskPage.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/Layout.tsx`

- [ ] **Step 1: Create `frontend/src/pages/AskPage.tsx`**

Content is identical to `PracticePage.tsx` except the heading changes:

```typescript
import { useQuizSession } from "../hooks/useQuizSession";
import QuizForm from "../components/QuizForm";
import ExplanationCard from "../components/ExplanationCard";
import StreamingProgress from "../components/StreamingProgress";
import ImageExtractor from "../components/ImageExtractor";
import RecommendationPanel from "../components/RecommendationPanel";

export default function AskPage() {
  const {
    fields, setFields, setOption,
    phase, streamText, result, error,
    submit, reset,
  } = useQuizSession();

  const fillFromImage = (question: string, options: string[]) => {
    setFields({ question, options: [...options, "", "", "", ""].slice(0, 4) });
  };

  return (
    <div>
      <h1 className="text-xl font-bold text-slate-800 mb-6">Ask</h1>
      <QuizForm
        question={fields.question}
        options={fields.options}
        onQuestionChange={q => setFields(f => ({ ...f, question: q }))}
        onOptionChange={setOption}
        onSubmit={submit}
        onReset={reset}
        disabled={phase !== "idle" && phase !== "done"}
        error={error}
      />
      <ImageExtractor onExtract={fillFromImage} />
      <StreamingProgress phase={phase} />
      <ExplanationCard streamText={streamText} result={result} phase={phase} />
      {result && <RecommendationPanel concepts={result.concepts} />}
    </div>
  );
}
```

- [ ] **Step 2: Update `frontend/src/App.tsx`**

Full replacement:

```typescript
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import LoginPage from "./pages/LoginPage";
import Layout from "./components/Layout";
import AskPage from "./pages/AskPage";
import HistoryPage from "./pages/HistoryPage";
import StatsPage from "./pages/StatsPage";
import QuizPage from "./pages/QuizPage";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/ask" element={<Layout><AskPage /></Layout>} />
          <Route path="/practice" element={<Navigate to="/ask" replace />} />
          <Route path="/quiz" element={<Layout><QuizPage /></Layout>} />
          <Route path="/history" element={<Layout><HistoryPage /></Layout>} />
          <Route path="/stats" element={<Layout><StatsPage /></Layout>} />
          <Route path="*" element={<Navigate to="/ask" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
```

- [ ] **Step 3: Update `frontend/src/components/Layout.tsx`** nav item

Change:
```typescript
const NAV_ITEMS = [
  { to: "/ask",      label: "Ask" },
  { to: "/quiz",     label: "Quiz" },
  { to: "/history",  label: "History" },
  { to: "/stats",    label: "Stats" },
];
```

- [ ] **Step 4: Delete old PracticePage.tsx**

```bash
rm frontend/src/pages/PracticePage.tsx
```

- [ ] **Step 5: Run type-check**

```bash
cd frontend && npm run type-check
```
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/AskPage.tsx frontend/src/App.tsx frontend/src/components/Layout.tsx
git rm frontend/src/pages/PracticePage.tsx
git commit -m "feat: rename Practice page to Ask (/ask route, nav label, heading)"
```

---

### Task F: Ask mode auto-record with explanation

**Files:**
- Modify: `frontend/src/hooks/useQuizSession.ts`

- [ ] **Step 1: Rewrite `frontend/src/hooks/useQuizSession.ts`**

The `submit` function must call `api.recordAttempt` with the full explanation after the SSE stream completes. The `optLabels` variable is already scoped inside `submit` and available at the `setPhase("done")` point.

Full file replacement:

```typescript
import { useState, useCallback } from "react";
import { api } from "../api/client";
import type { Phase, SolveResult } from "../types";

interface Fields { question: string; options: string[] }

export function useQuizSession() {
  const [fields, setFields] = useState<Fields>({ question: "", options: ["", "", "", ""] });
  const [phase, setPhase] = useState<Phase>("idle");
  const [streamText, setStreamText] = useState("");
  const [result, setResult] = useState<SolveResult | null>(null);
  const [error, setError] = useState("");

  const setOption = (i: number, val: string) =>
    setFields(f => { const opts = [...f.options]; opts[i] = val; return { ...f, options: opts }; });

  const submit = useCallback(async () => {
    if (!fields.question.trim() || fields.options.filter(o => o.trim()).length < 2) {
      setError("Enter a question and at least 2 options.");
      return;
    }
    setError(""); setStreamText(""); setResult(null); setPhase("answering");

    const optLabels = fields.options
      .filter(o => o.trim())
      .map((o, i) => `${String.fromCharCode(65 + i)}: ${o}`);

    try {
      let finalResult: SolveResult | null = null;

      for await (const { event, data } of api.solveStreamFetch(fields.question, optLabels)) {
        if (event === "token") {
          setStreamText(t => t + data);
          if (data.includes("EXPLANATION")) setPhase("explaining");
        } else if (event === "result") {
          finalResult = JSON.parse(data) as SolveResult;
          setResult(finalResult);
          setPhase("done");
        }
      }

      if (finalResult) {
        api.recordAttempt({
          question_text: fields.question,
          options: optLabels,
          correct_answer: finalResult.answer,
          llm_answer: finalResult.answer,
          user_marked_correct: true,
          concepts: finalResult.concepts,
          explanation: finalResult.explanation,
        }).catch(() => {/* non-fatal */});
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Is the backend running?");
      setPhase("idle");
    }
  }, [fields]);

  const reset = () => {
    setFields({ question: "", options: ["", "", "", ""] });
    setPhase("idle"); setStreamText(""); setResult(null); setError("");
  };

  return { fields, setFields, setOption, phase, streamText, result, error, submit, reset };
}
```

- [ ] **Step 2: Run type-check**

```bash
cd frontend && npm run type-check
```
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/hooks/useQuizSession.ts
git commit -m "feat: auto-record Ask mode attempts with cached explanation"
```

---

### Task G: Quiz question count control

**Files:**
- Modify: `frontend/src/components/QuizSetup.tsx`

- [ ] **Step 1: Add `maxQuestions` state and stepper to `QuizSetup.tsx`**

Add state after the `historyCount` state line (currently around line 16):
```typescript
const [maxQuestions, setMaxQuestions] = useState(0); // 0 = all
```

Add the stepper UI after the "Include wrong history" block (before the closing `</div>` of the config card, around line 127). Insert this new section:

```typescript
        {/* Question count */}
        <div className="flex items-center justify-between border-t border-slate-100 pt-3">
          <div>
            <p className="text-sm font-medium text-slate-700">Max questions</p>
            <p className="text-xs text-slate-400">Set 0 for all</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setMaxQuestions(n => Math.max(0, n - 5))}
              aria-label="Decrease max questions"
              className="w-7 h-7 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50
                active:scale-95 transition-all text-sm flex items-center justify-center"
            >−</button>
            <span className="text-sm font-semibold text-slate-800 w-12 text-center">
              {maxQuestions === 0 ? "All" : `${maxQuestions}`}
            </span>
            <button
              onClick={() => setMaxQuestions(n => n + 5)}
              aria-label="Increase max questions"
              className="w-7 h-7 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50
                active:scale-95 transition-all text-sm flex items-center justify-center"
            >+</button>
          </div>
        </div>
```

Update the Start button's `onClick` to slice the questions:

```typescript
onClick={() => {
  const qs = maxQuestions > 0 ? questions.slice(0, maxQuestions) : questions;
  onStart(qs, minutes * 60);
}}
```

Update the Start button text to reflect the actual count:

```typescript
{questions.length > 0
  ? `Start Quiz — ${maxQuestions > 0 ? Math.min(maxQuestions, questions.length) : questions.length} questions`
  : "Upload CSV to start"}
```

- [ ] **Step 2: Run type-check**

```bash
cd frontend && npm run type-check
```
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/QuizSetup.tsx
git commit -m "feat: max questions stepper in quiz setup"
```

---

### Task H: Quiz results persistence + per-wrong-answer explain

**Files:**
- Modify: `frontend/src/hooks/useQuizMode.ts`
- Modify: `frontend/src/components/QuizResults.tsx`
- Modify: `frontend/src/pages/QuizPage.tsx`

- [ ] **Step 1: Update `frontend/src/hooks/useQuizMode.ts`**

Full file replacement (adds `wrongAttemptIds` state, sessionStorage save, and sessionStorage restore):

```typescript
import { useState, useCallback, useRef, useEffect } from "react";
import { api } from "../api/client";
import type { QuizQuestion, AnalysisItem } from "../types";

export type QuizScreen = "setup" | "active" | "results";

const STORAGE_KEY = "quiz_last_results";

interface QuizAnswer {
  questionIndex: number;
  userAnswer: string;
  correctAnswer: string;
  isCorrect: boolean;
  question: string;
  options: string[];
}

interface PersistedResults {
  answers: QuizAnswer[];
  analyses: AnalysisItem[];
  wrongAttemptIds: number[];
  timestamp: number;
}

export function loadPersistedResults(): PersistedResults | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedResults;
  } catch {
    return null;
  }
}

export function useQuizMode() {
  const [screen, setScreen] = useState<QuizScreen>("setup");
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [answers, setAnswers] = useState<QuizAnswer[]>([]);
  const [timeLeft, setTimeLeft] = useState(0);
  const [timeLimitSec, setTimeLimitSec] = useState(0);
  const [analyses, setAnalyses] = useState<AnalysisItem[]>([]);
  const [analyzing, setAnalyzing(false);
  const [wrongAttemptIds, setWrongAttemptIds] = useState<number[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const answersRef = useRef<QuizAnswer[]>([]);

  const stopTimer = () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
  };

  const finishQuiz = useCallback(async (finalAnswers: QuizAnswer[]) => {
    stopTimer();
    setScreen("results");
    setAnalyzing(true);

    const wrong = finalAnswers
      .filter(a => !a.isCorrect)
      .map(a => ({
        question: a.question,
        options: a.options,
        correct_answer: a.correctAnswer,
        user_answer: a.userAnswer,
      }));

    // Record all attempts, collect IDs
    let resolvedWrongIds: number[] = [];
    try {
      const batchResult = await api.recordBatch(
        finalAnswers.map(a => ({
          question_text: a.question,
          options: a.options,
          correct_answer: a.correctAnswer,
          user_answer: a.userAnswer,
          user_marked_correct: a.isCorrect,
        }))
      );
      // Map attempt IDs to wrong answers (same order as finalAnswers)
      const allIds = batchResult.ids ?? [];
      resolvedWrongIds = finalAnswers
        .map((a, i) => ({ isCorrect: a.isCorrect, id: allIds[i] ?? 0 }))
        .filter(x => !x.isCorrect)
        .map(x => x.id);
    } catch {/* non-fatal */}

    setWrongAttemptIds(resolvedWrongIds);

    // Get LLM analysis for wrong answers
    let resolvedAnalyses: AnalysisItem[] = [];
    if (wrong.length > 0) {
      try {
        const res = await api.analyzePractice(wrong);
        resolvedAnalyses = res.analyses;
        setAnalyses(resolvedAnalyses);
      } catch { setAnalyses([]); }
    }
    setAnalyzing(false);

    // Persist results to sessionStorage
    const toSave: PersistedResults = {
      answers: finalAnswers,
      analyses: resolvedAnalyses,
      wrongAttemptIds: resolvedWrongIds,
      timestamp: Date.now(),
    };
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(toSave)); } catch {/* ignore */}
  }, []);

  const startQuiz = useCallback((qs: QuizQuestion[], timeSec: number) => {
    setQuestions(qs);
    setCurrent(0);
    setSelected(null);
    setAnswers([]);
    setAnalyses([]);
    setWrongAttemptIds([]);
    setTimeLimitSec(timeSec);
    setTimeLeft(timeSec);
    setScreen("active");
  }, []);

  const restoreResults = useCallback((saved: PersistedResults) => {
    setAnswers(saved.answers);
    setAnalyses(saved.analyses);
    setWrongAttemptIds(saved.wrongAttemptIds);
    setScreen("results");
  }, []);

  const finishedRef = useRef(false);

  // Countdown timer
  useEffect(() => {
    if (screen !== "active" || timeLimitSec === 0) return;
    finishedRef.current = false;
    timerRef.current = setInterval(() => {
      setTimeLeft(t => {
        if (t <= 1) {
          if (!finishedRef.current) {
            finishedRef.current = true;
            setTimeout(() => finishQuiz(answersRef.current), 0);
          }
          return 0;
        }
        return t - 1;
      });
    }, 1000);
    return stopTimer;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screen, timeLimitSec]);

  const confirmAnswer = useCallback(() => {
    if (!selected || current >= questions.length) return;
    const q = questions[current];
    const isCorrect = selected === q.correct_answer;
    const newAnswer: QuizAnswer = {
      questionIndex: current,
      userAnswer: selected,
      correctAnswer: q.correct_answer,
      isCorrect,
      question: q.question,
      options: q.options,
    };
    const newAnswers = [...answers, newAnswer];
    answersRef.current = newAnswers;
    setAnswers(newAnswers);
    setSelected(null);

    if (current + 1 >= questions.length) {
      finishQuiz(newAnswers);
    } else {
      setCurrent(c => c + 1);
    }
  }, [selected, current, questions, answers, finishQuiz]);

  const reset = () => {
    stopTimer();
    setScreen("setup");
    setQuestions([]);
    setCurrent(0);
    setSelected(null);
    answersRef.current = [];
    setAnswers([]);
    setAnalyses([]);
    setWrongAttemptIds([]);
  };

  const score = answers.filter(a => a.isCorrect).length;
  const timerWarning = timeLimitSec > 0 && timeLeft < 60;
  const timerCritical = timeLimitSec > 0 && timeLeft < 30;

  return {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    wrongAttemptIds,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset, restoreResults,
  };
}
```

> **Note**: The line `const [analyzing, setAnalyzing(false);` in the code above has a typo — write it as:
> ```typescript
> const [analyzing, setAnalyzing] = useState(false);
> ```

- [ ] **Step 2: Update `frontend/src/components/QuizResults.tsx`**

Full file replacement (adds per-wrong-answer Generate button and `wrongAttemptIds` prop):

```typescript
import { useState } from "react";
import { api } from "../api/client";
import type { AnalysisItem } from "../types";

interface Props {
  score: number;
  total: number;
  analyses: AnalysisItem[];
  analyzing: boolean;
  wrongAttemptIds: number[];
  onRetry: () => void;
}

export default function QuizResults({
  score, total, analyses, analyzing, wrongAttemptIds, onRetry,
}: Props) {
  const pct = total > 0 ? Math.round((score / total) * 100) : 0;
  const wrong = total - score;
  const [explanations, setExplanations] = useState<Record<number, string>>({});
  const [generating, setGenerating] = useState<Record<number, boolean>>({});

  const handleGenerate = async (index: number) => {
    const attemptId = wrongAttemptIds[index];
    if (!attemptId) return;
    setGenerating(g => ({ ...g, [index]: true }));
    try {
      const res = await api.explainAttempt(attemptId);
      setExplanations(e => ({ ...e, [index]: res.explanation }));
    } catch {/* ignore */} finally {
      setGenerating(g => ({ ...g, [index]: false }));
    }
  };

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Score card */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 text-center animate-pop-in">
        <p className="text-5xl font-black text-brand-500">{pct}%</p>
        <p className="text-sm text-slate-500 mt-1">
          {score} correct · {wrong} wrong · {total} total
        </p>
        <div className="h-2 bg-slate-100 rounded-full overflow-hidden mt-3">
          <div
            className="h-full bg-brand-500 rounded-full transition-all duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Wrong answers with LLM analysis */}
      {wrong > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            {analyzing ? "Analyzing wrong answers…" : `${wrong} wrong answer${wrong > 1 ? "s" : ""}`}
          </p>

          {analyzing ? (
            <div className="space-y-2">
              {Array.from({ length: wrong }).map((_, i) => (
                <div key={i} className="h-16 bg-slate-100 rounded-xl animate-timer-pulse" />
              ))}
            </div>
          ) : analyses.length === 0 ? (
            <p className="text-xs text-slate-400 italic">Analysis unavailable</p>
          ) : (
            analyses.map((item, i) => {
              const explanation = explanations[i] || item.explanation;
              return (
                <div
                  key={i}
                  className="bg-white border border-red-100 rounded-xl p-3 space-y-2 animate-slide-in"
                  style={{ animationDelay: `${i * 60}ms` }}
                >
                  <p className="text-sm text-slate-800 leading-snug">{item.question}</p>
                  <div className="flex gap-3 text-xs">
                    <span className="text-red-500 font-semibold">You: {item.user_answer}</span>
                    <span className="text-green-600 font-semibold">Correct: {item.correct_answer}</span>
                  </div>
                  {explanation ? (
                    <p className="text-xs text-slate-600 leading-relaxed border-t border-slate-100 pt-2">
                      {explanation}
                    </p>
                  ) : wrongAttemptIds[i] ? (
                    <div className="border-t border-slate-100 pt-2">
                      <button
                        onClick={() => handleGenerate(i)}
                        disabled={generating[i]}
                        className="text-xs text-brand-500 font-semibold hover:underline disabled:opacity-50"
                      >
                        {generating[i] ? "Generating…" : "Generate explanation"}
                      </button>
                    </div>
                  ) : null}
                </div>
              );
            })
          )}
        </div>
      )}

      <button
        onClick={onRetry}
        className="w-full border border-slate-200 hover:bg-slate-50 active:scale-[0.98]
          text-slate-600 py-2.5 rounded-xl text-sm font-medium transition-all"
      >
        ← Back to Setup
      </button>
    </div>
  );
}
```

- [ ] **Step 3: Update `frontend/src/pages/QuizPage.tsx` to pass `wrongAttemptIds` and add "View Last Results"**

First read `frontend/src/pages/QuizPage.tsx` to see its current structure, then update it to:
1. Import `loadPersistedResults` from `useQuizMode`
2. Pass `wrongAttemptIds` and `restoreResults` from the hook
3. Check for persisted results on the setup screen and show a "View Last Results" button

```typescript
import { loadPersistedResults, useQuizMode } from "../hooks/useQuizMode";
import QuizSetup from "../components/QuizSetup";
import QuizActive from "../components/QuizActive";
import QuizResults from "../components/QuizResults";
import type { QuizQuestion } from "../types";

export default function QuizPage() {
  const {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    wrongAttemptIds,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset, restoreResults,
  } = useQuizMode();

  const savedResults = loadPersistedResults();

  if (screen === "setup") {
    return (
      <div className="space-y-3">
        <QuizSetup onStart={(qs: QuizQuestion[], timeSec: number) => startQuiz(qs, timeSec)} />
        {savedResults && (
          <button
            onClick={() => restoreResults(savedResults)}
            className="w-full border border-slate-200 hover:bg-slate-50 active:scale-[0.98]
              text-slate-500 py-2 rounded-xl text-xs font-medium transition-all"
          >
            View Last Quiz Results
          </button>
        )}
      </div>
    );
  }

  if (screen === "active") {
    return (
      <QuizActive
        question={questions[current]}
        current={current}
        total={questions.length}
        selected={selected}
        onSelect={setSelected}
        onConfirm={confirmAnswer}
        timeLeft={timeLeft}
        timeLimitSec={timeLimitSec}
        timerWarning={timerWarning}
        timerCritical={timerCritical}
      />
    );
  }

  return (
    <QuizResults
      score={score}
      total={answers.length}
      analyses={analyses}
      analyzing={analyzing}
      wrongAttemptIds={wrongAttemptIds}
      onRetry={reset}
    />
  );
}
```

> **Note**: Read `frontend/src/pages/QuizPage.tsx` first to see the current prop names passed to `QuizActive` — use the exact same props, just adding `wrongAttemptIds` and `restoreResults`.

- [ ] **Step 4: Run type-check**

```bash
cd frontend && npm run type-check
```
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useQuizMode.ts frontend/src/components/QuizResults.tsx frontend/src/pages/QuizPage.tsx
git commit -m "feat: quiz results sessionStorage persistence, per-wrong-answer generate explanation, wrongAttemptIds from record-batch"
```

---

### Task I: Animation stagger — remaining list renders

**Files:**
- Modify: `frontend/src/pages/StatsPage.tsx`

> HistoryPage and HistoryItem already fixed in Task C/D. QuizResults already has `animationDelay` inline. This task handles StatsPage weak concepts list.

- [ ] **Step 1: Read `frontend/src/pages/StatsPage.tsx`** to find the weak_concepts render

- [ ] **Step 2: Add stagger delay to weak concepts list in StatsPage**

Find the weak concepts render (look for `stats.weak_concepts.map`) and add index-based delay. Example — if it currently reads:
```typescript
{stats.weak_concepts.map(c => (
  <span key={c} className="...">...</span>
))}
```

Change to:
```typescript
{stats.weak_concepts.map((c, i) => (
  <span key={c} className="... animate-pop-in" style={{ animationDelay: `${i * 50}ms` }}>...</span>
))}
```

Match the exact existing class names and only add `animate-pop-in` + `style` where not already present.

- [ ] **Step 3: Run type-check**

```bash
cd frontend && npm run type-check
```
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/StatsPage.tsx
git commit -m "fix: stagger animation delay on StatsPage weak concepts list"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|---|---|
| History: show options when clicking item | Task C |
| History: show cached LLM explanation | Task C |
| History: generate explanation on demand | Task A (endpoint) + Task C (button) |
| History: top 10 weak grammar points | Task A (endpoint) + Task D (panel) |
| Quiz: set how many questions | Task G |
| Quiz: resume/view past quiz records | Task H (sessionStorage) |
| Quiz: click wrong question, generate explanation | Task A (endpoint) + Task H (button) |
| Practice → Ask rename | Task E |
| Ask: cache explanation when recording | Task F |
| Animation fix (no visible change) | Task C (stagger on HistoryItem), Task D (stagger on concepts), Task I (StatsPage) |

**Placeholder scan:** No TBD/TODO left. All code blocks are complete.

**Type consistency:**
- `wrongAttemptIds: number[]` used in `useQuizMode` return, `QuizResults` props, `QuizPage` — consistent
- `restoreResults(saved: PersistedResults)` matches `loadPersistedResults()` return type — consistent
- `api.explainAttempt(number)` → `Promise<{explanation: string}>` used in HistoryItem and QuizResults — consistent
- `api.getWeakConcepts(limit?)` → `Promise<{concepts: string[]}>` used in HistoryPage — consistent
- `WeakConceptsResponse` schema in backend matches `{concepts: string[]}` on frontend — consistent

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-05-history-quiz-ask-enhancements.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - Fresh subagent per task, two-stage review between tasks

**2. Inline Execution** - Execute tasks in this session using executing-plans

**Which approach?**
