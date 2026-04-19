# Quiz Mode, UI Refresh & Smart Recommendations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a timed CSV-based practice quiz mode with post-quiz LLM analysis, compact the UI with entrance animations, and replace keyword-only recommendation links with LLM-curated song/anime/article suggestions.

**Architecture:** Backend gains a `/api/v1/practice/*` router (CSV parse, wrong-history mix, batch analysis, bulk history record) and a `/api/v1/recommendations/media` endpoint using a new `complete()` method on LLMProvider. Frontend gains a `/quiz` route with a 3-screen flow (setup → active → results), animated compact components via Tailwind v4 custom utilities, and an upgraded RecommendationPanel showing LLM-generated media cards.

**Tech Stack:** Python FastAPI, SQLAlchemy async, existing LLMProvider abstraction; React 18, TypeScript 5, Tailwind CSS v4, React Router 6.

---

## File Map

### Backend (new / modified)
```
backend/
├── services/
│   ├── practice.py              # CSV parse, wrong-history sample, batch analysis
│   └── media_recommendations.py # LLM-curated media suggestions
├── services/llm/
│   ├── base.py                  # Add abstract complete() method
│   ├── openrouter.py            # Implement complete()
│   └── local.py                 # Implement complete()
├── routers/
│   ├── practice.py              # /api/v1/practice/* endpoints
│   └── advanced.py              # Add POST /api/v1/recommendations/media
├── schemas/api.py               # New request/response models
└── main.py                      # Mount practice router
```

### Frontend (new / modified)
```
frontend/src/
├── index.css                    # Custom keyframes + @utility animate-fade-in/slide-in
├── types/index.ts               # QuizQuestion, AnalysisItem, MediaRecommendResponse
├── api/client.ts                # uploadPracticeCSV, analyzePractice, recordBatch, getMediaRecommendations
├── hooks/
│   └── useQuizMode.ts           # Timer + quiz state machine
├── pages/
│   └── QuizPage.tsx             # Orchestrates setup/active/results screens
├── components/
│   ├── Layout.tsx               # Add "Quiz" nav link, compact nav
│   ├── QuizForm.tsx             # Compact spacing + hover animations
│   ├── ExplanationCard.tsx      # animate-fade-in entrance
│   ├── StreamingProgress.tsx    # Smoother animated steps
│   ├── HistoryItem.tsx          # Compact, animate-fade-in
│   ├── QuizSetup.tsx            # CSV upload + timer + history-mix config
│   ├── ActiveQuiz.tsx           # One question at a time + countdown
│   ├── QuizResults.tsx          # Score + per-wrong LLM explanation cards
│   └── RecommendationPanel.tsx  # LLM media cards (songs/anime/articles)
└── App.tsx                      # Add /quiz route
```

### Tests
```
tests/
├── test_practice_service.py
└── test_media_recommendations.py
```

---

### Task 1: Custom Animations + Compact Global Styles

**Files:**
- Modify: `frontend/src/index.css`

- [ ] **Step 1: Add keyframes and utility classes**

Replace the entire contents of `frontend/src/index.css`:

```css
@import "tailwindcss";

@theme {
  --color-brand-50: #f0f9ff;
  --color-brand-100: #e0f2fe;
  --color-brand-500: #0ea5e9;
  --color-brand-600: #0284c7;
  --color-brand-700: #0369a1;
}

@keyframes fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

@keyframes slide-in {
  from { opacity: 0; transform: translateX(-10px); }
  to   { opacity: 1; transform: translateX(0); }
}

@keyframes pop-in {
  0%   { opacity: 0; transform: scale(0.92); }
  100% { opacity: 1; transform: scale(1); }
}

@keyframes timer-pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.6; }
}

@utility animate-fade-in {
  animation: fade-in 0.22s ease-out both;
}

@utility animate-slide-in {
  animation: slide-in 0.18s ease-out both;
}

@utility animate-pop-in {
  animation: pop-in 0.2s ease-out both;
}

@utility animate-timer-pulse {
  animation: timer-pulse 1s ease-in-out infinite;
}
```

- [ ] **Step 2: Build and verify no errors**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npm run build 2>&1 | tail -6
```

Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
git add frontend/src/index.css
git commit -m "feat: custom fade/slide/pop animations via Tailwind v4 utilities"
```

---

### Task 2: Compact UI Component Pass

**Files:**
- Modify: `frontend/src/components/Layout.tsx`
- Modify: `frontend/src/components/QuizForm.tsx`
- Modify: `frontend/src/components/ExplanationCard.tsx`
- Modify: `frontend/src/components/StreamingProgress.tsx`
- Modify: `frontend/src/components/HistoryItem.tsx`

- [ ] **Step 1: Compact `Layout.tsx` + add Quiz nav link**

```typescript
import { type ReactNode } from "react";
import { NavLink, Navigate } from "react-router-dom";
import { useAuthContext } from "../contexts/AuthContext";

const NAV_ITEMS = [
  { to: "/practice", label: "Practice" },
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
```

- [ ] **Step 2: Compact `QuizForm.tsx` with hover animations**

```typescript
import type { KeyboardEvent } from "react";

interface Props {
  question: string;
  options: string[];
  onQuestionChange: (v: string) => void;
  onOptionChange: (i: number, v: string) => void;
  onSubmit: () => void;
  onReset: () => void;
  disabled: boolean;
  error: string;
}

export default function QuizForm({
  question, options, onQuestionChange, onOptionChange,
  onSubmit, onReset, disabled, error
}: Props) {
  const handleKey = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") onSubmit();
  };
  const canSubmit = question.trim() && options.filter(o => o.trim()).length >= 2;

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4 animate-fade-in">
      <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1">Question</label>
      <textarea
        value={question}
        onChange={e => onQuestionChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder="彼女は毎日日本語を＿＿＿います。"
        disabled={disabled}
        rows={2}
        className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm mb-3
          focus:outline-none focus:ring-2 focus:ring-brand-500 resize-none
          disabled:bg-slate-50 transition-colors"
      />

      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Options</p>
      <div className="grid grid-cols-2 gap-2 mb-4">
        {options.map((opt, i) => (
          <div key={i} className="flex items-center gap-1.5">
            <span className="text-xs font-bold text-slate-400 w-3">{String.fromCharCode(65 + i)}</span>
            <input
              value={opt}
              onChange={e => onOptionChange(i, e.target.value)}
              onKeyDown={handleKey}
              placeholder={`Option ${String.fromCharCode(65 + i)}`}
              disabled={disabled}
              className="flex-1 border border-slate-200 rounded-lg px-2.5 py-1.5 text-sm
                focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:bg-slate-50
                transition-colors hover:border-slate-300"
            />
          </div>
        ))}
      </div>

      {error && <p className="text-red-500 text-xs mb-2">{error}</p>}

      <div className="flex gap-2">
        <button onClick={onSubmit} disabled={disabled || !canSubmit}
          className="flex-1 bg-brand-500 hover:bg-brand-600 active:scale-95 text-white py-2 rounded-lg
            font-medium text-sm transition-all disabled:opacity-40"
        >
          {disabled ? "Analyzing…" : "Get Answer ⌘↵"}
        </button>
        <button onClick={onReset} disabled={disabled}
          className="px-3 py-2 border border-slate-200 rounded-lg text-sm text-slate-500
            hover:bg-slate-50 active:scale-95 transition-all disabled:opacity-40"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Animate `ExplanationCard.tsx`**

```typescript
import type { Phase, SolveResult } from "../types";

export default function ExplanationCard({
  streamText, result, phase
}: {
  streamText: string; result: SolveResult | null; phase: Phase;
}) {
  if (phase === "idle") return null;

  return (
    <div className="mt-4 space-y-3 animate-fade-in">
      {result ? (
        <>
          <div className="flex items-start gap-3 p-3 bg-green-50 border border-green-200 rounded-xl animate-pop-in">
            <span className="text-xl font-black text-green-700 shrink-0">{result.answer}</span>
            <p className="text-sm text-slate-700 leading-relaxed">{result.explanation}</p>
          </div>

          {Object.entries(result.wrong_options).length > 0 && (
            <div className="p-3 bg-slate-50 rounded-xl space-y-1.5">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Why others are wrong</p>
              {Object.entries(result.wrong_options).map(([opt, reason]) => (
                <div key={opt} className="flex gap-2 text-sm animate-slide-in">
                  <span className="font-bold text-red-400 w-4 shrink-0">{opt}</span>
                  <span className="text-slate-600 text-xs leading-relaxed">{reason}</span>
                </div>
              ))}
            </div>
          )}

          {result.concepts.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {result.concepts.map(c => (
                <span key={c} className="bg-brand-50 text-brand-700 text-xs px-2 py-0.5 rounded-full font-medium">
                  {c}
                </span>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="p-3 bg-white border border-slate-200 rounded-xl min-h-16">
          <p className="text-sm text-slate-600 whitespace-pre-wrap leading-relaxed">
            {streamText}<span className="animate-pulse">▌</span>
          </p>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Animate `StreamingProgress.tsx`**

```typescript
import type { Phase } from "../types";

const PHASES: { key: Phase; label: string }[] = [
  { key: "answering",  label: "Selecting" },
  { key: "explaining", label: "Explaining" },
  { key: "done",       label: "Done" },
];

const ORDER: Record<string, number> = { answering: 0, explaining: 1, done: 2 };

export default function StreamingProgress({ phase }: { phase: Phase }) {
  if (phase === "idle") return null;
  const cur = ORDER[phase] ?? -1;
  return (
    <div className="flex items-center gap-2 my-3 animate-slide-in">
      {PHASES.map(({ key, label }, i) => {
        const idx = ORDER[key];
        const done   = idx < cur;
        const active = idx === cur;
        return (
          <div key={key} className="flex items-center gap-2">
            {i > 0 && (
              <div className={`h-px w-5 transition-all duration-500 ${done ? "bg-brand-500" : "bg-slate-200"}`} />
            )}
            <span className={`text-xs font-semibold px-2 py-0.5 rounded-full transition-all duration-300
              ${done   ? "bg-brand-100 text-brand-700"
              : active ? "bg-brand-500 text-white animate-timer-pulse"
              : "bg-slate-100 text-slate-400"}`}>
              {label}
            </span>
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 5: Compact `HistoryItem.tsx`**

```typescript
import type { AttemptResponse } from "../types";

export default function HistoryItem({ attempt }: { attempt: AttemptResponse }) {
  const date = new Date(attempt.created_at).toLocaleDateString();
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-3 space-y-1.5 animate-fade-in
      hover:shadow-sm transition-shadow">
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm text-slate-800 flex-1 leading-snug">{attempt.question_text}</p>
        <span className={`text-xs font-bold px-2 py-0.5 rounded-full shrink-0
          ${attempt.user_marked_correct ? "bg-green-100 text-green-700" : "bg-red-100 text-red-600"}`}>
          {attempt.user_marked_correct ? "✓" : "✗"}
        </span>
      </div>
      <div className="flex items-center gap-3 text-xs text-slate-400">
        <span>Answer: <span className="font-semibold text-slate-600">{attempt.correct_answer}</span></span>
        <span>{date}</span>
      </div>
      {attempt.concepts.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {attempt.concepts.map(c => (
            <span key={c} className="bg-slate-100 text-slate-500 text-xs px-1.5 py-0.5 rounded-full">{c}</span>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Build and verify**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npm run build 2>&1 | tail -6
```

Expected: Build succeeds, 0 TypeScript errors.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
git add frontend/src/components/Layout.tsx frontend/src/components/QuizForm.tsx \
  frontend/src/components/ExplanationCard.tsx frontend/src/components/StreamingProgress.tsx \
  frontend/src/components/HistoryItem.tsx
git commit -m "feat: compact UI with entrance animations on all cards"
```

---

### Task 3: Backend Practice Service + Schemas

**Files:**
- Modify: `backend/schemas/api.py`
- Create: `backend/services/practice.py`
- Create: `tests/test_practice_service.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_practice_service.py`:

```python
import pytest
from backend.services.practice import parse_grammar_csv


def test_parse_basic_row():
    csv = "Question,Options,Answer\n今日は___寒い。,a.とても b.たいして c.かなり,b\n"
    qs = parse_grammar_csv(csv)
    assert len(qs) == 1
    assert qs[0]["question"] == "今日は___寒い。"
    assert qs[0]["correct_answer"] == "B"
    assert qs[0]["options"] == ["A: とても", "B: たいして", "C: かなり"]
    assert qs[0]["from_history"] is False


def test_parse_four_options():
    csv = "Question,Options,Answer\nTest Q,a.aaa b.bbb c.ccc d.ddd,d\n"
    qs = parse_grammar_csv(csv)
    assert len(qs) == 1
    assert qs[0]["correct_answer"] == "D"
    assert len(qs[0]["options"]) == 4


def test_parse_skips_empty_question():
    csv = "Question,Options,Answer\n,a.x b.y,a\n"
    qs = parse_grammar_csv(csv)
    assert qs == []


def test_parse_multiple_rows():
    csv = (
        "Question,Options,Answer\n"
        "Q1,a.aaa b.bbb c.ccc,a\n"
        "Q2,a.xxx b.yyy c.zzz,c\n"
    )
    qs = parse_grammar_csv(csv)
    assert len(qs) == 2
    assert qs[1]["correct_answer"] == "C"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
python -m pytest tests/test_practice_service.py -v 2>&1 | tail -10
```

Expected: ImportError (module not found yet).

- [ ] **Step 3: Add schemas to `backend/schemas/api.py`**

Append to the end of `backend/schemas/api.py`:

```python

class PracticeQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    from_history: bool = False


class UploadPracticeResponse(BaseModel):
    questions: list[PracticeQuestion]
    total: int


class WrongItem(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    user_answer: str


class AnalysisItem(BaseModel):
    question: str
    correct_answer: str
    user_answer: str
    explanation: str


class AnalyzeRequest(BaseModel):
    wrong_items: list[WrongItem]


class AnalyzeResponse(BaseModel):
    analyses: list[AnalysisItem]


class BatchAttemptItem(BaseModel):
    question_text: str
    options: list[str]
    correct_answer: str
    user_answer: str
    user_marked_correct: bool


class BatchRecordRequest(BaseModel):
    attempts: list[BatchAttemptItem]


class MediaRecommendRequest(BaseModel):
    concept: str


class SongRec(BaseModel):
    title: str
    artist: str


class AnimeRec(BaseModel):
    title: str
    scene: str


class ArticleRec(BaseModel):
    title: str
    keywords: str


class MediaRecommendResponse(BaseModel):
    songs: list[SongRec]
    anime: list[AnimeRec]
    articles: list[ArticleRec]
```

- [ ] **Step 4: Create `backend/services/practice.py`**

```python
import csv
import io
import json
import re

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.tables import Attempt
from backend.services.llm.base import LLMProvider


def parse_grammar_csv(content: str) -> list[dict]:
    """Parse CSV with columns Question, Options, Answer.
    Options format: 'a.text b.text c.text [d.text]'
    Answer format: 'a'/'b'/'c'/'d'
    Returns list of {question, options, correct_answer, from_history} dicts.
    """
    reader = csv.DictReader(io.StringIO(content))
    questions = []
    for row in reader:
        question = row.get("Question", "").strip()
        if not question:
            continue
        opts_raw = row.get("Options", "").strip()
        parts = re.split(r"\s+(?=[a-d]\.)", opts_raw)
        options = []
        for part in parts:
            m = re.match(r"^([a-d])\.(.+)$", part.strip())
            if m:
                options.append(f"{m.group(1).upper()}: {m.group(2).strip()}")
        if not options:
            continue
        answer = row.get("Answer", "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            continue
        questions.append({
            "question": question,
            "options": options,
            "correct_answer": answer,
            "from_history": False,
        })
    return questions


async def get_wrong_history_sample(
    db: AsyncSession, user_id: int, n: int
) -> list[dict]:
    """Return up to n randomly sampled wrong attempts from history."""
    result = await db.execute(
        select(Attempt)
        .where(Attempt.user_id == user_id, Attempt.user_marked_correct == False)  # noqa: E712
        .order_by(func.random())
        .limit(n)
    )
    attempts = result.scalars().all()
    return [
        {
            "question": a.question_text,
            "options": json.loads(a.options or "[]"),
            "correct_answer": a.correct_answer,
            "from_history": True,
        }
        for a in attempts
    ]


async def analyze_wrong_answers(
    wrong_items: list[dict], provider: LLMProvider
) -> list[dict]:
    """Batch-analyze wrong answers via LLM. Returns list of analysis dicts."""
    if not wrong_items:
        return []

    lines = []
    for i, item in enumerate(wrong_items, 1):
        opts = ", ".join(item["options"])
        lines.append(
            f'{i}. Q: {item["question"]}\n'
            f'   Options: {opts}\n'
            f'   Correct: {item["correct_answer"]}, User chose: {item["user_answer"]}'
        )

    prompt = (
        "You are a JLPT grammar teacher. For each wrong answer below, write 1-2 sentences "
        "explaining why the correct answer is right and why the user's choice was wrong.\n\n"
        + "\n\n".join(lines)
        + "\n\nRespond ONLY with a JSON array (no markdown fences):\n"
        '[{"question_index": 1, "explanation": "..."}, ...]'
    )

    raw = await provider.complete(prompt)
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    parsed: list[dict] = []
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = []

    results = []
    for i, item in enumerate(wrong_items, 1):
        explanation = next(
            (p["explanation"] for p in parsed if p.get("question_index") == i), ""
        )
        results.append({
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "user_answer": item["user_answer"],
            "explanation": explanation,
        })
    return results
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
python -m pytest tests/test_practice_service.py -v
```

Expected: `4 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/services/practice.py backend/schemas/api.py tests/test_practice_service.py
git commit -m "feat: CSV practice service with parse, wrong-history sample, batch analysis"
```

---

### Task 4: LLM `complete()` Method + Practice Router

**Files:**
- Modify: `backend/services/llm/base.py`
- Modify: `backend/services/llm/openrouter.py`
- Modify: `backend/services/llm/local.py`
- Create: `backend/routers/practice.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Add `complete()` to `backend/services/llm/base.py`**

Add after the `generate_question` abstract method:

```python
    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Non-streaming completion for structured tasks (analysis, recommendations)."""
        ...
```

Full updated file `backend/services/llm/base.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class SolveResult:
    answer: str
    explanation: str
    wrong_options: dict[str, str]
    concepts: list[str]


@dataclass
class GeneratedQuestion:
    question: str
    options: list[str]
    correct_answer: str
    explanation: str
    concepts: list[str]


class LLMProvider(ABC):
    @abstractmethod
    async def stream_solve(
        self,
        question: str,
        options: list[str],
    ) -> AsyncIterator[str]:
        """Yield raw text tokens ending with <RESULT>{json}</RESULT>."""
        ...

    @abstractmethod
    async def generate_question(
        self,
        concept: str,
        level: str,
    ) -> GeneratedQuestion:
        """Return a fully formed question with answer and explanation."""
        ...

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Non-streaming completion for structured tasks."""
        ...
```

- [ ] **Step 2: Implement `complete()` in `backend/services/llm/openrouter.py`**

Add this method inside the `OpenRouterProvider` class after `generate_question`:

```python
    async def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        full = ""
        async for token in self._stream_chat(messages):
            full += token
        return full
```

- [ ] **Step 3: Implement `complete()` in `backend/services/llm/local.py`**

Add this method inside the `LocalModelProvider` class after `generate_question`:

```python
    async def complete(self, prompt: str) -> str:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._generate_sync, prompt, 1000
        )
```

- [ ] **Step 4: Create `backend/routers/practice.py`**

```python
import json

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.auth.dependencies import get_current_user
from backend.models.tables import User
from backend.schemas.api import (
    UploadPracticeResponse, PracticeQuestion,
    AnalyzeRequest, AnalyzeResponse, AnalysisItem,
    BatchRecordRequest,
)
from backend.services.practice import (
    parse_grammar_csv,
    get_wrong_history_sample,
    analyze_wrong_answers,
)
from backend.services.analytics import record_attempt
from backend.services.llm.factory import get_llm_provider

router = APIRouter(prefix="/api/v1/practice")


@router.post("/upload", response_model=UploadPracticeResponse)
async def upload_csv(
    file: UploadFile = File(...),
    include_history: bool = Form(False),
    history_count: int = Form(5),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    content = (await file.read()).decode("utf-8", errors="replace")
    questions = parse_grammar_csv(content)
    if not questions:
        raise HTTPException(status_code=422, detail="No valid questions found in CSV")

    if include_history and history_count > 0:
        history_qs = await get_wrong_history_sample(db, current_user.id, history_count)
        questions = questions + history_qs  # history appended at end

    return UploadPracticeResponse(
        questions=[PracticeQuestion(**q) for q in questions],
        total=len(questions),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    body: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
):
    provider = get_llm_provider()
    wrong_dicts = [item.model_dump() for item in body.wrong_items]
    analyses = await analyze_wrong_answers(wrong_dicts, provider)
    return AnalyzeResponse(
        analyses=[AnalysisItem(**a) for a in analyses]
    )


@router.post("/record-batch", status_code=201)
async def record_batch(
    body: BatchRecordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    for item in body.attempts:
        await record_attempt(
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
    return {"recorded": len(body.attempts)}
```

- [ ] **Step 5: Mount practice router in `backend/main.py`**

Add import and router mount. Full updated `backend/main.py`:

```python
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.database import init_db
from backend.routers import auth as auth_router
from backend.routers import quiz as quiz_router
from backend.routers import history as history_router
from backend.routers import advanced as advanced_router
from backend.routers import practice as practice_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="JLPT Sensei", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(quiz_router.router)
app.include_router(history_router.router)
app.include_router(advanced_router.router)
app.include_router(practice_router.router)


@app.get("/health/live")
async def health_live():
    return {"status": "ok"}


# Serve built frontend if dist/ exists
dist_path = Path(__file__).parent.parent / "frontend" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="frontend")
```

- [ ] **Step 6: Run unit tests to check nothing broken**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py 2>&1 | tail -10
```

Expected: all existing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add backend/services/llm/base.py backend/services/llm/openrouter.py \
  backend/services/llm/local.py backend/routers/practice.py backend/main.py
git commit -m "feat: LLM complete() method and practice upload/analyze/record-batch router"
```

---

### Task 5: Frontend Types + API Client + useQuizMode Hook

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/api/client.ts`
- Create: `frontend/src/hooks/useQuizMode.ts`

- [ ] **Step 1: Add new types to `frontend/src/types/index.ts`**

Append to end of `frontend/src/types/index.ts`:

```typescript
export interface QuizQuestion {
  question: string;
  options: string[];
  correct_answer: string;
  from_history?: boolean;
}

export interface AnalysisItem {
  question: string;
  correct_answer: string;
  user_answer: string;
  explanation: string;
}

export interface MediaRecommendResponse {
  songs: { title: string; artist: string }[];
  anime: { title: string; scene: string }[];
  articles: { title: string; keywords: string }[];
}
```

- [ ] **Step 2: Add API methods to `frontend/src/api/client.ts`**

Add these methods to the `api` object in `frontend/src/api/client.ts` (append before the closing `}`):

```typescript
  uploadPracticeCSV: (
    file: File,
    includeHistory: boolean,
    historyCount: number
  ) => {
    const form = new FormData();
    form.append("file", file);
    form.append("include_history", String(includeHistory));
    form.append("history_count", String(historyCount));
    return fetch(`${BASE}/api/v1/practice/upload`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    }).then(async r => {
      if (!r.ok) { const e = await r.json().catch(() => ({ detail: "Upload failed" })); throw new Error(e.detail); }
      return r.json() as Promise<{ questions: import("../types").QuizQuestion[]; total: number }>;
    });
  },

  analyzePractice: (wrongItems: {
    question: string; options: string[]; correct_answer: string; user_answer: string;
  }[]) =>
    post<{ analyses: import("../types").AnalysisItem[] }>(
      "/api/v1/practice/analyze",
      { wrong_items: wrongItems },
      true
    ),

  recordBatch: (attempts: {
    question_text: string; options: string[]; correct_answer: string;
    user_answer: string; user_marked_correct: boolean;
  }[]) =>
    post("/api/v1/practice/record-batch", { attempts }, true),

  getMediaRecommendations: (concept: string) =>
    post<import("../types").MediaRecommendResponse>(
      "/api/v1/recommendations/media",
      { concept },
      true
    ),
```

- [ ] **Step 3: Create `frontend/src/hooks/useQuizMode.ts`**

```typescript
import { useState, useCallback, useRef, useEffect } from "react";
import { api } from "../api/client";
import type { QuizQuestion, AnalysisItem } from "../types";

export type QuizScreen = "setup" | "active" | "results";

interface QuizAnswer {
  questionIndex: number;
  userAnswer: string;
  correctAnswer: string;
  isCorrect: boolean;
  question: string;
  options: string[];
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
  const [analyzing, setAnalyzing] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

    // Record all attempts to history
    await api.recordBatch(
      finalAnswers.map(a => ({
        question_text: a.question,
        options: a.options,
        correct_answer: a.correctAnswer,
        user_answer: a.userAnswer,
        user_marked_correct: a.isCorrect,
      }))
    ).catch(() => {/* non-fatal */});

    // Get LLM analysis for wrong answers
    if (wrong.length > 0) {
      try {
        const res = await api.analyzePractice(wrong);
        setAnalyses(res.analyses);
      } catch { setAnalyses([]); }
    }
    setAnalyzing(false);
  }, []);

  const startQuiz = useCallback((qs: QuizQuestion[], timeSec: number) => {
    setQuestions(qs);
    setCurrent(0);
    setSelected(null);
    setAnswers([]);
    setAnalyses([]);
    setTimeLimitSec(timeSec);
    setTimeLeft(timeSec);
    setScreen("active");
  }, []);

  // Countdown timer
  useEffect(() => {
    if (screen !== "active" || timeLimitSec === 0) return;
    timerRef.current = setInterval(() => {
      setTimeLeft(t => {
        if (t <= 1) {
          // time's up — finish with current answers
          setAnswers(prev => {
            finishQuiz(prev);
            return prev;
          });
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
    setAnswers([]);
    setAnalyses([]);
  };

  const score = answers.filter(a => a.isCorrect).length;
  const timerWarning = timeLimitSec > 0 && timeLeft < 60;
  const timerCritical = timeLimitSec > 0 && timeLeft < 30;

  return {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset,
  };
}
```

- [ ] **Step 4: Build check**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npm run build 2>&1 | tail -6
```

Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
git add frontend/src/types/index.ts frontend/src/api/client.ts frontend/src/hooks/useQuizMode.ts
git commit -m "feat: quiz mode types, API client methods, useQuizMode hook with timer"
```

---

### Task 6: QuizSetup + ActiveQuiz Components

**Files:**
- Create: `frontend/src/components/QuizSetup.tsx`
- Create: `frontend/src/components/ActiveQuiz.tsx`

- [ ] **Step 1: Create `frontend/src/components/QuizSetup.tsx`**

```typescript
import { useRef, useState } from "react";
import { api } from "../api/client";
import type { QuizQuestion } from "../types";

interface Props {
  onStart: (questions: QuizQuestion[], timeSec: number) => void;
}

export default function QuizSetup({ onStart }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState("");
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [minutes, setMinutes] = useState(10);
  const [includeHistory, setIncludeHistory] = useState(false);
  const [historyCount, setHistoryCount] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFile = async (file: File) => {
    setLoading(true); setError(""); setQuestions([]);
    setFileName(file.name);
    try {
      const res = await api.uploadPracticeCSV(file, includeHistory, historyCount);
      setQuestions(res.questions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to parse CSV");
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  return (
    <div className="space-y-4 animate-fade-in">
      <div>
        <h1 className="text-lg font-bold text-slate-800">Timed Quiz</h1>
        <p className="text-xs text-slate-400 mt-0.5">Upload a grammar CSV to start a timed practice session</p>
      </div>

      {/* CSV Upload */}
      <div
        onDrop={handleDrop}
        onDragOver={e => e.preventDefault()}
        onClick={() => inputRef.current?.click()}
        className="border-2 border-dashed border-slate-200 rounded-xl p-6 text-center cursor-pointer
          hover:border-brand-500 transition-colors active:scale-99"
      >
        <input ref={inputRef} type="file" accept=".csv" className="hidden"
          onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} />
        {loading ? (
          <p className="text-sm text-brand-500 animate-timer-pulse">Parsing CSV…</p>
        ) : fileName ? (
          <div>
            <p className="text-sm font-medium text-slate-700">{fileName}</p>
            <p className="text-xs text-green-600 mt-1">{questions.length} questions loaded</p>
          </div>
        ) : (
          <div>
            <p className="text-sm text-slate-400">Drop CSV here or click to upload</p>
            <p className="text-xs text-slate-300 mt-1">Format: Question, Options, Answer</p>
          </div>
        )}
      </div>
      {error && <p className="text-red-500 text-xs">{error}</p>}

      {/* Config */}
      <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-3">
        {/* Timer */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-slate-700">Time limit</p>
            <p className="text-xs text-slate-400">Set 0 for unlimited</p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setMinutes(m => Math.max(0, m - 5))}
              className="w-7 h-7 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50
                active:scale-95 transition-all text-sm flex items-center justify-center">−</button>
            <span className="text-sm font-semibold text-slate-800 w-12 text-center">
              {minutes === 0 ? "∞" : `${minutes}m`}
            </span>
            <button onClick={() => setMinutes(m => m + 5)}
              className="w-7 h-7 rounded-lg border border-slate-200 text-slate-500 hover:bg-slate-50
                active:scale-95 transition-all text-sm flex items-center justify-center">+</button>
          </div>
        </div>

        {/* Include wrong history */}
        <div className="flex items-center justify-between border-t border-slate-100 pt-3">
          <div>
            <p className="text-sm font-medium text-slate-700">Mix in wrong history</p>
            <p className="text-xs text-slate-400">Add past wrong answers to practice</p>
          </div>
          <button
            onClick={() => setIncludeHistory(v => !v)}
            className={`relative w-10 h-5 rounded-full transition-colors ${includeHistory ? "bg-brand-500" : "bg-slate-200"}`}
          >
            <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform
              ${includeHistory ? "translate-x-5" : "translate-x-0.5"}`} />
          </button>
        </div>

        {includeHistory && (
          <div className="flex items-center justify-between animate-fade-in">
            <p className="text-xs text-slate-500">How many wrong questions to add</p>
            <div className="flex items-center gap-2">
              <button onClick={() => setHistoryCount(n => Math.max(1, n - 1))}
                className="w-6 h-6 rounded border border-slate-200 text-xs text-slate-500
                  hover:bg-slate-50 active:scale-95 transition-all">−</button>
              <span className="text-sm font-semibold text-slate-800 w-6 text-center">{historyCount}</span>
              <button onClick={() => setHistoryCount(n => Math.min(20, n + 1))}
                className="w-6 h-6 rounded border border-slate-200 text-xs text-slate-500
                  hover:bg-slate-50 active:scale-95 transition-all">+</button>
            </div>
          </div>
        )}
      </div>

      <button
        disabled={questions.length === 0}
        onClick={() => onStart(questions, minutes * 60)}
        className="w-full bg-brand-500 hover:bg-brand-600 active:scale-98 text-white py-2.5 rounded-xl
          font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {questions.length > 0 ? `Start Quiz — ${questions.length} questions` : "Upload CSV to start"}
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Create `frontend/src/components/ActiveQuiz.tsx`**

```typescript
import type { QuizQuestion } from "../types";

interface Props {
  question: QuizQuestion;
  questionNumber: number;
  totalQuestions: number;
  selected: string | null;
  onSelect: (answer: string) => void;
  onConfirm: () => void;
  timeLeft: number;
  timeLimitSec: number;
  timerWarning: boolean;
  timerCritical: boolean;
}

function formatTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function ActiveQuiz({
  question, questionNumber, totalQuestions, selected, onSelect, onConfirm,
  timeLeft, timeLimitSec, timerWarning, timerCritical,
}: Props) {
  const progress = ((questionNumber - 1) / totalQuestions) * 100;

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400">
          {questionNumber} / {totalQuestions}
        </span>
        {timeLimitSec > 0 && (
          <span className={`text-sm font-bold tabular-nums transition-colors
            ${timerCritical ? "text-red-500 animate-timer-pulse"
            : timerWarning  ? "text-amber-500"
            : "text-slate-500"}`}>
            ⏱ {formatTime(timeLeft)}
          </span>
        )}
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-brand-500 rounded-full transition-all duration-500"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Question */}
      <div className="bg-white border border-slate-200 rounded-xl p-4">
        <p className="text-base text-slate-800 leading-relaxed font-medium">
          {question.question}
        </p>
        {question.from_history && (
          <span className="inline-block mt-2 text-xs bg-amber-50 text-amber-600 px-2 py-0.5 rounded-full">
            from history
          </span>
        )}
      </div>

      {/* Options */}
      <div className="space-y-2">
        {question.options.map((opt) => {
          const letter = opt.split(":")[0].trim();
          const isSelected = selected === letter;
          return (
            <button
              key={letter}
              onClick={() => onSelect(letter)}
              className={`w-full text-left p-3 rounded-xl border text-sm transition-all
                active:scale-99 font-medium
                ${isSelected
                  ? "border-brand-500 bg-brand-50 text-brand-700 shadow-sm"
                  : "border-slate-200 bg-white text-slate-700 hover:border-slate-300 hover:bg-slate-50"
                }`}
            >
              {opt}
            </button>
          );
        })}
      </div>

      <button
        onClick={onConfirm}
        disabled={!selected}
        className="w-full bg-brand-500 hover:bg-brand-600 active:scale-98 text-white py-2.5
          rounded-xl font-semibold text-sm transition-all disabled:opacity-40"
      >
        {questionNumber === totalQuestions ? "Finish Quiz" : "Next →"}
      </button>
    </div>
  );
}
```

- [ ] **Step 3: Build check**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npm run build 2>&1 | tail -6
```

Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
git add frontend/src/components/QuizSetup.tsx frontend/src/components/ActiveQuiz.tsx
git commit -m "feat: QuizSetup and ActiveQuiz components with timer and option cards"
```

---

### Task 7: QuizResults + QuizPage + App Router

**Files:**
- Create: `frontend/src/components/QuizResults.tsx`
- Create: `frontend/src/pages/QuizPage.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create `frontend/src/components/QuizResults.tsx`**

```typescript
import type { AnalysisItem } from "../types";

interface Props {
  score: number;
  total: number;
  analyses: AnalysisItem[];
  analyzing: boolean;
  onRetry: () => void;
}

export default function QuizResults({ score, total, analyses, analyzing, onRetry }: Props) {
  const pct = total > 0 ? Math.round((score / total) * 100) : 0;
  const wrong = total - score;

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
          ) : (
            analyses.map((item, i) => (
              <div key={i} className="bg-white border border-red-100 rounded-xl p-3 space-y-2 animate-slide-in"
                style={{ animationDelay: `${i * 60}ms` }}>
                <p className="text-sm text-slate-800 leading-snug">{item.question}</p>
                <div className="flex gap-3 text-xs">
                  <span className="text-red-500 font-semibold">You: {item.user_answer}</span>
                  <span className="text-green-600 font-semibold">Correct: {item.correct_answer}</span>
                </div>
                {item.explanation && (
                  <p className="text-xs text-slate-600 leading-relaxed border-t border-slate-100 pt-2">
                    {item.explanation}
                  </p>
                )}
              </div>
            ))
          )}
        </div>
      )}

      <button
        onClick={onRetry}
        className="w-full border border-slate-200 hover:bg-slate-50 active:scale-98
          text-slate-600 py-2.5 rounded-xl text-sm font-medium transition-all"
      >
        ← Back to Setup
      </button>
    </div>
  );
}
```

- [ ] **Step 2: Create `frontend/src/pages/QuizPage.tsx`**

```typescript
import { useQuizMode } from "../hooks/useQuizMode";
import QuizSetup from "../components/QuizSetup";
import ActiveQuiz from "../components/ActiveQuiz";
import QuizResults from "../components/QuizResults";

export default function QuizPage() {
  const {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset,
  } = useQuizMode();

  return (
    <div>
      {screen === "setup" && (
        <QuizSetup onStart={startQuiz} />
      )}

      {screen === "active" && questions[current] && (
        <ActiveQuiz
          question={questions[current]}
          questionNumber={current + 1}
          totalQuestions={questions.length}
          selected={selected}
          onSelect={setSelected}
          onConfirm={confirmAnswer}
          timeLeft={timeLeft}
          timeLimitSec={timeLimitSec}
          timerWarning={timerWarning}
          timerCritical={timerCritical}
        />
      )}

      {screen === "results" && (
        <QuizResults
          score={score}
          total={answers.length}
          analyses={analyses}
          analyzing={analyzing}
          onRetry={reset}
        />
      )}
    </div>
  );
}
```

- [ ] **Step 3: Update `frontend/src/App.tsx`**

```typescript
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import LoginPage from "./pages/LoginPage";
import Layout from "./components/Layout";
import PracticePage from "./pages/PracticePage";
import HistoryPage from "./pages/HistoryPage";
import StatsPage from "./pages/StatsPage";
import QuizPage from "./pages/QuizPage";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/practice" element={<Layout><PracticePage /></Layout>} />
          <Route path="/quiz"     element={<Layout><QuizPage /></Layout>} />
          <Route path="/history"  element={<Layout><HistoryPage /></Layout>} />
          <Route path="/stats"    element={<Layout><StatsPage /></Layout>} />
          <Route path="*" element={<Navigate to="/practice" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
```

- [ ] **Step 4: Build and verify**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npm run build 2>&1 | tail -6
```

Expected: Build succeeds, 0 TypeScript errors.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
git add frontend/src/components/QuizResults.tsx frontend/src/pages/QuizPage.tsx frontend/src/App.tsx
git commit -m "feat: QuizResults with LLM analysis cards, QuizPage, /quiz route"
```

---

### Task 8: LLM Media Recommendations Backend

**Files:**
- Create: `backend/services/media_recommendations.py`
- Modify: `backend/routers/advanced.py`
- Create: `tests/test_media_recommendations.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_media_recommendations.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_get_media_recommendations_parses_llm_json():
    from backend.services.media_recommendations import get_media_recommendations

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value='''
    {
      "songs": [{"title": "千本桜", "artist": "黒うさP"}],
      "anime": [{"title": "進撃の巨人", "scene": "第1話 誓いのシーン"}],
      "articles": [{"title": "て形の使い方", "keywords": "JLPT N4 て形 文法"}]
    }
    ''')

    result = await get_media_recommendations("て-form", mock_provider)
    assert len(result["songs"]) == 1
    assert result["songs"][0]["title"] == "千本桜"
    assert len(result["anime"]) == 1
    assert len(result["articles"]) == 1


@pytest.mark.asyncio
async def test_get_media_recommendations_handles_bad_json():
    from backend.services.media_recommendations import get_media_recommendations

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value="not json at all")

    result = await get_media_recommendations("て-form", mock_provider)
    assert result == {"songs": [], "anime": [], "articles": []}
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
python -m pytest tests/test_media_recommendations.py -v 2>&1 | tail -6
```

Expected: ImportError.

- [ ] **Step 3: Create `backend/services/media_recommendations.py`**

```python
import json
import re

from backend.services.llm.base import LLMProvider

MEDIA_PROMPT = """You are a JLPT study resource curator.
For the Japanese grammar point '{concept}', suggest real, famous Japanese media that naturally uses this grammar.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "songs": [{{"title": "...", "artist": "..."}}],
  "anime": [{{"title": "...", "scene": "..."}}],
  "articles": [{{"title": "...", "keywords": "..."}}]
}}

Rules:
- Suggest 2 songs, 2 anime scenes, 2 articles
- Songs and anime must be real and famous
- Article keywords should be Japanese search terms useful for JLPT study
- All titles in Japanese where applicable"""


async def get_media_recommendations(concept: str, provider: LLMProvider) -> dict:
    prompt = MEDIA_PROMPT.format(concept=concept)
    raw = await provider.complete(prompt)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"songs": [], "anime": [], "articles": []}
    try:
        data = json.loads(match.group(0))
        return {
            "songs":    data.get("songs", []),
            "anime":    data.get("anime", []),
            "articles": data.get("articles", []),
        }
    except json.JSONDecodeError:
        return {"songs": [], "anime": [], "articles": []}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
python -m pytest tests/test_media_recommendations.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Add `POST /api/v1/recommendations/media` to `backend/routers/advanced.py`**

Full updated `backend/routers/advanced.py`:

```python
from fastapi import APIRouter, Depends, Query

from backend.config import settings
from backend.schemas.api import (
    VideoRecommendation, GenerateRequest, GeneratedQuestionResponse,
    MediaRecommendRequest, MediaRecommendResponse, SongRec, AnimeRec, ArticleRec,
)
from backend.services.recommendations import search_youtube
from backend.services.media_recommendations import get_media_recommendations
from backend.services.llm.factory import get_llm_provider
from backend.auth.dependencies import get_current_user
from backend.models.tables import User

router = APIRouter(prefix="/api/v1")


@router.get("/recommendations", response_model=list[VideoRecommendation])
async def get_recommendations(
    concepts: str = Query(..., description="Comma-separated concept list"),
    current_user: User = Depends(get_current_user),
):
    concept_list = [c.strip() for c in concepts.split(",") if c.strip()]
    if not concept_list:
        return []
    videos = await search_youtube(concept_list[0], settings.youtube_api_key)
    return [
        VideoRecommendation(
            title=v.title,
            video_id=v.video_id,
            thumbnail_url=v.thumbnail_url,
            channel_title=v.channel_title,
        )
        for v in videos
    ]


@router.post("/recommendations/media", response_model=MediaRecommendResponse)
async def media_recommendations(
    body: MediaRecommendRequest,
    current_user: User = Depends(get_current_user),
):
    provider = get_llm_provider()
    data = await get_media_recommendations(body.concept, provider)
    return MediaRecommendResponse(
        songs=[SongRec(**s) for s in data.get("songs", [])],
        anime=[AnimeRec(**a) for a in data.get("anime", [])],
        articles=[ArticleRec(**ar) for ar in data.get("articles", [])],
    )


@router.post("/generate", response_model=GeneratedQuestionResponse)
async def generate_question(
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
):
    provider = get_llm_provider()
    q = await provider.generate_question(concept=body.concept, level=body.level)
    return GeneratedQuestionResponse(
        question=q.question,
        options=q.options,
        correct_answer=q.correct_answer,
        explanation=q.explanation,
        concepts=q.concepts,
    )
```

- [ ] **Step 6: Run all tests**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py 2>&1 | tail -12
```

Expected: all tests pass (18 passed).

- [ ] **Step 7: Commit**

```bash
git add backend/services/media_recommendations.py backend/routers/advanced.py \
  tests/test_media_recommendations.py
git commit -m "feat: LLM media recommendations service (songs, anime, articles)"
```

---

### Task 9: Rich RecommendationPanel with LLM Media Cards

**Files:**
- Modify: `frontend/src/components/RecommendationPanel.tsx`

- [ ] **Step 1: Rewrite `frontend/src/components/RecommendationPanel.tsx`**

```typescript
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { MediaRecommendResponse } from "../types";

const PLATFORMS = [
  { name: "TikTok", bg: "bg-black", text: "text-white",
    url: (q: string) => `https://www.tiktok.com/search?q=${encodeURIComponent(q)}` },
  { name: "Lemon8", bg: "bg-yellow-400", text: "text-black",
    url: (q: string) => `https://www.lemon8-app.com/search/result?keyword=${encodeURIComponent(q)}` },
  { name: "Spotify", bg: "bg-green-500", text: "text-white",
    url: (q: string) => `https://open.spotify.com/search/${encodeURIComponent(q)}/podcasts` },
];

interface MediaCardProps {
  label: string;
  query: string;
  sub?: string;
  delay: number;
}

function MediaCard({ label, query, sub, delay }: MediaCardProps) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-3 space-y-2 animate-slide-in hover:shadow-sm transition-shadow"
      style={{ animationDelay: `${delay}ms` }}>
      <p className="text-xs font-semibold text-slate-700 leading-snug">{label}</p>
      {sub && <p className="text-xs text-slate-400">{sub}</p>}
      <div className="flex flex-wrap gap-1.5">
        {PLATFORMS.map(p => (
          <a key={p.name} href={p.url(query)} target="_blank" rel="noreferrer"
            className={`inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full
              font-medium transition-opacity hover:opacity-75 ${p.bg} ${p.text}`}>
            {p.name}
          </a>
        ))}
      </div>
    </div>
  );
}

export default function RecommendationPanel({ concepts }: { concepts: string[] }) {
  const [data, setData] = useState<MediaRecommendResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (concepts.length === 0) return;
    setLoading(true); setData(null);
    api.getMediaRecommendations(concepts[0])
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [concepts.join(",")]);

  if (concepts.length === 0) return null;

  return (
    <div className="mt-4 space-y-3 animate-fade-in">
      <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
        Study resources for: <span className="text-brand-600 normal-case">{concepts[0]}</span>
      </p>

      {loading && (
        <div className="space-y-2">
          {[0, 1, 2].map(i => (
            <div key={i} className="h-16 bg-slate-100 rounded-xl animate-timer-pulse"
              style={{ animationDelay: `${i * 100}ms` }} />
          ))}
        </div>
      )}

      {data && !loading && (
        <>
          {data.songs.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-slate-400">🎵 Songs</p>
              {data.songs.map((s, i) => (
                <MediaCard key={i} label={s.title} sub={s.artist}
                  query={`${s.title} ${s.artist} Japanese`} delay={i * 60} />
              ))}
            </div>
          )}

          {data.anime.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-slate-400">🎌 Anime</p>
              {data.anime.map((a, i) => (
                <MediaCard key={i} label={a.title} sub={a.scene}
                  query={`${a.title} ${a.scene}`} delay={i * 60} />
              ))}
            </div>
          )}

          {data.articles.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-slate-400">📖 Articles</p>
              {data.articles.map((ar, i) => (
                <MediaCard key={i} label={ar.title} query={ar.keywords} delay={i * 60} />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Build and verify**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npm run build 2>&1 | tail -6
```

Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
cd C:/Users/Andrew/Desktop/Projects/NASC_2024
git add frontend/src/components/RecommendationPanel.tsx
git commit -m "feat: rich recommendation panel with LLM-curated songs, anime, articles"
```

---

## Self-Review

### Spec Coverage

| Requirement | Task |
|---|---|
| Compact UI + animations | Task 1 (CSS), Task 2 (components) |
| CSV upload practice mode | Task 3 (service), Task 4 (router), Task 6 (QuizSetup) |
| Timed quiz (set time) | Task 5 (useQuizMode timer), Task 6 (ActiveQuiz) |
| Wrong-answer analysis at end | Task 3 (analyze_wrong_answers), Task 7 (QuizResults) |
| History updated after quiz | Task 4 (record-batch endpoint), Task 5 (useQuizMode records on finish) |
| Include random wrong history questions | Task 3 (get_wrong_history_sample), Task 4 (/upload), Task 6 (QuizSetup toggle) |
| LLM media recommendations (songs/anime/articles) | Task 8 (service + endpoint), Task 9 (panel) |
| TikTok/Lemon8/Spotify links from LLM queries | Task 9 (MediaCard with platform links) |

### No Placeholders Found
All steps contain complete code, exact commands, and expected output.

### Type Consistency
- `QuizQuestion` defined in Task 5 `types/index.ts`, consumed in `useQuizMode` (Task 5), `QuizSetup` (Task 6), `ActiveQuiz` (Task 6), `QuizPage` (Task 7) ✓
- `AnalysisItem` defined in Task 5 `types/index.ts`, returned by `analyzePractice` API call, consumed by `QuizResults` (Task 7) ✓
- `MediaRecommendResponse` defined in Task 5 `types/index.ts`, returned by `getMediaRecommendations` API, consumed by `RecommendationPanel` (Task 9) ✓
- `PracticeQuestion.options` is `list[str]` backend / `string[]` frontend — both consistent ✓
- `BatchAttemptItem.user_answer` vs `AttemptRecord` (existing): existing record uses `llm_answer` field; batch record maps `user_answer` → `llm_answer` in Task 4 router ✓
