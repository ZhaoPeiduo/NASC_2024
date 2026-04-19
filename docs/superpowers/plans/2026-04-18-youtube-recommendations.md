# YouTube Grammar Recommendations + Discover Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two YouTube-powered features: (1) per-wrong-answer tutorial video recommendations shown in quiz results, and (2) a "Discover" tab with a "Hit Me" button that plays a random grammar-relevant Japanese song.

**Architecture:** A new backend service `youtube_query.py` provides LLM-generated search queries; two new FastAPI endpoints (`POST /api/v1/recommendations/for-wrong-answers` and `GET /api/v1/youtube/random-song`) call the existing `search_youtube()` helper with these queries. The frontend adds a `YouTubeCard` component (click-to-embed), integrates it into `QuizResults` as an on-demand tutorial section, and adds a `DiscoverPage` behind a new `/discover` route. No new npm packages are needed — YouTube embeds are plain `<iframe>` elements.

**Tech Stack:** FastAPI async, httpx (already used), YouTube Data API v3, React 18, TypeScript, Tailwind CSS v4, existing LLMProvider.complete() interface

---

## Stakeholder Notes

**UIUX Designer:** `YouTubeCard` uses a thumbnail-first layout — clicking the image swaps to the inline `<iframe>` embed. `DiscoverPage` is centered vertically with a big terracotta "Hit Me" button and the video appearing beneath. Matches the existing warm parchment design system (ivory cards, cream borders, brand-500 CTAs, serif headings). No modal overlays, no new UI primitives.

**Frontend Engineer:** New component `YouTubeCard` handles click-to-play. `QuizResults` gets a new `wrongAnswers` prop (derived from the quiz `answers` array in `useQuizMode`) and renders an on-demand "Find Tutorials" section — user must click to trigger the API call (preserves quota). `DiscoverPage` is a standalone page. Two new methods added to `api` client.

**Product Manager:** Tutorial videos are loaded on-demand (not auto-fetched) to save YouTube API quota (100 units/search × max 3 searches = 300 units per quiz). The "Hit Me" endpoint picks from grammar-themed search queries, returning a random video each time. Requires `YOUTUBE_API_KEY` in `.env` — feature gracefully degrades (empty state) when key is absent.

**Backend Engineer:** `youtube_query.py` is the only new service file. It adds `generate_tutorial_query()` (LLM → search string) and `get_random_japanese_song()` (random query → YouTube search). Both call existing infrastructure: `LLMProvider.complete()` and `search_youtube()`. Two new routes added to `backend/routers/advanced.py`. Three new Pydantic schemas in `backend/schemas/api.py`. No DB changes.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/services/youtube_query.py` | CREATE | LLM query generator + random song query logic |
| `backend/schemas/api.py` | MODIFY | 3 new schemas: `WrongItemRec`, `VideoTutorialSet`, `WrongAnswerRecsResponse` |
| `backend/routers/advanced.py` | MODIFY | 2 new endpoints: `POST /for-wrong-answers`, `GET /random-song` |
| `tests/test_youtube_query.py` | CREATE | Unit tests for `generate_tutorial_query` and `get_random_japanese_song` |
| `frontend/src/types/index.ts` | MODIFY | 2 new types: `VideoTutorialSet`, `WrongAnswerRecsResponse` |
| `frontend/src/api/client.ts` | MODIFY | 2 new methods: `getRecsForWrongAnswers`, `getRandomSong` |
| `frontend/src/components/YouTubeCard.tsx` | CREATE | Thumbnail → click-to-embed YouTube player card |
| `frontend/src/components/QuizResults.tsx` | MODIFY | Add `wrongAnswers` prop + on-demand tutorial section |
| `frontend/src/hooks/useQuizMode.ts` | MODIFY | Expose `wrongAnswers` derived from `answers` array |
| `frontend/src/pages/QuizPage.tsx` | MODIFY | Pass `wrongAnswers` to `QuizResults`; update stale button styles |
| `frontend/src/pages/DiscoverPage.tsx` | CREATE | "Hit Me" page with embedded player |
| `frontend/src/App.tsx` | MODIFY | Add `/discover` route |
| `frontend/src/components/Layout.tsx` | MODIFY | Add Discover nav item |

---

### Task 1: `youtube_query.py` — LLM Query Generator + Random Song

**Files:**
- Create: `backend/services/youtube_query.py`
- Create: `tests/test_youtube_query.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_youtube_query.py`:

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_generate_tutorial_query_returns_llm_string():
    from backend.services.youtube_query import generate_tutorial_query

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value="JLPT N4 te-form permission grammar")

    result = await generate_tutorial_query(
        question="彼女は毎日日本語を＿＿＿います。",
        correct_answer="A",
        user_answer="B",
        concepts=["て-form", "継続"],
        provider=mock_provider,
    )
    assert result == "JLPT N4 te-form permission grammar"
    mock_provider.complete.assert_called_once()


@pytest.mark.asyncio
async def test_generate_tutorial_query_falls_back_on_garbage_response():
    from backend.services.youtube_query import generate_tutorial_query

    mock_provider = MagicMock()
    # LLM returns a multi-line garbage response
    mock_provider.complete = AsyncMock(return_value="Here is a query:\nJLPT grammar\nFor your reference")

    result = await generate_tutorial_query(
        question="question text",
        correct_answer="A",
        user_answer="C",
        concepts=["は vs が"],
        provider=mock_provider,
    )
    # Fallback uses first concept
    assert "は vs が" in result


@pytest.mark.asyncio
async def test_get_random_japanese_song_returns_video():
    from backend.services.youtube_query import get_random_japanese_song
    from backend.services.recommendations import Video

    fake_video = Video(
        video_id="xyz789",
        title="夜に駆ける - YOASOBI",
        thumbnail_url="https://img.youtube.com/vi/xyz789/mqdefault.jpg",
        channel_title="YOASOBI Official",
    )

    with patch("backend.services.youtube_query.search_youtube", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [fake_video]
        result = await get_random_japanese_song(api_key="fake-key")

    assert result is not None
    assert result.video_id == "xyz789"


@pytest.mark.asyncio
async def test_get_random_japanese_song_returns_none_when_no_results():
    from backend.services.youtube_query import get_random_japanese_song

    with patch("backend.services.youtube_query.search_youtube", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        result = await get_random_japanese_song(api_key="fake-key")

    assert result is None


@pytest.mark.asyncio
async def test_search_tutorials_deduplicates_by_concept():
    from backend.services.youtube_query import search_tutorials_for_wrong_answers
    from backend.services.recommendations import Video

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value="JLPT N4 grammar")

    fake_video = Video(
        video_id="v1", title="Tutorial", thumbnail_url="https://img.youtube.com/vi/v1/mqdefault.jpg",
        channel_title="Channel",
    )

    wrong_items = [
        {"question": "q1", "correct_answer": "A", "user_answer": "B", "concepts": ["て-form"]},
        {"question": "q2", "correct_answer": "C", "user_answer": "A", "concepts": ["て-form"]},  # duplicate concept
        {"question": "q3", "correct_answer": "B", "user_answer": "D", "concepts": ["は vs が"]},
    ]

    with patch("backend.services.youtube_query.search_youtube", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [fake_video]
        results = await search_tutorials_for_wrong_answers(
            wrong_items=wrong_items, api_key="fake-key", provider=mock_provider
        )

    # Only 2 unique concepts, not 3 items
    assert len(results) == 2
    assert results[0]["concepts"] == ["て-form"]
    assert results[1]["concepts"] == ["は vs が"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024
python -m pytest tests/test_youtube_query.py -v
```
Expected: `ModuleNotFoundError: No module named 'backend.services.youtube_query'`

- [ ] **Step 3: Create `backend/services/youtube_query.py`**

```python
import random

from backend.services.llm.base import LLMProvider
from backend.services.recommendations import search_youtube, Video

TUTORIAL_PROMPT = """You are a JLPT grammar expert.
A student answered a Japanese grammar question incorrectly.
Generate a focused YouTube search query (in English) to find a tutorial explaining the specific grammar rule they missed.

Question: {question}
Correct answer: {correct_answer}
Student's answer: {user_answer}
Grammar concepts: {concepts}

Return ONLY the search query (3-8 words). Example: "JLPT N4 te-form permission grammar"
No explanation. No quotes. Just the query."""

# Curated grammar-themed song queries for random discovery
SONG_QUERIES = [
    "日本語 JLPT Japanese learning song grammar",
    "Japanese pop song everyday conversation 日常",
    "anime opening Japanese language N4 N5",
    "J-pop Japanese song 2024 popular",
    "YOASOBI Official Music Video Japanese",
    "Kenshi Yonezu Japanese music popular",
    "Japanese city pop classics 70s 80s",
    "anime ending theme Japanese language",
    "Japanese traditional folk song grammar story",
    "J-rock popular Japanese band song",
]


async def generate_tutorial_query(
    question: str,
    correct_answer: str,
    user_answer: str,
    concepts: list[str],
    provider: LLMProvider,
) -> str:
    """Ask the LLM to produce a focused YouTube search query for this wrong answer."""
    prompt = TUTORIAL_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        user_answer=user_answer,
        concepts=", ".join(concepts) if concepts else "unknown",
    )
    raw = await provider.complete(prompt)
    query = raw.strip().strip('"').strip("'").splitlines()[0].strip()
    # Fallback: use first concept if LLM returns something suspicious
    if len(query) > 120 or not query:
        fallback = concepts[0] if concepts else "JLPT grammar Japanese"
        query = f"JLPT {fallback} grammar explanation"
    return query


async def search_tutorials_for_wrong_answers(
    wrong_items: list[dict],
    api_key: str,
    provider: LLMProvider,
    max_per_item: int = 2,
    max_items: int = 3,
) -> list[dict]:
    """
    For up to `max_items` unique concepts in `wrong_items`:
      1. Generate a targeted YouTube search query via LLM.
      2. Call YouTube API.
      3. Return structured results.

    Deduplicates by first concept to conserve YouTube API quota.
    """
    seen_concepts: set[str] = set()
    results: list[dict] = []

    for item in wrong_items:
        if len(results) >= max_items:
            break
        concept_key = item["concepts"][0] if item.get("concepts") else item["correct_answer"]
        if concept_key in seen_concepts:
            continue
        seen_concepts.add(concept_key)

        query = await generate_tutorial_query(
            question=item["question"],
            correct_answer=item["correct_answer"],
            user_answer=item["user_answer"],
            concepts=item.get("concepts", []),
            provider=provider,
        )
        videos = await search_youtube(query, api_key, max_results=max_per_item)
        results.append({
            "question_snippet": item["question"][:80],
            "concepts": item.get("concepts", []),
            "search_query": query,
            "videos": videos,
        })

    return results


async def get_random_japanese_song(
    api_key: str,
    concept: str | None = None,
) -> Video | None:
    """
    Return a random Japanese song video from YouTube.
    If `concept` is given, bias toward grammar-relevant music.
    """
    if concept:
        query = f"Japanese song {concept} 日本語 lyrics"
    else:
        query = random.choice(SONG_QUERIES)

    videos = await search_youtube(query, api_key, max_results=10)
    if not videos:
        return None
    return random.choice(videos)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024
python -m pytest tests/test_youtube_query.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py
```
Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add backend/services/youtube_query.py tests/test_youtube_query.py
git commit -m "feat: LLM-powered YouTube query generator and random Japanese song service"
```

---

### Task 2: Backend Schemas + Two New Endpoints

**Files:**
- Modify: `backend/schemas/api.py`
- Modify: `backend/routers/advanced.py`

- [ ] **Step 1: Add three new schemas to `backend/schemas/api.py`**

Append to the bottom of `backend/schemas/api.py`:

```python
class WrongItemRec(BaseModel):
    question: str
    correct_answer: str
    user_answer: str
    concepts: list[str] = []


class WrongAnswerRecsRequest(BaseModel):
    wrong_items: list[WrongItemRec]


class VideoTutorialSet(BaseModel):
    question_snippet: str
    concepts: list[str]
    search_query: str
    videos: list[VideoRecommendation]


class WrongAnswerRecsResponse(BaseModel):
    recommendations: list[VideoTutorialSet]
```

- [ ] **Step 2: Add two new routes to `backend/routers/advanced.py`**

Add these imports at the top of `backend/routers/advanced.py` (after existing imports):

```python
from fastapi import HTTPException
from backend.schemas.api import (
    WrongAnswerRecsRequest, WrongAnswerRecsResponse, VideoTutorialSet,
)
from backend.services.youtube_query import (
    search_tutorials_for_wrong_answers,
    get_random_japanese_song,
)
```

Append these two routes at the end of `backend/routers/advanced.py`:

```python
@router.post("/recommendations/for-wrong-answers", response_model=WrongAnswerRecsResponse)
async def recs_for_wrong_answers(
    body: WrongAnswerRecsRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Given a list of wrong quiz answers, return YouTube tutorial videos
    for up to 3 unique grammar concepts found in those answers.
    Returns an empty list when YOUTUBE_API_KEY is not configured.
    """
    if not settings.youtube_api_key:
        return WrongAnswerRecsResponse(recommendations=[])

    provider = get_llm_provider()
    raw_items = [
        {
            "question": item.question,
            "correct_answer": item.correct_answer,
            "user_answer": item.user_answer,
            "concepts": item.concepts,
        }
        for item in body.wrong_items
    ]
    results = await search_tutorials_for_wrong_answers(
        wrong_items=raw_items,
        api_key=settings.youtube_api_key,
        provider=provider,
    )
    recommendations = [
        VideoTutorialSet(
            question_snippet=r["question_snippet"],
            concepts=r["concepts"],
            search_query=r["search_query"],
            videos=[
                VideoRecommendation(
                    title=v.title,
                    video_id=v.video_id,
                    thumbnail_url=v.thumbnail_url,
                    channel_title=v.channel_title,
                )
                for v in r["videos"]
            ],
        )
        for r in results
    ]
    return WrongAnswerRecsResponse(recommendations=recommendations)


@router.get("/youtube/random-song", response_model=VideoRecommendation)
async def random_japanese_song(
    concept: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """
    Return a random Japanese song video from YouTube.
    Optional `concept` query param biases toward grammar-relevant music.
    Returns 404 when YouTube API key is not configured or no results found.
    """
    if not settings.youtube_api_key:
        raise HTTPException(status_code=404, detail="YouTube API not configured")

    video = await get_random_japanese_song(
        api_key=settings.youtube_api_key,
        concept=concept,
    )
    if video is None:
        raise HTTPException(status_code=404, detail="No video found")

    return VideoRecommendation(
        title=video.title,
        video_id=video.video_id,
        thumbnail_url=video.thumbnail_url,
        channel_title=video.channel_title,
    )
```

- [ ] **Step 3: Run the full backend test suite**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py
```
Expected: all tests pass (the new endpoints have no unit tests beyond Task 1; integration is covered by smoke tests).

- [ ] **Step 4: Start the dev server and verify the new routes appear in OpenAPI**

```bash
uvicorn backend.main:app --reload --app-dir .
```
Then open `http://127.0.0.1:8000/docs` and confirm:
- `POST /api/v1/recommendations/for-wrong-answers` is listed
- `GET /api/v1/youtube/random-song` is listed

Stop the server (Ctrl+C) after verifying.

- [ ] **Step 5: Commit**

```bash
git add backend/schemas/api.py backend/routers/advanced.py
git commit -m "feat: add for-wrong-answers and random-song YouTube endpoints"
```

---

### Task 3: Frontend Types + API Client Methods

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Add new types to `frontend/src/types/index.ts`**

Append to the bottom of `frontend/src/types/index.ts`:

```typescript
export interface VideoTutorialSet {
  question_snippet: string;
  concepts: string[];
  search_query: string;
  videos: VideoRecommendation[];
}

export interface WrongAnswerRecsResponse {
  recommendations: VideoTutorialSet[];
}
```

- [ ] **Step 2: Add two new methods to `frontend/src/api/client.ts`**

Append inside the `api` object in `frontend/src/api/client.ts`, before the closing `};`:

```typescript
  getRecsForWrongAnswers: (wrongItems: {
    question: string;
    correct_answer: string;
    user_answer: string;
    concepts: string[];
  }[]) =>
    post<import("../types").WrongAnswerRecsResponse>(
      "/api/v1/recommendations/for-wrong-answers",
      { wrong_items: wrongItems },
      true
    ),

  getRandomSong: (concept?: string) =>
    get<import("../types").VideoRecommendation>(
      `/api/v1/youtube/random-song${concept ? `?concept=${encodeURIComponent(concept)}` : ""}`
    ),
```

- [ ] **Step 3: Run type-check**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024\frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/types/index.ts frontend/src/api/client.ts
git commit -m "feat: add YouTube recommendation types and API client methods"
```

---

### Task 4: `YouTubeCard` Component

**Files:**
- Create: `frontend/src/components/YouTubeCard.tsx`

- [ ] **Step 1: Create `frontend/src/components/YouTubeCard.tsx`**

```typescript
import { useState } from "react";
import type { VideoRecommendation } from "../types";

interface Props {
  video: VideoRecommendation;
  autoPlay?: boolean;
  delay?: number;
}

export default function YouTubeCard({ video, autoPlay = false, delay = 0 }: Props) {
  const [playing, setPlaying] = useState(autoPlay);

  return (
    <div
      className="rounded-2xl overflow-hidden border border-cream bg-ivory animate-pop-in"
      style={{ animationDelay: `${delay}ms` }}
    >
      {playing ? (
        <iframe
          src={`https://www.youtube.com/embed/${video.video_id}?autoplay=1&rel=0`}
          className="w-full aspect-video"
          allow="autoplay; encrypted-media; picture-in-picture"
          allowFullScreen
          title={video.title}
        />
      ) : (
        <button
          onClick={() => setPlaying(true)}
          className="w-full relative group focus:outline-none"
          aria-label={`Play ${video.title}`}
        >
          {video.thumbnail_url ? (
            <img
              src={video.thumbnail_url}
              alt={video.title}
              className="w-full aspect-video object-cover"
            />
          ) : (
            <div className="w-full aspect-video bg-sand flex items-center justify-center">
              <span className="text-ash text-sm">No thumbnail</span>
            </div>
          )}
          {/* Play overlay */}
          <div className="absolute inset-0 flex items-center justify-center
            bg-ink/10 group-hover:bg-ink/20 transition-colors">
            <span className="w-12 h-12 rounded-full bg-red-600 flex items-center justify-center
              shadow-lg group-hover:scale-110 transition-transform">
              <svg viewBox="0 0 24 24" fill="white" className="w-5 h-5 ml-0.5">
                <path d="M8 5v14l11-7z" />
              </svg>
            </span>
          </div>
        </button>
      )}

      <div className="px-3 py-2.5 space-y-0.5">
        <p className="text-xs font-medium text-ink leading-snug line-clamp-2">{video.title}</p>
        <p className="text-xs text-ash">{video.channel_title}</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Run type-check**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024\frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/YouTubeCard.tsx
git commit -m "feat: YouTubeCard component with click-to-embed and autoplay support"
```

---

### Task 5: YouTube Tutorials in QuizResults

**Files:**
- Modify: `frontend/src/hooks/useQuizMode.ts`
- Modify: `frontend/src/pages/QuizPage.tsx`
- Modify: `frontend/src/components/QuizResults.tsx`

The goal: after quiz results, show a "Find Tutorial Videos" button. On click, call `getRecsForWrongAnswers` and render `YouTubeCard` grids per concept.

- [ ] **Step 1: Expose `wrongAnswers` from `useQuizMode.ts`**

In `frontend/src/hooks/useQuizMode.ts`, add the following derived value in the return object (at the bottom, just before the closing `}`):

Current return block ends with:
```typescript
  return {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    wrongAttemptIds,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset, restoreResults,
  };
```

Replace it with:
```typescript
  const wrongAnswers = answers
    .filter(a => !a.isCorrect)
    .map(a => ({
      question: a.question,
      correct_answer: a.correctAnswer,
      user_answer: a.userAnswer,
      concepts: [] as string[],
    }));

  return {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    wrongAttemptIds, wrongAnswers,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset, restoreResults,
  };
```

- [ ] **Step 2: Pass `wrongAnswers` to `QuizResults` from `QuizPage.tsx`**

Read `frontend/src/pages/QuizPage.tsx`. The destructuring at the top currently reads:
```typescript
  const {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    wrongAttemptIds,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset, restoreResults,
  } = useQuizMode();
```

Replace with:
```typescript
  const {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    wrongAttemptIds, wrongAnswers,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset, restoreResults,
  } = useQuizMode();
```

Then in the JSX, update the `<QuizResults>` call from:
```typescript
      {screen === "results" && (
        <QuizResults
          score={score}
          total={answers.length}
          analyses={analyses}
          analyzing={analyzing}
          wrongAttemptIds={wrongAttemptIds}
          onRetry={reset}
        />
      )}
```
to:
```typescript
      {screen === "results" && (
        <QuizResults
          score={score}
          total={answers.length}
          analyses={analyses}
          analyzing={analyzing}
          wrongAttemptIds={wrongAttemptIds}
          wrongAnswers={wrongAnswers}
          onRetry={reset}
        />
      )}
```

Also update the stale button styles (currently uses old `slate-*` classes):
```typescript
          {savedResults && (
            <button
              onClick={() => restoreResults(savedResults)}
              className="w-full border border-cream bg-sand hover:bg-sand/80 active:scale-[0.98]
                text-charcoal py-2 rounded-xl text-xs font-medium transition-all"
            >
              View Last Quiz Results
            </button>
          )}
```

- [ ] **Step 3: Update `QuizResults.tsx` — add `wrongAnswers` prop + tutorial section**

The full updated `frontend/src/components/QuizResults.tsx`:

```typescript
import { useState, useCallback } from "react";
import { api } from "../api/client";
import type { AnalysisItem, VideoTutorialSet } from "../types";
import YouTubeCard from "./YouTubeCard";

interface WrongAnswer {
  question: string;
  correct_answer: string;
  user_answer: string;
  concepts: string[];
}

interface Props {
  score: number;
  total: number;
  analyses: AnalysisItem[];
  analyzing: boolean;
  wrongAttemptIds: number[];
  wrongAnswers?: WrongAnswer[];
  onRetry: () => void;
}

export default function QuizResults({
  score, total, analyses, analyzing, wrongAttemptIds, wrongAnswers = [], onRetry,
}: Props) {
  const pct = total > 0 ? Math.round((score / total) * 100) : 0;
  const wrong = total - score;
  const [explanations, setExplanations] = useState<Record<number, string>>({});
  const [generating, setGenerating] = useState<Record<number, boolean>>({});
  const [ytRecs, setYtRecs] = useState<VideoTutorialSet[] | null>(null);
  const [ytLoading, setYtLoading] = useState(false);

  const handleGenerate = async (index: number) => {
    const attemptId = wrongAttemptIds[index];
    if (attemptId == null) return;
    setGenerating(g => ({ ...g, [index]: true }));
    try {
      const res = await api.explainAttempt(attemptId);
      setExplanations(e => ({ ...e, [index]: res.explanation }));
    } catch {/* ignore */} finally {
      setGenerating(g => ({ ...g, [index]: false }));
    }
  };

  const loadTutorials = useCallback(async () => {
    if (wrongAnswers.length === 0) return;
    setYtLoading(true);
    try {
      const res = await api.getRecsForWrongAnswers(wrongAnswers);
      setYtRecs(res.recommendations);
    } catch {
      setYtRecs([]);
    } finally {
      setYtLoading(false);
    }
  }, [wrongAnswers]);

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Score card */}
      <div className="bg-ivory border border-cream rounded-2xl p-6 text-center animate-pop-in shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
        <p className="text-5xl font-black text-brand-500">{pct}%</p>
        <p className="text-sm text-ash mt-1">
          {score} correct · {wrong} wrong · {total} total
        </p>
        <div className="h-2 bg-sand rounded-full overflow-hidden mt-4">
          <div
            className="h-full bg-brand-500 rounded-full transition-all duration-700"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      {/* Wrong answers with LLM analysis */}
      {wrong > 0 && (
        <div className="space-y-3">
          <p className="text-xs font-medium text-ash uppercase tracking-wide">
            {analyzing ? "Analyzing wrong answers…" : `${wrong} wrong answer${wrong > 1 ? "s" : ""}`}
          </p>

          {analyzing ? (
            <div className="space-y-2">
              {Array.from({ length: wrong }).map((_, i) => (
                <div key={i} className="h-16 bg-sand rounded-2xl animate-timer-pulse" />
              ))}
            </div>
          ) : analyses.length === 0 ? (
            wrongAttemptIds.length > 0 ? (
              <div className="space-y-2">
                {wrongAttemptIds.map((_id, i) => (
                  <div key={i} className="bg-ivory border border-cream rounded-2xl p-3.5 animate-slide-in"
                    style={{ animationDelay: `${i * 60}ms` }}>
                    <button
                      onClick={() => handleGenerate(i)}
                      disabled={generating[i]}
                      className="text-xs text-brand-500 font-semibold hover:underline disabled:opacity-50"
                    >
                      {generating[i] ? "Generating…" : "Generate explanation"}
                    </button>
                    {explanations[i] && (
                      <p className="text-xs text-bark leading-relaxed mt-2">{explanations[i]}</p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-xs text-ash italic">Analysis unavailable</p>
            )
          ) : (
            analyses.map((item, i) => {
              const explanation = explanations[i] || item.explanation;
              return (
                <div
                  key={i}
                  className="bg-ivory border border-cream rounded-2xl p-4 space-y-2 animate-slide-in"
                  style={{ animationDelay: `${i * 60}ms` }}
                >
                  <p className="text-sm text-ink leading-snug">{item.question}</p>
                  <div className="flex gap-3 text-xs">
                    <span className="text-red-600 font-semibold">You: {item.user_answer}</span>
                    <span className="text-green-700 font-semibold">Correct: {item.correct_answer}</span>
                  </div>
                  {explanation ? (
                    <p className="text-xs text-bark leading-relaxed border-t border-cream pt-2">
                      {explanation}
                    </p>
                  ) : wrongAttemptIds[i] ? (
                    <div className="border-t border-cream pt-2">
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

      {/* YouTube Tutorial Recommendations */}
      {wrong > 0 && wrongAnswers.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium text-ash uppercase tracking-wide">Tutorial Videos</p>
            {ytRecs === null && !ytLoading && (
              <button
                onClick={loadTutorials}
                className="text-xs text-brand-500 font-semibold hover:underline"
              >
                Find tutorials →
              </button>
            )}
          </div>

          {ytLoading && (
            <div className="h-32 bg-sand rounded-2xl animate-timer-pulse" />
          )}

          {ytRecs !== null && ytRecs.length === 0 && (
            <p className="text-xs text-ash italic">No tutorials found. Try again later.</p>
          )}

          {ytRecs && ytRecs.map((rec, i) => (
            <div key={i} className="space-y-2 animate-fade-in" style={{ animationDelay: `${i * 80}ms` }}>
              {rec.concepts.length > 0 && (
                <p className="text-xs font-medium text-bark">
                  {rec.concepts.join(" · ")}
                </p>
              )}
              <div className="grid grid-cols-2 gap-2">
                {rec.videos.map((v, j) => (
                  <YouTubeCard key={j} video={v} delay={j * 100} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <button
        onClick={onRetry}
        className="w-full border border-cream bg-sand hover:bg-sand/80 active:scale-[0.98]
          text-charcoal py-2.5 rounded-xl text-sm font-medium transition-all"
      >
        ← Back to Setup
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Run type-check**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024\frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 5: Run build**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024\frontend && npm run build 2>&1 | tail -5
```
Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/hooks/useQuizMode.ts \
        frontend/src/pages/QuizPage.tsx \
        frontend/src/components/QuizResults.tsx
git commit -m "feat: YouTube tutorial recommendations in quiz results"
```

---

### Task 6: `DiscoverPage` — "Hit Me" Japanese Song

**Files:**
- Create: `frontend/src/pages/DiscoverPage.tsx`

- [ ] **Step 1: Create `frontend/src/pages/DiscoverPage.tsx`**

```typescript
import { useState } from "react";
import { api } from "../api/client";
import type { VideoRecommendation } from "../types";
import YouTubeCard from "../components/YouTubeCard";

export default function DiscoverPage() {
  const [video, setVideo] = useState<VideoRecommendation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hitCount, setHitCount] = useState(0);

  const hitMe = async () => {
    setLoading(true);
    setVideo(null);
    setError("");
    try {
      const v = await api.getRandomSong();
      setVideo(v);
      setHitCount(n => n + 1);
    } catch {
      setError("Couldn't find a video right now. Try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-serif text-2xl font-medium text-ink">Discover</h1>
        <p className="text-sm text-bark mt-1 leading-relaxed">
          Warm up with a random Japanese song before you study.
        </p>
      </div>

      <div className="flex flex-col items-center gap-6">
        <button
          onClick={hitMe}
          disabled={loading}
          className="bg-brand-500 hover:bg-brand-600 active:scale-95 text-ivory
            text-base font-semibold px-10 py-4 rounded-2xl transition-all
            disabled:opacity-50 shadow-[rgba(0,0,0,0.08)_0px_8px_32px]"
        >
          {loading ? "Finding…" : hitCount === 0 ? "🎵 Hit Me" : "🎵 Another One"}
        </button>

        {error && (
          <p className="text-sm text-red-700 text-center">{error}</p>
        )}

        {video && !loading && (
          <div className="w-full max-w-lg animate-fade-in">
            <YouTubeCard video={video} autoPlay />
          </div>
        )}

        {!video && !loading && hitCount === 0 && (
          <div className="text-center space-y-2 py-8">
            <p className="text-3xl">🎌</p>
            <p className="text-bark text-sm">Hit the button to discover a Japanese song.</p>
            <p className="text-ash text-xs max-w-xs">
              Songs are picked from J-pop, anime, city pop, and traditional music — sometimes with grammar relevance.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Run type-check**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024\frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/DiscoverPage.tsx
git commit -m "feat: Discover page with Hit Me random Japanese song player"
```

---

### Task 7: Routing + Navigation

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/Layout.tsx`

- [ ] **Step 1: Add `/discover` route to `frontend/src/App.tsx`**

Current `App.tsx`:
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

Replace with:
```typescript
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import LoginPage from "./pages/LoginPage";
import Layout from "./components/Layout";
import AskPage from "./pages/AskPage";
import HistoryPage from "./pages/HistoryPage";
import StatsPage from "./pages/StatsPage";
import QuizPage from "./pages/QuizPage";
import DiscoverPage from "./pages/DiscoverPage";

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
          <Route path="/discover" element={<Layout><DiscoverPage /></Layout>} />
          <Route path="*" element={<Navigate to="/ask" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
```

- [ ] **Step 2: Add Discover nav item to `frontend/src/components/Layout.tsx`**

In `Layout.tsx`, update `NAV_ITEMS` from:
```typescript
const NAV_ITEMS = [
  { to: "/ask",      label: "Ask" },
  { to: "/quiz",     label: "Quiz" },
  { to: "/history",  label: "History" },
  { to: "/stats",    label: "Stats" },
];
```
to:
```typescript
const NAV_ITEMS = [
  { to: "/ask",      label: "Ask" },
  { to: "/quiz",     label: "Quiz" },
  { to: "/history",  label: "History" },
  { to: "/stats",    label: "Stats" },
  { to: "/discover", label: "Discover" },
];
```

- [ ] **Step 3: Run type-check**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024\frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 4: Run final build**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024\frontend && npm run build 2>&1 | tail -6
```
Expected: build succeeds.

- [ ] **Step 5: Run full backend test suite**

```bash
cd C:\Users\Andrew\Desktop\Projects\NASC_2024
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py
```
Expected: all tests pass.

- [ ] **Step 6: Final commit**

```bash
git add frontend/src/App.tsx frontend/src/components/Layout.tsx
git commit -m "feat: add Discover route and nav item for Hit Me page"
```

---

## Self-Review

### 1. Spec Coverage

| Requirement | Task |
|---|---|
| For wrong quiz answers, search for grammar tutorial videos | Task 1 (service), Task 2 (endpoint), Task 5 (UI) |
| LLM generates search keywords | Task 1 (`generate_tutorial_query`) |
| YouTube API v3 integration | Task 1+2 (reuses existing `search_youtube`) |
| Tutorial videos shown in quiz results | Task 5 (`QuizResults` + `YouTubeCard`) |
| "Hit Me" starter tab | Task 6 (`DiscoverPage`) |
| Plays a random Japanese song | Task 1 (`get_random_japanese_song`), Task 2 (`/random-song`), Task 6 |
| Grammar relevance for songs (if possible) | Task 1 (grammar-themed `SONG_QUERIES`; optional `concept` param) |
| UIUX: matches warm parchment design system | Task 4–6 (ivory/cream/brand-500 throughout) |
| Backend: YouTube API v3 calling flow | Task 1 (service), Task 2 (router) |
| YouTube API key in env | Already in `backend/config.py` as `youtube_api_key` |
| Graceful degradation without API key | Task 2 (returns empty/404 when key absent) |

### 2. Placeholder Scan

None found. All steps contain complete code.

### 3. Type Consistency

- `VideoRecommendation` used in `VideoTutorialSet.videos` (Task 2) matches the existing schema definition ✓
- `WrongAnswerRecsResponse.recommendations: list[VideoTutorialSet]` matches `VideoTutorialSet[]` in frontend ✓
- `wrongAnswers` derived in `useQuizMode` matches `WrongAnswer` interface in `QuizResults` ✓
- `YouTubeCard` receives `VideoRecommendation` — same type used in `DiscoverPage` ✓
- `api.getRandomSong()` returns `VideoRecommendation` in both frontend type and backend schema ✓

---

## Prerequisites

Before starting:
1. Confirm `YOUTUBE_API_KEY=your-key-here` is set in `.env` (already wired in `backend/config.py`)
2. Confirm the YouTube Data API v3 is enabled in your Google Cloud project at `console.cloud.google.com`
3. Verify quota: each `search.list` call costs 100 units; free tier = 10,000 units/day

The app functions without a YouTube key — tutorial and song sections show empty states.
