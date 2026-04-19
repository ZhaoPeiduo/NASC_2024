# Discover Revamp, TikTok Integration, Quiz Generation & Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Discover the landing page, add YouTube/TikTok source toggle with self-hosted video playback, add three-mode QuizSetup (CSV / Screenshots / AI Generate), and remove all legacy competition files + rewrite the README.

**Architecture:** TikTok videos are fetched via the unofficial `TikTokApi` (Playwright-based), downloaded as bytes by the backend, and served as a `StreamingResponse`; the frontend fetches with auth headers, creates a blob URL, and plays in an HTML5 `<video>` tag. QuizSetup grows a mode toggle that reuses the existing `POST /api/v1/generate` endpoint (AI mode) and a new `POST /api/v1/quiz/ocr/batch` endpoint (Screenshots mode). All other plumbing (routing, nav, README) is mechanical.

**Tech Stack:** FastAPI async, TikTokApi + Playwright, EasyOCR (existing), React 18, TypeScript, Tailwind CSS v4

---

## Stakeholder Notes

**Backend Engineer:** Two new service files: `tiktok_service.py` (TikTokApi wrapper) and a new OCR batch helper in `quiz.py`. `tiktok_service.py` opens a fresh `TikTokApi` async context per request (correct but expensive; acceptable for a dev/demo app). Add `tiktok_ms_token` to `Settings`. Add `TikTokApi` to `base.txt`. `StreamingResponse` streams bytes directly.

**Frontend Engineer:** `DiscoverPage` gets a source toggle (YouTube | TikTok) and per-source topic chips. `TikTokCard` fetches the stream endpoint with `fetch` + `Authorization` header, converts to blob URL, assigns to a `<video>` ref. `QuizSetup` gets a 3-way mode toggle; Screenshots and AI Generate modes both produce a preview list of `QuizQuestion[]` before handing off to the existing `onStart`.

---

## File Map

| File | Action |
|------|--------|
| `backend/config.py` | MODIFY — add `tiktok_ms_token: str = ""` |
| `backend/requirements/base.txt` | MODIFY — add `TikTokApi` |
| `backend/services/tiktok_service.py` | CREATE |
| `backend/schemas/api.py` | MODIFY — add `TikTokVideoMeta`, `OcrBatchResponse` |
| `backend/routers/advanced.py` | MODIFY — add 2 TikTok endpoints |
| `backend/routers/quiz.py` | MODIFY — add OCR batch endpoint |
| `tests/test_tiktok_service.py` | CREATE |
| `tests/test_ocr_batch.py` | CREATE |
| `frontend/src/types/index.ts` | MODIFY — add `TikTokVideoMeta` |
| `frontend/src/api/client.ts` | MODIFY — add `getRandomTikTok`, `uploadOcrImages` |
| `frontend/src/components/TikTokCard.tsx` | CREATE |
| `frontend/src/pages/DiscoverPage.tsx` | MODIFY — source toggle + topic chips |
| `frontend/src/components/QuizSetup.tsx` | MODIFY — 3-mode toggle |
| `frontend/src/App.tsx` | MODIFY — `*` → `/discover` |
| `frontend/src/components/Layout.tsx` | MODIFY — nav order |
| `readme.md` | REWRITE |
| Legacy files/dirs | DELETE |

---

### Task 1: Routing + Nav Reorder

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/components/Layout.tsx`

- [ ] **Step 1: Update catch-all route in `frontend/src/App.tsx`**

Change the last `<Route>` from:
```typescript
<Route path="*" element={<Navigate to="/ask" replace />} />
```
to:
```typescript
<Route path="*" element={<Navigate to="/discover" replace />} />
```

- [ ] **Step 2: Reorder nav items in `frontend/src/components/Layout.tsx`**

Replace:
```typescript
const NAV_ITEMS = [
  { to: "/ask",      label: "Ask" },
  { to: "/quiz",     label: "Quiz" },
  { to: "/history",  label: "History" },
  { to: "/stats",    label: "Stats" },
  { to: "/discover", label: "Discover" },
];
```
with:
```typescript
const NAV_ITEMS = [
  { to: "/discover", label: "Discover" },
  { to: "/ask",      label: "Ask" },
  { to: "/quiz",     label: "Quiz" },
  { to: "/history",  label: "History" },
  { to: "/stats",    label: "Stats" },
];
```

- [ ] **Step 3: Run type-check**
```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 4: Commit**
```bash
git add frontend/src/App.tsx frontend/src/components/Layout.tsx
git commit -m "feat: make Discover the default landing page and move it first in nav"
```

---

### Task 2: TikTok Config + Requirements

**Files:**
- Modify: `backend/config.py`
- Modify: `backend/requirements/base.txt`

- [ ] **Step 1: Add `tiktok_ms_token` to `backend/config.py`**

Add the following field after the `youtube_api_key` line:
```python
    tiktok_ms_token: str = ""
```

Full updated `Settings` class (replace the whole class):
```python
class Settings(BaseSettings):
    llm_provider: Literal["openrouter", "local"] = "openrouter"

    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"

    local_model_name: str = "stabilityai/japanese-stablelm-instruct-gamma-7b"

    secret_key: SecretStr = SecretStr("dev-secret-change-in-production-!!!")
    access_token_expire_minutes: int = 60 * 24 * 7

    database_url: str = "sqlite+aiosqlite:///./jlpt_sensei.db"

    youtube_api_key: str = ""
    tiktok_ms_token: str = ""

    allowed_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env")
```

- [ ] **Step 2: Add TikTokApi to `backend/requirements/base.txt`**

Append to the bottom of the file:
```
TikTokApi
```

- [ ] **Step 3: Install TikTokApi and Playwright**
```bash
pip install TikTokApi
python -m playwright install chromium
```
Expected: no errors.

- [ ] **Step 4: Commit**
```bash
git add backend/config.py backend/requirements/base.txt
git commit -m "feat: add tiktok_ms_token config and TikTokApi dependency"
```

---

### Task 3: TikTok Backend Service

**Files:**
- Create: `backend/services/tiktok_service.py`
- Create: `tests/test_tiktok_service.py`

- [ ] **Step 1: Write the failing tests — create `tests/test_tiktok_service.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, AsyncContextManager
from types import SimpleNamespace


def _make_fake_video(video_id="abc123", desc="Test video", nickname="TestUser"):
    v = MagicMock()
    v.as_dict = {
        "id": video_id,
        "desc": desc,
        "author": {"nickname": nickname},
    }
    v.bytes = AsyncMock(return_value=b"fakevideobytes")
    return v


@pytest.mark.asyncio
async def test_get_random_tiktok_video_trending():
    from backend.services.tiktok_service import get_random_tiktok_video

    fake_video = _make_fake_video("v1", "Japanese lesson", "JapaneseTeacher")

    mock_api = MagicMock()
    mock_api.__aenter__ = AsyncMock(return_value=mock_api)
    mock_api.__aexit__ = AsyncMock(return_value=False)
    mock_api.create_sessions = AsyncMock()

    async def fake_trending_videos(count=30):
        yield fake_video

    mock_api.trending = MagicMock()
    mock_api.trending.videos = fake_trending_videos

    with patch("backend.services.tiktok_service.TikTokApi", return_value=mock_api):
        result = await get_random_tiktok_video(topic="trending", ms_token="tok")

    assert result is not None
    assert result.video_id == "v1"
    assert result.title == "Japanese lesson"
    assert result.author == "JapaneseTeacher"


@pytest.mark.asyncio
async def test_get_random_tiktok_video_returns_none_on_empty():
    from backend.services.tiktok_service import get_random_tiktok_video

    mock_api = MagicMock()
    mock_api.__aenter__ = AsyncMock(return_value=mock_api)
    mock_api.__aexit__ = AsyncMock(return_value=False)
    mock_api.create_sessions = AsyncMock()

    async def fake_trending_videos(count=30):
        return
        yield  # make it an async generator

    mock_api.trending = MagicMock()
    mock_api.trending.videos = fake_trending_videos

    with patch("backend.services.tiktok_service.TikTokApi", return_value=mock_api):
        result = await get_random_tiktok_video(topic="trending", ms_token="tok")

    assert result is None


@pytest.mark.asyncio
async def test_fetch_video_bytes():
    from backend.services.tiktok_service import fetch_video_bytes

    fake_video = _make_fake_video("v2")

    mock_api = MagicMock()
    mock_api.__aenter__ = AsyncMock(return_value=mock_api)
    mock_api.__aexit__ = AsyncMock(return_value=False)
    mock_api.create_sessions = AsyncMock()
    mock_api.video = MagicMock(return_value=fake_video)

    with patch("backend.services.tiktok_service.TikTokApi", return_value=mock_api):
        data = await fetch_video_bytes(video_id="v2", ms_token="tok")

    assert data == b"fakevideobytes"


@pytest.mark.asyncio
async def test_get_random_tiktok_video_hashtag():
    from backend.services.tiktok_service import get_random_tiktok_video

    fake_video = _make_fake_video("v3", "Anime scene", "AnimeChannel")

    mock_api = MagicMock()
    mock_api.__aenter__ = AsyncMock(return_value=mock_api)
    mock_api.__aexit__ = AsyncMock(return_value=False)
    mock_api.create_sessions = AsyncMock()

    mock_hashtag = MagicMock()

    async def fake_hashtag_videos(count=30):
        yield fake_video

    mock_hashtag.videos = fake_hashtag_videos
    mock_api.hashtag = MagicMock(return_value=mock_hashtag)

    with patch("backend.services.tiktok_service.TikTokApi", return_value=mock_api):
        result = await get_random_tiktok_video(topic="anime", ms_token="tok")

    assert result is not None
    assert result.video_id == "v3"
    mock_api.hashtag.assert_called_once_with(name="anime")
```

- [ ] **Step 2: Run tests to verify they fail**
```bash
python -m pytest tests/test_tiktok_service.py -v
```
Expected: `ModuleNotFoundError: No module named 'backend.services.tiktok_service'`

- [ ] **Step 3: Create `backend/services/tiktok_service.py`**

```python
import random
from dataclasses import dataclass

from TikTokApi import TikTokApi

TOPIC_HASHTAG_MAP: dict[str, str | None] = {
    "trending": None,
    "日本語": "日本語",
    "anime": "anime",
    "grammar": "日本語文法",
}


@dataclass
class TikTokVideoMeta:
    video_id: str
    title: str
    author: str


async def get_random_tiktok_video(
    topic: str,
    ms_token: str,
    count: int = 30,
) -> TikTokVideoMeta | None:
    """Fetch a random TikTok video matching `topic`. Returns None on empty results."""
    hashtag = TOPIC_HASHTAG_MAP.get(topic.lower(), None)
    videos: list[TikTokVideoMeta] = []

    async with TikTokApi() as api:
        tokens = [ms_token] if ms_token else []
        await api.create_sessions(
            ms_tokens=tokens,
            num_sessions=1,
            sleep_after=3,
            browser="chromium",
        )

        if hashtag is None:
            async for video in api.trending.videos(count=count):
                d = video.as_dict
                videos.append(TikTokVideoMeta(
                    video_id=str(d.get("id", "")),
                    title=d.get("desc", "")[:120],
                    author=d.get("author", {}).get("nickname", ""),
                ))
        else:
            async for video in api.hashtag(name=hashtag).videos(count=count):
                d = video.as_dict
                videos.append(TikTokVideoMeta(
                    video_id=str(d.get("id", "")),
                    title=d.get("desc", "")[:120],
                    author=d.get("author", {}).get("nickname", ""),
                ))

    if not videos:
        return None
    return random.choice(videos)


async def fetch_video_bytes(video_id: str, ms_token: str) -> bytes:
    """Download raw bytes for a TikTok video by ID."""
    async with TikTokApi() as api:
        tokens = [ms_token] if ms_token else []
        await api.create_sessions(
            ms_tokens=tokens,
            num_sessions=1,
            sleep_after=3,
            browser="chromium",
        )
        video = api.video(id=video_id)
        return await video.bytes()
```

- [ ] **Step 4: Run tests to verify they pass**
```bash
python -m pytest tests/test_tiktok_service.py -v
```
Expected: 4 tests PASS.

- [ ] **Step 5: Run full backend test suite**
```bash
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py
```
Expected: all existing tests still pass.

- [ ] **Step 6: Commit**
```bash
git add backend/services/tiktok_service.py tests/test_tiktok_service.py
git commit -m "feat: TikTok service — random video discovery and byte-stream download"
```

---

### Task 4: TikTok Backend Endpoints + Schemas

**Files:**
- Modify: `backend/schemas/api.py`
- Modify: `backend/routers/advanced.py`

- [ ] **Step 1: Add `TikTokVideoMeta` schema to `backend/schemas/api.py`**

Append at the bottom of `backend/schemas/api.py`:
```python
class TikTokVideoMeta(BaseModel):
    video_id: str
    title: str
    author: str
```

- [ ] **Step 2: Add TikTok imports to `backend/routers/advanced.py`**

Add to the import block (after existing imports):
```python
from fastapi.responses import StreamingResponse
from backend.schemas.api import TikTokVideoMeta
from backend.services.tiktok_service import get_random_tiktok_video, fetch_video_bytes
```

- [ ] **Step 3: Append two new routes to `backend/routers/advanced.py`**

```python
@router.get("/tiktok/random-video", response_model=TikTokVideoMeta)
async def tiktok_random_video(
    topic: str = "trending",
    current_user: User = Depends(get_current_user),
):
    """
    Return metadata for a random TikTok video matching the given topic.
    Returns 503 when TIKTOK_MS_TOKEN is not configured.
    Topics: trending, 日本語, anime, grammar
    """
    if not settings.tiktok_ms_token:
        raise HTTPException(status_code=503, detail="TikTok not configured")

    video = await get_random_tiktok_video(
        topic=topic,
        ms_token=settings.tiktok_ms_token,
    )
    if video is None:
        raise HTTPException(status_code=404, detail="No TikTok video found")

    return TikTokVideoMeta(
        video_id=video.video_id,
        title=video.title,
        author=video.author,
    )


@router.get("/tiktok/video-stream/{video_id}")
async def tiktok_video_stream(
    video_id: str,
    current_user: User = Depends(get_current_user),
):
    """
    Stream raw bytes for a TikTok video. Frontend fetches this with auth headers,
    creates a blob URL, and assigns it to an HTML5 <video> element.
    """
    if not settings.tiktok_ms_token:
        raise HTTPException(status_code=503, detail="TikTok not configured")

    video_bytes = await fetch_video_bytes(
        video_id=video_id,
        ms_token=settings.tiktok_ms_token,
    )

    async def byte_iterator():
        chunk_size = 1024 * 64
        for i in range(0, len(video_bytes), chunk_size):
            yield video_bytes[i : i + chunk_size]

    return StreamingResponse(byte_iterator(), media_type="video/mp4")
```

- [ ] **Step 4: Run the full backend test suite**
```bash
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py
```
Expected: all tests pass.

- [ ] **Step 5: Commit**
```bash
git add backend/schemas/api.py backend/routers/advanced.py
git commit -m "feat: add TikTok random-video and video-stream endpoints"
```

---

### Task 5: TikTok Frontend — Types, API Client, TikTokCard, DiscoverPage

**Files:**
- Modify: `frontend/src/types/index.ts`
- Modify: `frontend/src/api/client.ts`
- Create: `frontend/src/components/TikTokCard.tsx`
- Modify: `frontend/src/pages/DiscoverPage.tsx`

- [ ] **Step 1: Add `TikTokVideoMeta` type to `frontend/src/types/index.ts`**

Append at the bottom:
```typescript
export interface TikTokVideoMeta {
  video_id: string;
  title: string;
  author: string;
}
```

- [ ] **Step 2: Add `getRandomTikTok` to `frontend/src/api/client.ts`**

Inside the `api` object, after `getRandomSong`, before the closing `};`, add:

```typescript
  getRandomTikTok: (topic?: string) =>
    get<import("../types").TikTokVideoMeta>(
      `/api/v1/tiktok/random-video${topic ? `?topic=${encodeURIComponent(topic)}` : ""}`
    ),
```

- [ ] **Step 3: Create `frontend/src/components/TikTokCard.tsx`**

```typescript
import { useEffect, useRef, useState } from "react";
import type { TikTokVideoMeta } from "../types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

interface Props {
  video: TikTokVideoMeta;
  autoPlay?: boolean;
  delay?: number;
}

export default function TikTokCard({ video, autoPlay = false, delay = 0 }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(false);
    const token = localStorage.getItem("token");
    let objectUrl = "";

    fetch(`${BASE}/api/v1/tiktok/video-stream/${video.video_id}`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
      .then(r => {
        if (!r.ok) throw new Error("stream failed");
        return r.blob();
      })
      .then(blob => {
        objectUrl = URL.createObjectURL(blob);
        if (videoRef.current) {
          videoRef.current.src = objectUrl;
          if (autoPlay) videoRef.current.play().catch(() => {});
        }
        setLoading(false);
      })
      .catch(() => {
        setError(true);
        setLoading(false);
      });

    return () => {
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [video.video_id, autoPlay]);

  return (
    <div
      className="rounded-2xl overflow-hidden border border-cream bg-ivory animate-pop-in"
      style={{ animationDelay: `${delay}ms` }}
    >
      {loading && (
        <div className="w-full aspect-video bg-sand flex items-center justify-center animate-timer-pulse">
          <span className="text-ash text-xs">Loading video…</span>
        </div>
      )}
      {error && (
        <div className="w-full aspect-video bg-sand flex items-center justify-center">
          <span className="text-ash text-xs">Video unavailable</span>
        </div>
      )}
      <video
        ref={videoRef}
        controls
        className={`w-full aspect-video ${loading || error ? "hidden" : ""}`}
        onLoadedData={() => setLoading(false)}
      />
      <div className="px-3 py-2.5 space-y-0.5">
        <p className="text-xs font-medium text-ink leading-snug line-clamp-2">{video.title || "Untitled"}</p>
        <p className="text-xs text-ash">@{video.author}</p>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Replace `frontend/src/pages/DiscoverPage.tsx`**

```typescript
import { useState } from "react";
import { api } from "../api/client";
import type { VideoRecommendation, TikTokVideoMeta } from "../types";
import YouTubeCard from "../components/YouTubeCard";
import TikTokCard from "../components/TikTokCard";

type Source = "youtube" | "tiktok";

const YOUTUBE_TOPICS = ["Random", "J-Pop", "Anime", "Grammar"] as const;
const TIKTOK_TOPICS  = ["Trending", "日本語", "Anime", "Grammar"] as const;

const YOUTUBE_TOPIC_MAP: Record<string, string | undefined> = {
  "Random": undefined,
  "J-Pop": "J-Pop",
  "Anime": "anime opening",
  "Grammar": "JLPT grammar",
};

const TIKTOK_TOPIC_MAP: Record<string, string> = {
  "Trending": "trending",
  "日本語": "日本語",
  "Anime": "anime",
  "Grammar": "grammar",
};

export default function DiscoverPage() {
  const [source, setSource] = useState<Source>("youtube");
  const [ytTopic, setYtTopic]   = useState<string>("Random");
  const [ttTopic, setTtTopic]   = useState<string>("Trending");
  const [ytVideo, setYtVideo]   = useState<VideoRecommendation | null>(null);
  const [ttVideo, setTtVideo]   = useState<TikTokVideoMeta | null>(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");
  const [hitCount, setHitCount] = useState(0);

  const hitMe = async () => {
    setLoading(true);
    setYtVideo(null);
    setTtVideo(null);
    setError("");
    try {
      if (source === "youtube") {
        const concept = YOUTUBE_TOPIC_MAP[ytTopic];
        const v = await api.getRandomSong(concept);
        setYtVideo(v);
      } else {
        const topic = TIKTOK_TOPIC_MAP[ttTopic];
        const v = await api.getRandomTikTok(topic);
        setTtVideo(v);
      }
      setHitCount(n => n + 1);
    } catch {
      setError("Couldn't find a video right now. Try again.");
    } finally {
      setLoading(false);
    }
  };

  const activeVideo = source === "youtube" ? ytVideo : ttVideo;
  const activeTopic = source === "youtube" ? ytTopic : ttTopic;
  const topics = source === "youtube" ? YOUTUBE_TOPICS : TIKTOK_TOPICS;
  const setTopic = source === "youtube" ? setYtTopic : setTtTopic;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-serif text-2xl font-medium text-ink">Discover</h1>
        <p className="text-sm text-bark mt-1 leading-relaxed">
          Warm up with a random Japanese video before you study.
        </p>
      </div>

      {/* Source toggle */}
      <div className="flex bg-sand rounded-xl p-1 gap-1 w-fit">
        {(["youtube", "tiktok"] as Source[]).map(s => (
          <button
            key={s}
            onClick={() => { setSource(s); setError(""); }}
            className={`px-5 py-1.5 rounded-lg text-sm font-medium transition-all ${
              source === s
                ? "bg-ivory text-ink shadow-sm border border-cream"
                : "text-bark hover:text-ink"
            }`}
          >
            {s === "youtube" ? "YouTube" : "TikTok"}
          </button>
        ))}
      </div>

      {/* Topic chips */}
      <div className="flex flex-wrap gap-2">
        {topics.map(t => (
          <button
            key={t}
            onClick={() => setTopic(t)}
            className={`px-3 py-1 rounded-full text-xs font-medium border transition-all ${
              activeTopic === t
                ? "bg-brand-500 text-ivory border-brand-500"
                : "bg-ivory text-bark border-cream hover:border-brand-500"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Hit Me button */}
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

        {error && <p className="text-sm text-red-700 text-center">{error}</p>}

        {ytVideo && source === "youtube" && !loading && (
          <div className="w-full max-w-lg animate-fade-in">
            <YouTubeCard video={ytVideo} autoPlay />
          </div>
        )}

        {ttVideo && source === "tiktok" && !loading && (
          <div className="w-full max-w-lg animate-fade-in">
            <TikTokCard video={ttVideo} autoPlay />
          </div>
        )}

        {!activeVideo && !loading && hitCount === 0 && (
          <div className="text-center space-y-2 py-8">
            <p className="text-3xl">{source === "youtube" ? "🎌" : "🎵"}</p>
            <p className="text-bark text-sm">
              Hit the button to discover a Japanese {source === "youtube" ? "song" : "TikTok"}.
            </p>
            <p className="text-ash text-xs max-w-xs">
              {source === "youtube"
                ? "Songs are picked from J-pop, anime, city pop, and traditional music."
                : "Videos are picked from trending and topic-based TikTok searches."}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Run type-check**
```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 6: Run build**
```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npm run build 2>&1 | tail -6
```
Expected: build succeeds.

- [ ] **Step 7: Commit**
```bash
git add frontend/src/types/index.ts frontend/src/api/client.ts \
        frontend/src/components/TikTokCard.tsx \
        frontend/src/pages/DiscoverPage.tsx
git commit -m "feat: TikTok source toggle, topic chips, and TikTokCard self-hosted player on Discover page"
```

---

### Task 6: OCR Batch Backend Endpoint

**Files:**
- Modify: `backend/routers/quiz.py`
- Modify: `backend/schemas/api.py`
- Create: `tests/test_ocr_batch.py`

- [ ] **Step 1: Add `OcrBatchResponse` schema to `backend/schemas/api.py`**

Append at the bottom:
```python
class OcrBatchResponse(BaseModel):
    questions: list[PracticeQuestion]
```

- [ ] **Step 2: Write failing tests — create `tests/test_ocr_batch.py`**

```python
import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.auth.dependencies import get_current_user
from backend.models.tables import User


def _fake_user():
    u = User()
    u.id = 1
    u.email = "test@example.com"
    return u


def _mock_jwt():
    app.dependency_overrides[get_current_user] = lambda: _fake_user()


def _clear_jwt():
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ocr_batch_returns_questions():
    _mock_jwt()
    try:
        from PIL import Image
        import io, base64

        # Create a tiny 10x10 white image
        img = Image.new("RGB", (10, 10), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        structured = {"question": "彼女は日本語を＿＿います。", "options": ["A: 話して", "B: 話した", "C: 話す", "D: 話せ"], "answer": "A"}

        with patch("backend.routers.quiz.extract_text", return_value={"question": "raw text", "options": []}):
            with patch("backend.routers.quiz._structure_ocr_with_llm", new_callable=AsyncMock, return_value=structured):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/quiz/ocr/batch",
                        files={"images": ("test.png", buf.getvalue(), "image/png")},
                    )

        assert response.status_code == 200
        data = response.json()
        assert "questions" in data
        assert len(data["questions"]) == 1
        assert data["questions"][0]["question"] == "彼女は日本語を＿＿います。"
    finally:
        _clear_jwt()


@pytest.mark.asyncio
async def test_ocr_batch_skips_failed_structure():
    _mock_jwt()
    try:
        from PIL import Image
        import io

        img = Image.new("RGB", (10, 10), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        with patch("backend.routers.quiz.extract_text", return_value={"question": "garbage", "options": []}):
            with patch("backend.routers.quiz._structure_ocr_with_llm", new_callable=AsyncMock, return_value=None):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/quiz/ocr/batch",
                        files={"images": ("test.png", buf.getvalue(), "image/png")},
                    )

        assert response.status_code == 200
        assert response.json()["questions"] == []
    finally:
        _clear_jwt()
```

- [ ] **Step 3: Run tests to verify they fail**
```bash
python -m pytest tests/test_ocr_batch.py -v
```
Expected: import errors or `404` for the missing endpoint.

- [ ] **Step 4: Update `backend/routers/quiz.py` — add OCR batch endpoint**

Replace the entire file with:
```python
import base64
import io
import json
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse
from PIL import Image

from backend.schemas.api import SolveRequest, OcrBatchResponse, PracticeQuestion
from backend.services.llm.factory import get_llm_provider
from backend.services.quiz import solve_question, parse_result_block
from backend.services.ocr import extract_text
from backend.auth.dependencies import get_current_user, get_current_user_optional
from backend.models.tables import User

router = APIRouter(prefix="/api/v1")

_STRUCTURE_PROMPT = """You are parsing Japanese grammar question text extracted via OCR from a screenshot.
The raw text may contain noise or imperfect line breaks.

Raw OCR text:
{raw_text}

Output a JSON object with exactly these keys:
- "question": the question stem (Japanese text)
- "options": array of exactly 4 strings, each prefixed like ["A: ...", "B: ...", "C: ...", "D: ..."]
- "answer": one of "A", "B", "C", "D" (guess "A" if not determinable)

Output ONLY the JSON object. No other text, no markdown fences."""


async def _structure_ocr_with_llm(raw_text: str, provider) -> dict | None:
    """Ask the LLM to parse raw OCR text into structured question format."""
    try:
        response = await provider.complete(_STRUCTURE_PROMPT.format(raw_text=raw_text))
        cleaned = response.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)
        if not all(k in parsed for k in ("question", "options", "answer")):
            return None
        if len(parsed["options"]) != 4:
            return None
        return parsed
    except Exception:
        return None


@router.post("/quiz/solve")
async def solve(
    body: SolveRequest,
    current_user: User | None = Depends(get_current_user_optional),
):
    provider = get_llm_provider()
    stream = provider.stream_solve(body.question, body.options)

    async def event_generator():
        full_text = ""
        async for token in solve_question(body.question, body.options, stream):
            full_text += token
            yield {"event": "token", "data": token}

        result = parse_result_block(full_text)
        if result:
            yield {
                "event": "result",
                "data": json.dumps(
                    {
                        "answer": result.answer,
                        "explanation": result.explanation,
                        "wrong_options": result.wrong_options,
                        "concepts": result.concepts,
                    }
                ),
            }
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@router.post("/ocr/extract")
async def ocr_extract(
    image_data: str = Form(...),
    x1: int = Form(...),
    y1: int = Form(...),
    x2: int = Form(...),
    y2: int = Form(...),
    num_options: int = Form(4),
):
    try:
        result = extract_text(image_data, x1, y1, x2, y2, num_options)
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"OCR failed: {e}")


@router.post("/quiz/ocr/batch", response_model=OcrBatchResponse)
async def ocr_batch(
    images: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Accept multiple screenshot uploads, OCR each, LLM-structure into questions.
    Failed extractions are silently skipped.
    """
    provider = get_llm_provider()
    questions: list[PracticeQuestion] = []

    for upload in images:
        img_bytes = await upload.read()
        try:
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
            b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
            ocr_result = extract_text(b64, 0, 0, w, h, 4)
            raw_text = ocr_result.get("question", "") + " " + " ".join(ocr_result.get("options", []))
            structured = await _structure_ocr_with_llm(raw_text.strip(), provider)
            if structured:
                questions.append(PracticeQuestion(
                    question=structured["question"],
                    options=structured["options"],
                    correct_answer=structured["answer"],
                ))
        except Exception:
            continue

    return OcrBatchResponse(questions=questions)
```

- [ ] **Step 5: Run tests to verify they pass**
```bash
python -m pytest tests/test_ocr_batch.py -v
```
Expected: 2 tests PASS.

- [ ] **Step 6: Run full backend test suite**
```bash
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py
```
Expected: all tests pass.

- [ ] **Step 7: Commit**
```bash
git add backend/routers/quiz.py backend/schemas/api.py tests/test_ocr_batch.py
git commit -m "feat: OCR batch endpoint — multi-screenshot upload to structured quiz questions"
```

---

### Task 7: QuizSetup — Three-Mode Toggle (CSV | Screenshots | AI Generate)

**Files:**
- Modify: `frontend/src/components/QuizSetup.tsx`
- Modify: `frontend/src/api/client.ts`

- [ ] **Step 1: Add `uploadOcrImages` to `frontend/src/api/client.ts`**

Inside the `api` object, after `getRandomTikTok`, before the closing `};`:
```typescript
  uploadOcrImages: (images: File[]) => {
    const form = new FormData();
    images.forEach(f => form.append("images", f));
    return fetch(`${BASE}/api/v1/quiz/ocr/batch`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    }).then(async r => {
      if (!r.ok) { const e = await r.json().catch(() => ({ detail: "OCR failed" })); throw new Error(e.detail); }
      return r.json() as Promise<{ questions: import("../types").QuizQuestion[] }>;
    });
  },
```

- [ ] **Step 2: Replace `frontend/src/components/QuizSetup.tsx`**

```typescript
import { useRef, useState } from "react";
import { api } from "../api/client";
import type { QuizQuestion } from "../types";

type Mode = "csv" | "screenshots" | "generate";
type Level = "N5" | "N4" | "N3" | "N2" | "N1";

interface Props {
  onStart: (questions: QuizQuestion[], timeSec: number) => void;
}

export default function QuizSetup({ onStart }: Props) {
  const [mode, setMode] = useState<Mode>("csv");

  // Shared
  const [questions, setQuestions]   = useState<QuizQuestion[]>([]);
  const [minutes, setMinutes]       = useState(10);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState("");
  const [maxQuestions, setMaxQuestions] = useState(0);

  // CSV mode
  const inputRef   = useRef<HTMLInputElement>(null);
  const [fileName, setFileName]     = useState("");
  const [includeHistory, setIncludeHistory] = useState(false);
  const [historyCount, setHistoryCount]     = useState(5);

  // Screenshots mode
  const screenshotRef = useRef<HTMLInputElement>(null);
  const [screenshotFiles, setScreenshotFiles] = useState<File[]>([]);

  // Generate mode
  const [level, setLevel]           = useState<Level>("N4");
  const [concept, setConcept]       = useState("");
  const [genCount, setGenCount]     = useState(5);
  const [genProgress, setGenProgress] = useState(0);

  // --- CSV handlers ---
  const handleCSVFile = async (file: File) => {
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

  const handleCSVDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleCSVFile(file);
  };

  // --- Screenshots handlers ---
  const handleScreenshots = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const arr = Array.from(files);
    setScreenshotFiles(arr);
    setLoading(true); setError(""); setQuestions([]);
    try {
      const res = await api.uploadOcrImages(arr);
      setQuestions(res.questions);
    } catch (err) {
      setError(err instanceof Error ? err.message : "OCR failed");
    } finally {
      setLoading(false);
    }
  };

  const handleScreenshotDrop = (e: React.DragEvent) => {
    e.preventDefault();
    handleScreenshots(e.dataTransfer.files);
  };

  // --- Generate handlers ---
  const handleGenerate = async () => {
    setLoading(true); setError(""); setQuestions([]); setGenProgress(0);
    const generated: QuizQuestion[] = [];
    try {
      await Promise.all(
        Array.from({ length: genCount }).map(async (_, i) => {
          const q = await api.generateQuestion(concept || "grammar", level);
          generated.push({
            question: q.question,
            options: q.options,
            correct_answer: q.correct_answer,
          });
          setGenProgress(p => p + 1);
        })
      );
      setQuestions(generated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  const handleStart = () => {
    const qs = maxQuestions > 0 ? questions.slice(0, maxQuestions) : questions;
    onStart(qs, minutes * 60);
  };

  const MODES: { id: Mode; label: string }[] = [
    { id: "csv",         label: "Upload CSV" },
    { id: "screenshots", label: "Screenshots" },
    { id: "generate",    label: "Generate with AI" },
  ];

  const LEVELS: Level[] = ["N5", "N4", "N3", "N2", "N1"];

  return (
    <div className="space-y-4 animate-fade-in">
      <div>
        <h1 className="font-serif text-xl font-medium text-ink">Timed Quiz</h1>
        <p className="text-xs text-ash mt-1">
          Build a question set, then test yourself under timed conditions. AI reviews every mistake after.
        </p>
      </div>

      {/* Mode toggle */}
      <div className="flex bg-sand rounded-xl p-1 gap-1">
        {MODES.map(m => (
          <button
            key={m.id}
            onClick={() => { setMode(m.id); setQuestions([]); setError(""); }}
            className={`flex-1 py-1.5 rounded-lg text-xs font-medium transition-all ${
              mode === m.id
                ? "bg-ivory text-ink shadow-sm border border-cream"
                : "text-bark hover:text-ink"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {/* --- CSV mode --- */}
      {mode === "csv" && (
        <>
          <div
            onDrop={handleCSVDrop}
            onDragOver={e => e.preventDefault()}
            onClick={() => inputRef.current?.click()}
            className="border-2 border-dashed border-cream rounded-2xl p-8 text-center cursor-pointer
              hover:border-brand-500 transition-colors active:scale-[0.99] bg-ivory"
          >
            <input ref={inputRef} type="file" accept=".csv" className="hidden"
              onChange={e => e.target.files?.[0] && handleCSVFile(e.target.files[0])} />
            {loading ? (
              <p className="text-sm text-brand-500 animate-timer-pulse">Parsing CSV…</p>
            ) : fileName ? (
              <div>
                <p className="text-sm font-medium text-ink">{fileName}</p>
                <p className="text-xs text-green-700 mt-1">{questions.length} questions loaded</p>
              </div>
            ) : (
              <div>
                <p className="text-sm text-ash">Drop CSV here or click to upload</p>
                <p className="text-xs text-ash/60 mt-1">Format: Question, Options, Answer</p>
              </div>
            )}
          </div>

          {/* CSV config */}
          <div className="bg-ivory border border-cream rounded-2xl p-5 space-y-4 shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-ink">Mix in wrong history</p>
                <p className="text-xs text-ash mt-0.5">Add past wrong answers to practice</p>
              </div>
              <button
                onClick={() => setIncludeHistory(v => !v)}
                role="switch"
                aria-checked={includeHistory}
                className={`relative w-10 h-5 rounded-full transition-colors ${includeHistory ? "bg-brand-500" : "bg-sand"}`}
              >
                <span className={`absolute top-0.5 w-4 h-4 bg-white rounded-full shadow transition-transform
                  ${includeHistory ? "translate-x-5" : "translate-x-0.5"}`} />
              </button>
            </div>
            {includeHistory && (
              <div className="flex items-center justify-between animate-fade-in">
                <p className="text-xs text-bark">Wrong questions to add</p>
                <div className="flex items-center gap-2">
                  <button onClick={() => setHistoryCount(n => Math.max(1, n - 1))}
                    className="w-6 h-6 rounded border border-cream bg-sand text-xs text-charcoal hover:bg-sand/80 active:scale-95 transition-all">−</button>
                  <span className="text-sm font-semibold text-ink w-6 text-center">{historyCount}</span>
                  <button onClick={() => setHistoryCount(n => Math.min(20, n + 1))}
                    className="w-6 h-6 rounded border border-cream bg-sand text-xs text-charcoal hover:bg-sand/80 active:scale-95 transition-all">+</button>
                </div>
              </div>
            )}
          </div>
        </>
      )}

      {/* --- Screenshots mode --- */}
      {mode === "screenshots" && (
        <div
          onDrop={handleScreenshotDrop}
          onDragOver={e => e.preventDefault()}
          onClick={() => screenshotRef.current?.click()}
          className="border-2 border-dashed border-cream rounded-2xl p-8 text-center cursor-pointer
            hover:border-brand-500 transition-colors active:scale-[0.99] bg-ivory"
        >
          <input ref={screenshotRef} type="file" accept="image/*" multiple className="hidden"
            onChange={e => handleScreenshots(e.target.files)} />
          {loading ? (
            <p className="text-sm text-brand-500 animate-timer-pulse">Running OCR…</p>
          ) : screenshotFiles.length > 0 ? (
            <div>
              <p className="text-sm font-medium text-ink">{screenshotFiles.length} image{screenshotFiles.length > 1 ? "s" : ""} uploaded</p>
              <p className="text-xs text-green-700 mt-1">{questions.length} questions extracted</p>
            </div>
          ) : (
            <div>
              <p className="text-sm text-ash">Drop screenshots here or click to upload</p>
              <p className="text-xs text-ash/60 mt-1">PNG or JPG — one question per image</p>
            </div>
          )}
        </div>
      )}

      {/* --- Generate with AI mode --- */}
      {mode === "generate" && (
        <div className="bg-ivory border border-cream rounded-2xl p-5 space-y-4 shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
          {/* Level */}
          <div>
            <p className="text-sm font-medium text-ink mb-2">JLPT Level</p>
            <div className="flex gap-2">
              {LEVELS.map(l => (
                <button
                  key={l}
                  onClick={() => setLevel(l)}
                  className={`flex-1 py-1.5 rounded-lg text-xs font-semibold border transition-all ${
                    level === l
                      ? "bg-brand-500 text-ivory border-brand-500"
                      : "bg-sand text-bark border-cream hover:border-brand-500"
                  }`}
                >
                  {l}
                </button>
              ))}
            </div>
          </div>

          {/* Concept */}
          <div>
            <p className="text-sm font-medium text-ink mb-1">Grammar concept <span className="text-ash font-normal">(optional)</span></p>
            <input
              type="text"
              value={concept}
              onChange={e => setConcept(e.target.value)}
              placeholder="e.g. て-form, は vs が, conditional"
              className="w-full px-3 py-2 rounded-xl border border-cream bg-parchment text-sm text-ink
                placeholder:text-ash/60 focus:outline-none focus:border-brand-500 transition-colors"
            />
          </div>

          {/* Count */}
          <div className="flex items-center justify-between border-t border-cream pt-4">
            <div>
              <p className="text-sm font-medium text-ink">Questions to generate</p>
            </div>
            <div className="flex items-center gap-2">
              <button onClick={() => setGenCount(n => Math.max(1, n - 1))}
                className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">−</button>
              <span className="text-sm font-semibold text-ink w-8 text-center">{genCount}</span>
              <button onClick={() => setGenCount(n => Math.min(20, n + 1))}
                className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">+</button>
            </div>
          </div>

          <button
            onClick={handleGenerate}
            disabled={loading}
            className="w-full bg-brand-500 hover:bg-brand-600 active:scale-[0.98] text-ivory py-2 rounded-xl
              font-semibold text-sm transition-all disabled:opacity-40"
          >
            {loading
              ? `Generating ${genProgress} of ${genCount}…`
              : questions.length > 0
              ? `Regenerate (${questions.length} ready)`
              : "Generate Questions"}
          </button>

          {questions.length > 0 && !loading && (
            <p className="text-xs text-green-700 text-center">{questions.length} questions ready</p>
          )}
        </div>
      )}

      {error && <p className="text-red-700 text-xs">{error}</p>}

      {/* Shared: timer + max questions */}
      <div className="bg-ivory border border-cream rounded-2xl p-5 space-y-4 shadow-[rgba(0,0,0,0.05)_0px_4px_24px]">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-ink">Time limit</p>
            <p className="text-xs text-ash mt-0.5">Set 0 for unlimited</p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setMinutes(m => Math.max(0, m - 5))}
              aria-label="Decrease time limit"
              className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">−</button>
            <span className="text-sm font-semibold text-ink w-12 text-center">
              {minutes === 0 ? "∞" : `${minutes}m`}
            </span>
            <button onClick={() => setMinutes(m => m + 5)}
              aria-label="Increase time limit"
              className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">+</button>
          </div>
        </div>

        <div className="flex items-center justify-between border-t border-cream pt-4">
          <div>
            <p className="text-sm font-medium text-ink">Max questions</p>
            <p className="text-xs text-ash mt-0.5">Set 0 for all</p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setMaxQuestions(n => Math.max(0, n - 5))}
              aria-label="Decrease max questions"
              className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">−</button>
            <span className="text-sm font-semibold text-ink w-12 text-center">
              {maxQuestions === 0 ? "All" : `${maxQuestions}`}
            </span>
            <button onClick={() => setMaxQuestions(n => n + 5)}
              aria-label="Increase max questions"
              className="w-7 h-7 rounded-lg border border-cream bg-sand text-charcoal hover:bg-sand/80 active:scale-95 transition-all text-sm flex items-center justify-center">+</button>
          </div>
        </div>
      </div>

      <button
        disabled={questions.length === 0}
        onClick={handleStart}
        className="w-full bg-brand-500 hover:bg-brand-600 active:scale-[0.98] text-ivory py-2.5 rounded-xl
          font-semibold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {questions.length > 0
          ? `Start Quiz — ${maxQuestions > 0 ? Math.min(maxQuestions, questions.length) : questions.length} questions`
          : "Load questions to start"}
      </button>
    </div>
  );
}
```

- [ ] **Step 3: Run type-check**
```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 4: Run build**
```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024/frontend && npm run build 2>&1 | tail -6
```
Expected: build succeeds.

- [ ] **Step 5: Commit**
```bash
git add frontend/src/components/QuizSetup.tsx frontend/src/api/client.ts
git commit -m "feat: QuizSetup three-mode toggle — CSV, Screenshots OCR, and AI Generate"
```

---

### Task 8: README Revamp + Legacy Cleanup

**Files:**
- Delete: `backend.py`, `frontend.html`, `model.py`, `progress_manager.py`, `test.ipynb`, `requirements.txt` (root), `DESIGN.md`
- Delete dirs: `static/`, `Evaluation/`, `ImageSamples/`, `NASC_2024_Team_LLMers/`
- Rewrite: `readme.md`

- [ ] **Step 1: Delete legacy files**
```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024
rm -f backend.py frontend.html model.py progress_manager.py test.ipynb requirements.txt DESIGN.md
rm -rf static/ Evaluation/ ImageSamples/ NASC_2024_Team_LLMers/
```

- [ ] **Step 2: Rewrite `readme.md`**

```bash
cat > readme.md << 'EOF'
# JLPT Sensei

AI-powered JLPT Japanese grammar prep — ask questions, take timed quizzes, review mistakes, and discover Japanese content.

## Features

- **Ask** — paste a grammar question, get an AI explanation of every option
- **Quiz** — timed quiz mode with three ways to build a question set: upload CSV, scan screenshots (OCR), or generate with AI (choose JLPT level + concept)
- **History** — browse past attempts, filter by wrong answers
- **Stats** — accuracy, study streak, weak concepts with one-click quiz generation
- **Discover** — hit "Hit Me" to get a random Japanese YouTube video or TikTok, filtered by topic

## Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, SQLAlchemy 2.0 async, aiosqlite |
| LLM | OpenRouter (default) or local HuggingFace model |
| Frontend | React 18, Vite 5, TypeScript, Tailwind CSS v4 |
| Database | SQLite (dev) — swap to PostgreSQL via `DATABASE_URL` |

## Quick Start

### Backend

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r backend/requirements/base.txt
python -m playwright install chromium               # for TikTok integration
cp .env.example .env                                # fill in keys
uvicorn backend.main:app --reload --app-dir .
# API at http://127.0.0.1:8000  |  Swagger at http://127.0.0.1:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key |
| `OPENROUTER_MODEL` | No | Model ID (default: `openai/gpt-4o-mini`) |
| `SECRET_KEY` | Yes (prod) | JWT signing secret |
| `DATABASE_URL` | No | SQLAlchemy URL (default: SQLite) |
| `YOUTUBE_API_KEY` | No | YouTube Data API v3 key — Discover YouTube feature |
| `TIKTOK_MS_TOKEN` | No | TikTok `msToken` cookie — Discover TikTok feature |

Both `YOUTUBE_API_KEY` and `TIKTOK_MS_TOKEN` are optional. The respective Discover sources show an empty state when not configured.

## TikTok Setup

The TikTok integration uses the unofficial [TikTok-Api](https://github.com/davidteather/TikTok-api) library (Playwright-based). After installing dependencies:

```bash
python -m playwright install chromium
```

To get your `msToken`, log into TikTok in a browser and copy the `msToken` cookie value into `.env`.

## Tests

```bash
# Unit + integration (no server needed)
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py

# E2E smoke (backend must be running)
python -m pytest tests/test_e2e_smoke.py -v
```
EOF
```

- [ ] **Step 3: Run full backend test suite to confirm nothing broke**
```bash
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py
```
Expected: all tests pass.

- [ ] **Step 4: Commit**
```bash
git add -A
git commit -m "chore: delete legacy competition files and rewrite README for current product"
```

---

## Self-Review

### Spec Coverage

| Spec requirement | Task |
|---|---|
| `*` route → `/discover` | Task 1 |
| Nav reordered (Discover first) | Task 1 |
| YouTube/TikTok source toggle on Discover | Task 5 (DiscoverPage) |
| Per-source topic chips | Task 5 (DiscoverPage) |
| TikTok backend service (trending + hashtag) | Task 3 |
| TikTok random-video endpoint | Task 4 |
| TikTok video-stream endpoint (StreamingResponse) | Task 4 |
| TikTokCard HTML5 video with blob URL auth | Task 5 |
| `tiktok_ms_token` in config | Task 2 |
| 503 graceful degradation when token absent | Task 4 |
| QuizSetup CSV mode (unchanged) | Task 7 |
| QuizSetup Screenshots mode + OCR batch | Tasks 6, 7 |
| QuizSetup Generate with AI (level + concept + count) | Task 7 |
| OCR batch endpoint with LLM structuring | Task 6 |
| README rewrite | Task 8 |
| Legacy file deletion | Task 8 |

### Type Consistency

- `TikTokVideoMeta`: defined in `backend/services/tiktok_service.py` as dataclass, mirrored as Pydantic schema in `backend/schemas/api.py`, mirrored as TS interface in `frontend/src/types/index.ts` — all three have `video_id`, `title`, `author` ✓
- `OcrBatchResponse.questions: list[PracticeQuestion]` matches `{ questions: QuizQuestion[] }` in frontend ✓
- `api.uploadOcrImages` returns `{ questions: QuizQuestion[] }` matching `OcrBatchResponse` ✓
- `api.getRandomTikTok` returns `TikTokVideoMeta` matching `TikTokCard` props ✓
- `_structure_ocr_with_llm` referenced in test mock matches function name in `quiz.py` ✓
