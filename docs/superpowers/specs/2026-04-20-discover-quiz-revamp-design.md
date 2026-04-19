# Discover Revamp, TikTok Integration, Quiz Generation & Cleanup

**Date:** 2026-04-20
**Scope:** Four coordinated changes to LLM Sensei.

---

## 1. Routing & Landing Page

**Change:** The catch-all `*` route in `frontend/src/App.tsx` redirects to `/discover` instead of `/ask`. After login the user lands on Discover. Nav order updated: Discover → Ask → Quiz → History → Stats.

**Files:** `frontend/src/App.tsx`, `frontend/src/components/Layout.tsx`

---

## 2. Discover Page — YouTube + TikTok with Source Toggle

### UI

Single `/discover` page, top-to-bottom:

1. **Source toggle** — `YouTube | TikTok` pill (brand-500 active state). Persists in component state only (no URL param needed).
2. **Topic selector** — chips per source:
   - YouTube: Random · J-Pop · Anime · Grammar
   - TikTok: Trending · 日本語 · Anime · Grammar
3. **"Hit Me" / "Another One" button** — fires the selected source's endpoint.
4. **Player area:**
   - YouTube → existing `YouTubeCard` (click-to-embed iframe, autoPlay on hit)
   - TikTok → new `TikTokCard` with HTML5 `<video>` whose `src` points to the backend streaming endpoint; shows title + creator below

### Backend — TikTok

**New service:** `backend/services/tiktok_service.py`
- Uses `TikTokApi` (davidteather, Playwright-based)
- `get_random_tiktok_video(topic: str) -> TikTokVideoMeta` — searches trending or hashtag, picks a random result, returns metadata (video_id, title, author)
- `stream_video_bytes(video_id: str) -> AsyncIterator[bytes]` — downloads video bytes via `video.bytes()`

**New endpoints in `backend/routers/advanced.py`:**

```
GET /api/v1/tiktok/random-video?topic=trending
→ { video_id, title, author }

GET /api/v1/tiktok/video-stream/{video_id}
→ StreamingResponse (video/mp4), auth required
```

`random-video` fetches metadata only (fast). The `<video src>` points to `video-stream/{video_id}` — the actual bytes stream on demand, avoiding embedding large payloads in JSON.

**Config:** `TIKTOK_MS_TOKEN` added to `.env` and `backend/config.py` (optional; feature returns 503 when absent).

**New Pydantic schema:** `TikTokVideoMeta(video_id, title, author)` in `backend/schemas/api.py`.

**Playwright setup:** `pip install TikTokApi && python -m playwright install chromium`. Documented in README.

### Frontend

**New component:** `frontend/src/components/TikTokCard.tsx`
- Props: `{ video: TikTokVideoMeta; autoPlay?: boolean }`
- Because HTML5 `<video src>` cannot send JWT auth headers, the component fetches the stream via `fetch` with the `Authorization` header → receives the full response blob → creates an `objectURL` → assigns to `video.src`. The object URL is revoked on component unmount.
- Shows title + author below, matches warm parchment design tokens

**Updated `frontend/src/pages/DiscoverPage.tsx`:**
- Source toggle state + topic state
- Calls `api.getRandomSong(topic)` (YouTube) or `api.getRandomTikTok(topic)` (TikTok)
- Renders `YouTubeCard` or `TikTokCard` based on active source

**New API client method:** `api.getRandomTikTok(topic?: string)` → `GET /api/v1/tiktok/random-video`

**New type:** `TikTokVideoMeta` in `frontend/src/types/index.ts`

### Graceful degradation
- No `TIKTOK_MS_TOKEN` → backend returns 503 → frontend shows "TikTok not configured" empty state
- Playwright not installed → backend raises on startup log, endpoint returns 503

---

## 3. QuizSetup — Three Input Modes

### UI

A 3-way toggle at the top of `QuizSetup`: `Upload CSV | Screenshots | Generate with AI`

#### Mode A: Upload CSV (unchanged)
Existing drag-drop CSV flow, no changes.

#### Mode B: Screenshots → OCR → Question Set

- Multi-image drag-drop zone (PNG/JPG, multiple files)
- On upload: calls new `POST /api/v1/quiz/ocr/batch`
- Shows a preview list of detected questions (question text + options); user can delete bad extractions
- "Use these questions" button feeds them into the quiz flow

**New backend endpoint:** `POST /api/v1/quiz/ocr/batch`
- Accepts `multipart/form-data` with `images: list[UploadFile]`
- For each image: decodes bytes → reads dimensions with Pillow → base64-encodes → calls `extract_text(b64, 0, 0, width, height, num_options=4)` to OCR full image → passes raw OCR text through LLM to structure into `{question, options, answer}`
- Returns `{ questions: QuizQuestion[] }`
- Reuses existing `backend/services/ocr.py` and `LLMProvider.complete()`

#### Mode C: Generate with AI

- JLPT level pills: N5 · N4 · N3 · N2 · N1 (default N4)
- Optional concept text input (e.g. "て-form", "は vs が")
- Question count stepper: 1–20, default 5
- "Generate" button → calls `POST /api/v1/generate` N times in parallel
- Progress indicator: "Generating 3 of 5…"
- Generated questions appear in a preview list (same as Screenshots mode)
- Start Quiz activates once all N complete

**No backend changes needed for Mode C** — `POST /api/v1/generate` already exists.

---

## 4. README Revamp + Legacy Cleanup

### Delete (legacy competition artifacts)
- Root files: `backend.py`, `frontend.html`, `model.py`, `progress_manager.py`, `test.ipynb`, `requirements.txt`
- Directories: `static/`, `Evaluation/`, `ImageSamples/`, `NASC_2024_Team_LLMers/`

### Rewrite `readme.md`
New README covers:
- What LLM Sensei is (JLPT prep platform, brief feature list)
- Stack (FastAPI, SQLite, React 18, Vite, Tailwind)
- Quick start: backend + frontend commands
- Environment variables: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `YOUTUBE_API_KEY`, `TIKTOK_MS_TOKEN` (with notes on which are required vs optional)
- Playwright setup for TikTok (`python -m playwright install chromium`)
- Brief API overview (current endpoints only)

---

## File Map

| File | Action |
|------|--------|
| `backend/services/tiktok_service.py` | CREATE |
| `backend/schemas/api.py` | MODIFY — add TikTokVideoMeta |
| `backend/routers/advanced.py` | MODIFY — add 2 TikTok endpoints |
| `backend/routers/quiz.py` | MODIFY — add OCR batch endpoint |
| `backend/config.py` | MODIFY — add tiktok_ms_token |
| `backend/requirements/base.txt` | MODIFY — add TikTokApi |
| `frontend/src/types/index.ts` | MODIFY — add TikTokVideoMeta |
| `frontend/src/api/client.ts` | MODIFY — add getRandomTikTok |
| `frontend/src/components/TikTokCard.tsx` | CREATE |
| `frontend/src/pages/DiscoverPage.tsx` | MODIFY — source toggle + topic chips |
| `frontend/src/components/QuizSetup.tsx` | MODIFY — 3-mode toggle |
| `frontend/src/App.tsx` | MODIFY — default route to /discover |
| `frontend/src/components/Layout.tsx` | MODIFY — nav order |
| `readme.md` | REWRITE |
| Legacy files/dirs | DELETE |

---

## Prerequisites

- `TIKTOK_MS_TOKEN` from TikTok browser cookies (optional; feature degrades without it)
- `python -m playwright install chromium` after installing TikTokApi
- `YOUTUBE_API_KEY` already required for existing Discover functionality
