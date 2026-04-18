# YouTube Integration Design

**Date:** 2026-04-18
**Status:** Approved

## Summary

Integrate YouTube Data API v3 into LLM Sensei to surface tutorial videos for wrong answers and enable a random Japanese song discovery feature. The backend service layer (`youtube_query.py`) already exists; this spec covers wiring it to new API endpoints and building the frontend UI.

---

## Backend

### New Endpoints (added to `backend/routers/advanced.py`)

#### `POST /api/v1/youtube/tutorials`
- **Auth:** Required
- **Body:**
  ```json
  {
    "wrong_items": [
      {
        "question": "string",
        "correct_answer": "string",
        "user_answer": "string",
        "concepts": ["string"]
      }
    ]
  }
  ```
- **Response:** Array of concept groups, each with up to 2 video results.
  ```json
  [
    {
      "question_snippet": "string",
      "concepts": ["string"],
      "search_query": "string",
      "videos": [
        {
          "video_id": "string",
          "title": "string",
          "thumbnail_url": "string",
          "channel_title": "string"
        }
      ]
    }
  ]
  ```
- **Service call:** `search_tutorials_for_wrong_answers(wrong_items, api_key, provider, max_per_item=2, max_items=3)`
- **Quota:** Max 3 YouTube search calls per request (deduped by concept).

#### `GET /api/v1/youtube/song`
- **Auth:** Required
- **Query param:** `concept` (optional) — biases song selection toward grammar-relevant music
- **Response:**
  ```json
  {
    "video_id": "string",
    "title": "string",
    "thumbnail_url": "string",
    "channel_title": "string"
  }
  ```
  Returns `null` if no video found.
- **Service call:** `get_random_japanese_song(api_key, concept)`

### New Schemas (added to `backend/schemas/api.py`)

- `TutorialWrongItem` — input item shape for tutorials endpoint
- `TutorialConceptGroup` — one concept group in tutorials response
- `TutorialRecommendRequest` — wrapper for `wrong_items` list
- `TutorialRecommendResponse` — list of `TutorialConceptGroup`
- `SongResponse` — single video result (or null)

### Unchanged

- `backend/services/youtube_query.py` — no changes needed
- `backend/services/recommendations.py` — no changes needed
- `backend/config.py` — `settings.youtube_api_key` already present

---

## Frontend

### New Component: `YouTubeVideoCard`

**File:** `frontend/src/components/YouTubeVideoCard.tsx`

Reusable card displaying:
- Thumbnail image
- Video title
- Channel name
- Clicking opens `https://youtube.com/watch?v={video_id}` in a new tab

Styled with existing design system: `rounded-2xl`, `bg-ivory`, `border-cream`, `text-ink`.

---

### New Component: `YouTubeTutorialPanel`

**File:** `frontend/src/components/YouTubeTutorialPanel.tsx`

- Accepts `wrongItems` prop (same shape as `TutorialWrongItem[]`)
- Renders a "Find YouTube tutorials" button
- On click: calls `POST /api/v1/youtube/tutorials`, shows skeleton loaders during fetch
- Renders concept groups with `YouTubeVideoCard`s per group
- If response is empty: shows "No tutorials found for these concepts"
- Error state: shows retry option

**Used in:**
- `AskPage.tsx` — replaces current `RecommendationPanel` when `result` exists. Since `AskPage` does not have an explicit correct_answer field to compare against, show the panel for all results (the user can judge relevance). The panel button label is "Find related YouTube tutorials".
- `QuizResults.tsx` — added below wrong-answer analysis section; fetches once on mount when wrong count > 0

---

### New Page: `/discover`

**File:** `frontend/src/pages/DiscoverPage.tsx`

Layout:
- Header: "日本語 Discovery" (serif, matches design system) with subtitle "Find your next Japanese obsession"
- Large centered "Hit Me 🎲" button (brand-500, rounded-xl)
- On click: calls `GET /api/v1/youtube/song`, renders one `YouTubeVideoCard` with parchment/sand background treatment
- Spinner on button during fetch
- "Hit Me again →" text link beneath the card after a result
- If API returns null: "Couldn't find a song — try again!" with retry

**Nav:** Add "Discover" entry to `Layout.tsx` nav between Stats and end of nav list.

---

### API Client (`frontend/src/api/client.ts`)

Two new methods:

```typescript
getTutorials: (wrongItems: TutorialWrongItem[]) =>
  post<TutorialConceptGroup[]>("/api/v1/youtube/tutorials", { wrong_items: wrongItems }, true),

getRandomSong: (concept?: string) =>
  get<VideoRecommendation | null>(
    `/api/v1/youtube/song${concept ? `?concept=${encodeURIComponent(concept)}` : ""}`
  ),
```

New type `TutorialWrongItem` added to `frontend/src/types/index.ts`.

---

## Data Flow

```
User gets wrong answer
  → QuizResults mounts (wrong count > 0)
  → auto-calls POST /api/v1/youtube/tutorials
  → backend: LLM generates search query per concept
  → backend: YouTube Data API search per concept (max 3 calls)
  → frontend: renders YouTubeTutorialPanel with grouped cards

User clicks "Hit Me" on /discover
  → GET /api/v1/youtube/song (optional concept param)
  → backend: picks random query from SONG_QUERIES or concept-biased query
  → YouTube Data API search (1 call, max_results=10)
  → backend: returns random pick from results
  → frontend: renders single YouTubeVideoCard
```

---

## Error Handling

| Scenario | Backend | Frontend |
|---|---|---|
| API key missing | `search_youtube` returns `[]` | Panel shows "No tutorials found" |
| Quota exceeded (429) | `httpx` raises → caught, returns `[]` | Same as above |
| No videos found | Returns empty list / null | Retry prompt shown |
| LLM query fallback | Falls back to `"JLPT {concept} grammar explanation"` | Transparent to user |

---

## Quota Estimate

- Tutorials: max 3 search calls per quiz session = 3 units
- Song: 1 search call per "Hit Me" = 1 unit
- Free tier: 10,000 units/day → comfortable for dev/demo scale

No caching in this iteration. A TTL in-memory cache can be added to the service layer later without changing the API contract.

---

## Out of Scope

- YouTube iframe embeds (linking to YouTube is sufficient for v1)
- Persisting recommendations to the database
- Rate limiting per user
- Caching layer
