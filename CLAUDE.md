# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend
```bash
# Install deps
pip install -r backend/requirements/base.txt        # production
pip install -r backend/requirements/dev.txt         # with test tools

# Copy and fill env vars
cp .env.example .env

# Run dev server (from repo root)
uvicorn backend.main:app --reload --app-dir .
# App at http://127.0.0.1:8000

# Run unit tests
python -m pytest tests/ -v --ignore=tests/test_e2e_smoke.py

# Run smoke tests (backend must be running)
python -m pytest tests/test_e2e_smoke.py -v
```

### Frontend
```bash
cd frontend
npm install
npm run dev         # Vite dev server on http://localhost:5173
npm run build       # Output to frontend/dist/
npm run type-check  # tsc --noEmit
```

## Architecture

**LLM Sensei** is a JLPT Japanese exam prep platform. Users input grammar questions, get AI explanations of why each option is right/wrong, review history, see weak concept analytics, get YouTube video recommendations, and generate new practice questions.

### Stack
| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, SQLAlchemy 2.0 async (aiosqlite), python-jose JWT |
| LLM | Pluggable: OpenRouter (default) or local HuggingFace model |
| Frontend | React 18, Vite 5, TypeScript, Tailwind CSS, React Router 6 |
| Database | SQLite (dev), swappable to PostgreSQL via `DATABASE_URL` env var |

### LLM Provider Selection
Set `LLM_PROVIDER=openrouter` (default) or `LLM_PROVIDER=local` in `.env`.
- OpenRouter: requires `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`
- Local: requires GPU + transformers install; set `LOCAL_MODEL_NAME`
The factory (`backend/services/llm/factory.py`) is cached with `lru_cache`.

### Key Request Flow
1. `POST /api/v1/quiz/solve` → SSE stream of tokens → final `<RESULT>{json}</RESULT>` block parsed by `backend/services/quiz.py:parse_result_block()`
2. Frontend `useQuizSession` hook reads the SSE stream via `api.solveStreamFetch()`
3. On result: frontend records attempt via `POST /api/v1/history/record`
4. `GET /api/v1/recommendations?concepts=...` → YouTube Data API v3
5. `POST /api/v1/generate` → LLM generates new question JSON

### Database Models
- `User`: id, email, hashed_password, created_at
- `Attempt`: user_id, question_text, options (JSON), correct_answer, llm_answer, concepts (JSON), user_marked_correct, created_at

### Frontend Pages
- `/login` — Auth (login/register), no layout wrapper
- `/practice` — QuizForm + ExplanationCard + ImageExtractor + RecommendationPanel
- `/history` — Past attempts, filterable by wrong-only
- `/stats` — Accuracy, study days, weak concepts with generate-question buttons
