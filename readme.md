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
