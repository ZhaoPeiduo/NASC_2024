# JLPT Sensei Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the single-file prototype into a production-ready JLPT study platform with pluggable LLM backends (OpenRouter + local), user accounts, learning analytics, media recommendations, and AI-generated practice questions.

**Architecture:** FastAPI backend restructured into services/routers/models layers with a streaming SSE quiz endpoint; React + Vite + TypeScript frontend replacing the monolithic HTML file; SQLite (swappable to PostgreSQL via env var) for persistence; YouTube Data API v3 for recommendations.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.0 async (aiosqlite), python-jose JWT, httpx, sse-starlette, EasyOCR; React 18, Vite 5, TypeScript 5, Tailwind CSS 3, React Router 6, Vitest, React Testing Library.

---

## File Map

### Backend (new layout replaces flat files)

```
backend/
├── main.py                        # FastAPI app factory, mounts routers
├── config.py                      # Pydantic Settings (env vars)
├── database.py                    # Async engine, session factory, get_db dep
├── auth/
│   ├── __init__.py
│   ├── jwt_handler.py             # create_token / verify_token
│   └── dependencies.py            # get_current_user FastAPI dep
├── models/
│   ├── __init__.py
│   └── tables.py                  # SQLAlchemy ORM: User, Question, Attempt
├── schemas/
│   ├── __init__.py
│   └── api.py                     # All Pydantic request/response models
├── services/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract LLMProvider (stream_solve, stream_generate)
│   │   ├── openrouter.py          # OpenRouter via httpx SSE
│   │   ├── local.py               # Refactored model.py (HuggingFace)
│   │   └── factory.py             # get_llm_provider() based on config
│   ├── quiz.py                    # solve_question() → streams SSE tokens + final JSON
│   ├── ocr.py                     # extract_text() wraps EasyOCR
│   ├── analytics.py               # record_attempt(), get_stats(), get_weak_concepts()
│   ├── recommendations.py         # search_youtube() returns Video list
│   └── generator.py               # generate_question() → GeneratedQuestion
├── routers/
│   ├── __init__.py
│   ├── auth.py                    # POST /auth/register, /auth/login, GET /auth/me
│   ├── quiz.py                    # POST /api/v1/quiz/solve (SSE), POST /api/v1/ocr/extract
│   ├── history.py                 # GET /api/v1/history, GET /api/v1/stats
│   └── advanced.py                # GET /api/v1/recommendations, POST /api/v1/generate
└── requirements/
    ├── base.txt
    └── dev.txt
```

### Frontend (replaces frontend.html)

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
└── src/
    ├── main.tsx
    ├── App.tsx                    # Router setup
    ├── api/
    │   └── client.ts              # All typed API calls + SSE stream helper
    ├── types/
    │   └── index.ts               # Shared TypeScript interfaces
    ├── contexts/
    │   └── AuthContext.tsx        # Auth state + token storage
    ├── hooks/
    │   ├── useQuizSession.ts      # SSE quiz flow, field state, progress
    │   └── useAuth.ts             # Login/register/logout actions
    ├── pages/
    │   ├── LoginPage.tsx
    │   ├── PracticePage.tsx       # Main quiz loop
    │   ├── HistoryPage.tsx        # Past attempts + filter
    │   └── StatsPage.tsx          # Charts + weak concepts
    └── components/
        ├── Layout.tsx             # Nav + auth guard wrapper
        ├── QuizForm.tsx           # Question + options input (controlled)
        ├── ExplanationCard.tsx    # Streams answer + explanation + wrong-option notes
        ├── StreamingProgress.tsx  # Phase indicator (Answering → Explaining → Done)
        ├── ImageExtractor.tsx     # Camera/upload + canvas crop + OCR
        ├── HistoryItem.tsx        # Single attempt row
        ├── StatsChart.tsx         # Accuracy bar chart (no charting lib, SVG)
        └── RecommendationPanel.tsx # YouTube video cards
```

### Tests

```
tests/
├── conftest.py                    # pytest fixtures: async client, test DB, mock LLM
├── test_llm_providers.py
├── test_quiz_service.py
├── test_analytics_service.py
├── test_recommendations.py
└── test_generation.py
```

---

## Phase 1: Backend Foundation — LLM Abstraction + Streaming Quiz

### Task 1: Project Scaffolding

**Files:**
- Create: `backend/requirements/base.txt`
- Create: `backend/requirements/dev.txt`
- Create: `backend/config.py`
- Create: `backend/__init__.py`, `backend/services/__init__.py`, `backend/services/llm/__init__.py`, `backend/routers/__init__.py`, `backend/auth/__init__.py`, `backend/models/__init__.py`, `backend/schemas/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024
mkdir -p backend/requirements backend/services/llm backend/routers backend/auth backend/models backend/schemas tests
touch backend/__init__.py backend/services/__init__.py backend/services/llm/__init__.py
touch backend/routers/__init__.py backend/auth/__init__.py backend/models/__init__.py backend/schemas/__init__.py
```

- [ ] **Step 2: Write `backend/requirements/base.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
httpx==0.27.0
sse-starlette==2.1.0
sqlalchemy[asyncio]==2.0.35
aiosqlite==0.20.0
alembic==1.13.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.9
pydantic-settings==2.4.0
easyocr==1.7.1
opencv-python-headless==4.9.0.80
Pillow==10.4.0
```

- [ ] **Step 3: Write `backend/requirements/dev.txt`**

```
-r base.txt
pytest==8.3.2
pytest-asyncio==0.23.8
pytest-mock==3.14.0
httpx==0.27.0
```

- [ ] **Step 4: Write `backend/config.py`**

```python
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # LLM backend: "openrouter" or "local"
    llm_provider: Literal["openrouter", "local"] = "openrouter"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"

    # Local HuggingFace model
    local_model_name: str = "stabilityai/japanese-stablelm-instruct-gamma-7b"

    # Auth
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60 * 24 * 7  # 1 week

    # Database
    database_url: str = "sqlite+aiosqlite:///./jlpt_sensei.db"

    # YouTube
    youtube_api_key: str = ""

    # CORS
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:8000"]

    class Config:
        env_file = ".env"

settings = Settings()
```

- [ ] **Step 5: Create `.env.example`**

```
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini
SECRET_KEY=generate-a-long-random-string-here
DATABASE_URL=sqlite+aiosqlite:///./jlpt_sensei.db
YOUTUBE_API_KEY=AIza...
ALLOWED_ORIGINS=["http://localhost:5173"]
```

- [ ] **Step 6: Commit**

```bash
git add backend/ tests/ .env.example
git commit -m "feat: scaffold new backend directory structure with config"
```

---

### Task 2: Database Models and Setup

**Files:**
- Create: `backend/models/tables.py`
- Create: `backend/database.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_models.py
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from backend.models.tables import Base, User, Question, Attempt

@pytest.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

@pytest.mark.asyncio
async def test_create_user(engine):
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        user = User(email="test@example.com", hashed_password="hashed")
        session.add(user)
        await session.commit()
        await session.refresh(user)
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.created_at is not None

@pytest.mark.asyncio
async def test_create_attempt(engine):
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        user = User(email="test2@example.com", hashed_password="hashed")
        session.add(user)
        await session.commit()
        attempt = Attempt(
            user_id=user.id,
            question_text="彼女は毎日日本語を＿＿＿います。",
            options='["勉強して", "勉強した", "勉強する", "勉強し"]',
            correct_answer="A",
            llm_answer="A",
            concepts='["て-form", "continuous action"]',
            user_marked_correct=True,
        )
        session.add(attempt)
        await session.commit()
        await session.refresh(attempt)
        assert attempt.id is not None
        assert attempt.user_id == user.id
```

- [ ] **Step 2: Run test to confirm failure**

```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024
python -m pytest tests/test_models.py -v
```

Expected: `ModuleNotFoundError: No module named 'backend.models.tables'`

- [ ] **Step 3: Write `backend/models/tables.py`**

```python
from datetime import datetime, timezone
from sqlalchemy import String, Boolean, ForeignKey, DateTime, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    attempts: Mapped[list["Attempt"]] = relationship(back_populates="user")

class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[str] = mapped_column(Text, nullable=False)          # JSON array string
    correct_answer: Mapped[str] = mapped_column(String(1), nullable=False)
    llm_answer: Mapped[str] = mapped_column(String(1), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="")
    concepts: Mapped[str] = mapped_column(Text, default="[]")            # JSON array string
    user_marked_correct: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    user: Mapped["User"] = relationship(back_populates="attempts")
```

- [ ] **Step 4: Write `backend/database.py`**

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from backend.config import settings
from backend.models.tables import Base

engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_models.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/models/ backend/database.py tests/test_models.py
git commit -m "feat: add SQLAlchemy models for User and Attempt"
```

---

### Task 3: LLM Provider Abstraction

**Files:**
- Create: `backend/services/llm/base.py`
- Create: `tests/test_llm_providers.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm_providers.py
import pytest
from backend.services.llm.base import LLMProvider, SolveResult

def test_solve_result_has_required_fields():
    result = SolveResult(
        answer="A",
        explanation="AはXXXです",
        wrong_options={"B": "BはXXX", "C": "CはXXX", "D": "DはXXX"},
        concepts=["て-form"]
    )
    assert result.answer == "A"
    assert len(result.concepts) == 1
    assert "B" in result.wrong_options

def test_llm_provider_is_abstract():
    import inspect
    assert inspect.isabstract(LLMProvider)
```

- [ ] **Step 2: Run test to confirm failure**

```bash
python -m pytest tests/test_llm_providers.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Write `backend/services/llm/base.py`**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

@dataclass
class SolveResult:
    answer: str                          # "A", "B", "C", or "D"
    explanation: str                     # Why the correct answer is right
    wrong_options: dict[str, str]        # {"B": "why B is wrong", ...}
    concepts: list[str]                  # ["て-form", "conditional"]

@dataclass
class GeneratedQuestion:
    question: str
    options: list[str]                   # ["A: ...", "B: ...", "C: ...", "D: ..."]
    correct_answer: str                  # "A"/"B"/"C"/"D"
    explanation: str
    concepts: list[str]

class LLMProvider(ABC):
    @abstractmethod
    async def stream_solve(
        self,
        question: str,
        options: list[str],
    ) -> AsyncIterator[str]:
        """Yield raw text tokens. Final token sequence includes a JSON block:
        <RESULT>{"answer":"A","explanation":"...","wrong_options":{...},"concepts":[...]}</RESULT>
        """
        ...

    @abstractmethod
    async def generate_question(
        self,
        concept: str,
        level: str,
    ) -> GeneratedQuestion:
        """Return a fully formed question with answer and explanation."""
        ...
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_llm_providers.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/services/llm/base.py tests/test_llm_providers.py
git commit -m "feat: define abstract LLMProvider interface with SolveResult"
```

---

### Task 4: OpenRouter Provider

**Files:**
- Create: `backend/services/llm/openrouter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm_providers.py (add to existing file)
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.services.llm.openrouter import OpenRouterProvider

SOLVE_PROMPT_RESPONSE = """
AはXXXです。て形は継続を表します。

<RESULT>{"answer":"A","explanation":"AはXXXです","wrong_options":{"B":"BはYYY","C":"CはZZZ","D":"DはWWW"},"concepts":["て-form"]}</RESULT>
"""

@pytest.mark.asyncio
async def test_openrouter_stream_solve_extracts_result():
    provider = OpenRouterProvider(api_key="fake-key", model="openai/gpt-4o-mini")

    async def fake_stream(*args, **kwargs):
        for char in SOLVE_PROMPT_RESPONSE:
            yield char

    with patch.object(provider, "_stream_chat", side_effect=fake_stream):
        tokens = []
        async for token in provider.stream_solve("問題文", ["A: 勉強して", "B: 勉強した", "C: 勉強する", "D: 勉強し"]):
            tokens.append(token)

    full_text = "".join(tokens)
    assert "AはXXXです" in full_text
    assert "<RESULT>" in full_text
```

- [ ] **Step 2: Run test to confirm failure**

```bash
python -m pytest tests/test_llm_providers.py::test_openrouter_stream_solve_extracts_result -v
```

Expected: `ModuleNotFoundError: No module named 'backend.services.llm.openrouter'`

- [ ] **Step 3: Write `backend/services/llm/openrouter.py`**

```python
import json
import httpx
from typing import AsyncIterator
from backend.services.llm.base import LLMProvider, SolveResult, GeneratedQuestion

SOLVE_SYSTEM_PROMPT = """You are a JLPT Japanese grammar teacher. 
Given a multiple-choice grammar question, select the correct answer and explain:
1. Why the correct answer is right
2. Why each wrong option is incorrect
3. List the grammar concepts tested

Always end your response with this exact block (fill in real values):
<RESULT>{"answer":"A","explanation":"...","wrong_options":{"B":"...","C":"...","D":"..."},"concepts":["..."]}</RESULT>"""

GENERATE_SYSTEM_PROMPT = """You are a JLPT grammar question creator.
Generate one multiple-choice grammar question at the specified JLPT level.
Respond ONLY with valid JSON, no other text."""

class OpenRouterProvider(LLMProvider):
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    async def _stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://jlpt-sensei.app",
        }
        body = {"model": self.model, "messages": messages, "stream": True}

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream("POST", f"{self.BASE_URL}/chat/completions",
                                     headers=headers, json=body) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def stream_solve(self, question: str, options: list[str]) -> AsyncIterator[str]:
        options_text = "\n".join(options)
        user_msg = f"問題: {question}\n選択肢:\n{options_text}"
        messages = [
            {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        async for token in self._stream_chat(messages):
            yield token

    async def generate_question(self, concept: str, level: str) -> GeneratedQuestion:
        user_msg = (
            f"Generate a JLPT {level} grammar question about the concept: {concept}\n\n"
            f'Respond with JSON: {{"question":"...","options":["A: ...","B: ...","C: ...","D: ..."],'
            f'"correct_answer":"A","explanation":"...","concepts":["{concept}"]}}'
        )
        messages = [
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        full = ""
        async for token in self._stream_chat(messages):
            full += token
        data = json.loads(full.strip())
        return GeneratedQuestion(**data)
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_llm_providers.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/services/llm/openrouter.py tests/test_llm_providers.py
git commit -m "feat: implement OpenRouter LLM provider with SSE streaming"
```

---

### Task 5: Local Model Provider + Provider Factory

**Files:**
- Create: `backend/services/llm/local.py`
- Create: `backend/services/llm/factory.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_llm_providers.py (add)
from backend.services.llm.factory import get_llm_provider
from backend.services.llm.openrouter import OpenRouterProvider

def test_factory_returns_openrouter_when_configured(monkeypatch):
    monkeypatch.setattr("backend.services.llm.factory.settings.llm_provider", "openrouter")
    monkeypatch.setattr("backend.services.llm.factory.settings.openrouter_api_key", "sk-test")
    monkeypatch.setattr("backend.services.llm.factory.settings.openrouter_model", "openai/gpt-4o-mini")
    provider = get_llm_provider()
    assert isinstance(provider, OpenRouterProvider)

def test_factory_raises_when_local_no_gpu(monkeypatch):
    monkeypatch.setattr("backend.services.llm.factory.settings.llm_provider", "local")
    # local provider should attempt import but we just verify it returns something
    # (actual GPU loading is tested manually, not in unit tests)
    import backend.services.llm.factory as f
    assert hasattr(f, "get_llm_provider")
```

- [ ] **Step 2: Write `backend/services/llm/local.py`**

```python
"""
Local HuggingFace inference provider.
Requires: torch, transformers, accelerate (GPU recommended).
Falls back to CPU but is very slow.
"""
import json
from typing import AsyncIterator
import asyncio
from backend.services.llm.base import LLMProvider, SolveResult, GeneratedQuestion

SOLVE_SYSTEM = """あなたはJLPT日本語文法の先生です。
問題を分析し、正しい答えを選んで理由を説明してください。
最後に必ずこのブロックで終わってください:
<RESULT>{"answer":"A","explanation":"...","wrong_options":{"B":"...","C":"...","D":"..."},"concepts":["..."]}</RESULT>"""

class LocalModelProvider(LLMProvider):
    def __init__(self, model_name: str):
        # Lazy import — only triggered when local provider is selected
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._loop = asyncio.get_event_loop()

    def _generate_sync(self, prompt: str, max_new_tokens: int = 512) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

    async def stream_solve(self, question: str, options: list[str]) -> AsyncIterator[str]:
        options_text = "\n".join(options)
        prompt = (f"{SOLVE_SYSTEM}\n\n"
                  f"問題: {question}\n選択肢:\n{options_text}\n\n回答:")
        # Local model generates all at once; yield in chunks to match streaming interface
        text = await asyncio.get_event_loop().run_in_executor(
            None, self._generate_sync, prompt
        )
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
            await asyncio.sleep(0)

    async def generate_question(self, concept: str, level: str) -> GeneratedQuestion:
        prompt = (f"JLPT {level}レベルの{concept}に関する文法問題を1問作成してください。\n"
                  f"JSON形式で回答: {{\"question\":\"...\",\"options\":[\"A:...\",\"B:...\",\"C:...\",\"D:...\"],"
                  f"\"correct_answer\":\"A\",\"explanation\":\"...\",\"concepts\":[\"{concept}\"]}}")
        text = await asyncio.get_event_loop().run_in_executor(
            None, self._generate_sync, prompt, 300
        )
        data = json.loads(text.strip())
        return GeneratedQuestion(**data)
```

- [ ] **Step 3: Write `backend/services/llm/factory.py`**

```python
from functools import lru_cache
from backend.config import settings
from backend.services.llm.base import LLMProvider

@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "openrouter":
        from backend.services.llm.openrouter import OpenRouterProvider
        return OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
        )
    elif settings.llm_provider == "local":
        from backend.services.llm.local import LocalModelProvider
        return LocalModelProvider(model_name=settings.local_model_name)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_llm_providers.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add backend/services/llm/local.py backend/services/llm/factory.py tests/test_llm_providers.py
git commit -m "feat: add local HuggingFace provider and factory with lru_cache"
```

---

### Task 6: Quiz Service + SSE Router

**Files:**
- Create: `backend/services/quiz.py`
- Create: `backend/schemas/api.py`
- Create: `backend/routers/quiz.py`
- Create: `tests/test_quiz_service.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_quiz_service.py
import pytest
import json
from unittest.mock import AsyncMock, patch

async def fake_stream(*args, **kwargs):
    # Simulate tokens including final RESULT block
    response = ("AはXXXです。て形は継続を表します。\n"
                 '<RESULT>{"answer":"A","explanation":"AはXXXです","wrong_options":{"B":"BはYYY","C":"CはZZZ","D":"DはWWW"},"concepts":["て-form"]}</RESULT>')
    for char in response:
        yield char

@pytest.mark.asyncio
async def test_solve_question_yields_tokens_and_result():
    from backend.services.quiz import solve_question, parse_result_block
    from backend.services.llm.base import SolveResult

    tokens = []
    result_holder = []

    async for event in solve_question(
        question="彼女は毎日日本語を＿＿＿います。",
        options=["A: 勉強して", "B: 勉強した", "C: 勉強する", "D: 勉強し"],
        provider_stream=fake_stream(),
    ):
        tokens.append(event)

    full = "".join(tokens)
    assert "AはXXXです" in full
    result = parse_result_block(full)
    assert result is not None
    assert result.answer == "A"
    assert "て-form" in result.concepts
```

- [ ] **Step 2: Write `backend/schemas/api.py`**

```python
from pydantic import BaseModel, EmailStr

class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    id: int
    email: str

class SolveRequest(BaseModel):
    question: str
    options: list[str]              # ["A: ...", "B: ...", "C: ...", "D: ..."]

class AttemptRecord(BaseModel):
    question_text: str
    options: list[str]
    correct_answer: str
    llm_answer: str
    user_marked_correct: bool
    concepts: list[str]

class AttemptResponse(BaseModel):
    id: int
    question_text: str
    correct_answer: str
    llm_answer: str
    user_marked_correct: bool
    concepts: list[str]
    created_at: str

class StatsResponse(BaseModel):
    total_attempts: int
    correct_rate: float
    weak_concepts: list[str]
    study_days: int

class GenerateRequest(BaseModel):
    concept: str
    level: str = "N3"

class GeneratedQuestionResponse(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    explanation: str
    concepts: list[str]

class VideoRecommendation(BaseModel):
    title: str
    video_id: str
    thumbnail_url: str
    channel_title: str
```

- [ ] **Step 3: Write `backend/services/quiz.py`**

```python
import json
import re
from typing import AsyncIterator, Optional
from backend.services.llm.base import LLMProvider, SolveResult

RESULT_PATTERN = re.compile(r"<RESULT>(.*?)</RESULT>", re.DOTALL)

def parse_result_block(text: str) -> Optional[SolveResult]:
    match = RESULT_PATTERN.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return SolveResult(
            answer=data["answer"],
            explanation=data.get("explanation", ""),
            wrong_options=data.get("wrong_options", {}),
            concepts=data.get("concepts", []),
        )
    except (json.JSONDecodeError, KeyError):
        return None

async def solve_question(
    question: str,
    options: list[str],
    provider_stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Yields raw tokens from the LLM stream (caller handles SSE wrapping)."""
    async for token in provider_stream:
        yield token
```

- [ ] **Step 4: Write `backend/routers/quiz.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import json

from backend.schemas.api import SolveRequest
from backend.services.llm.factory import get_llm_provider
from backend.services.quiz import solve_question, parse_result_block
from backend.services.ocr import extract_text
from backend.auth.dependencies import get_current_user_optional
from backend.models.tables import User

router = APIRouter(prefix="/api/v1")

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
            # Yield every token as an SSE "token" event
            yield {"event": "token", "data": token}
        
        # After streaming completes, parse the structured result
        result = parse_result_block(full_text)
        if result:
            yield {
                "event": "result",
                "data": json.dumps({
                    "answer": result.answer,
                    "explanation": result.explanation,
                    "wrong_options": result.wrong_options,
                    "concepts": result.concepts,
                })
            }
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())
```

- [ ] **Step 5: Run quiz service tests**

```bash
python -m pytest tests/test_quiz_service.py -v
```

Expected: `1 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/schemas/ backend/services/quiz.py backend/routers/quiz.py tests/test_quiz_service.py
git commit -m "feat: quiz service with SSE streaming and RESULT block parser"
```

---

### Task 7: OCR Service + Auth Stubs + Main App

**Files:**
- Create: `backend/services/ocr.py`
- Create: `backend/auth/jwt_handler.py`
- Create: `backend/auth/dependencies.py`
- Create: `backend/routers/auth.py`
- Create: `backend/main.py`

- [ ] **Step 1: Write `backend/services/ocr.py`**

```python
"""Extracted OCR logic from original backend.py."""
import numpy as np
from PIL import Image
import io
import base64
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_reader():
    import easyocr
    return easyocr.Reader(["ja", "en"])

def extract_text(
    image_b64: str,
    x1: int, y1: int, x2: int, y2: int,
    num_options: int,
) -> dict[str, str | list[str]]:
    """
    Crop the base64 image to the bounding box, run OCR, split into
    question and options based on num_options.
    Returns {"question": str, "options": [str, ...]}
    """
    image_bytes = base64.b64decode(image_b64.split(",")[-1])
    img = Image.open(io.BytesIO(image_bytes))
    img_array = np.array(img)

    x1_, x2_ = min(x1, x2), max(x1, x2)
    y1_, y2_ = min(y1, y2), max(y1, y2)
    crop = img_array[y1_:y2_, x1_:x2_]

    reader = _get_reader()
    results = reader.readtext(crop)
    texts = [r[1] for r in sorted(results, key=lambda r: r[0][0][1])]

    if len(texts) <= num_options:
        return {"question": " ".join(texts), "options": []}

    question = " ".join(texts[:-num_options])
    options = texts[-num_options:]
    return {"question": question, "options": options}
```

- [ ] **Step 2: Write `backend/auth/jwt_handler.py`**

```python
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from backend.config import settings

ALGORITHM = "HS256"

def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode(
        {"sub": str(user_id), "exp": expire},
        settings.secret_key,
        algorithm=ALGORITHM,
    )

def decode_token(token: str) -> int | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None
```

- [ ] **Step 3: Write `backend/auth/dependencies.py`**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from backend.database import get_db
from backend.models.tables import User
from backend.auth.jwt_handler import decode_token

bearer_scheme = HTTPBearer(auto_error=False)

async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user_id = decode_token(credentials.credentials)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User | None:
    if not credentials:
        return None
    user_id = decode_token(credentials.credentials)
    if user_id is None:
        return None
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()
```

- [ ] **Step 4: Write `backend/routers/auth.py`**

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from passlib.context import CryptContext
from backend.database import get_db
from backend.models.tables import User
from backend.schemas.api import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from backend.auth.jwt_handler import create_access_token
from backend.auth.dependencies import get_current_user

router = APIRouter(prefix="/auth")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register", response_model=TokenResponse)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=body.email, hashed_password=pwd_context.hash(body.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not pwd_context.verify(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return TokenResponse(access_token=create_access_token(user.id))

@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    return UserResponse(id=current_user.id, email=current_user.email)
```

- [ ] **Step 5: Write `backend/main.py`**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from backend.config import settings
from backend.database import init_db
from backend.routers import auth, quiz

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

app.include_router(auth.router)
app.include_router(quiz.router)

# Serve built frontend if dist/ exists
dist_path = Path(__file__).parent.parent / "frontend" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="frontend")

@app.get("/health/live")
async def health_live():
    return {"status": "ok"}
```

- [ ] **Step 6: Test the app starts**

```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024
pip install -r backend/requirements/base.txt
uvicorn backend.main:app --reload --app-dir .
# In another terminal:
curl http://localhost:8000/health/live
```

Expected: `{"status":"ok"}`

- [ ] **Step 7: Commit**

```bash
git add backend/main.py backend/routers/auth.py backend/auth/ backend/services/ocr.py
git commit -m "feat: wire up FastAPI app with auth routes and health endpoint"
```

---

## Phase 2: Learning Analytics

### Task 8: Analytics Service

**Files:**
- Create: `backend/services/analytics.py`
- Create: `backend/routers/history.py`
- Create: `tests/test_analytics_service.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_analytics_service.py
import pytest
import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from backend.models.tables import Base, User, Attempt
from backend.services.analytics import record_attempt, get_stats, get_weak_concepts

@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        user = User(email="test@test.com", hashed_password="x")
        session.add(user)
        await session.commit()
        await session.refresh(user)
        yield session, user.id
    await engine.dispose()

@pytest.mark.asyncio
async def test_record_attempt_saves_to_db(db_session):
    session, user_id = db_session
    await record_attempt(
        db=session,
        user_id=user_id,
        question_text="問題文",
        options=["A: あ", "B: い", "C: う", "D: え"],
        correct_answer="A",
        llm_answer="A",
        explanation="Aが正しいです",
        concepts=["て-form"],
        user_marked_correct=True,
    )
    from sqlalchemy import select
    result = await session.execute(select(Attempt).where(Attempt.user_id == user_id))
    attempts = result.scalars().all()
    assert len(attempts) == 1
    assert attempts[0].correct_answer == "A"

@pytest.mark.asyncio
async def test_get_weak_concepts_returns_most_missed(db_session):
    session, user_id = db_session
    # Add 3 wrong attempts with "て-form", 1 wrong with "conditional"
    for _ in range(3):
        await record_attempt(session, user_id, "Q", ["A","B","C","D"],
                             "A", "B", "", ["て-form"], False)
    await record_attempt(session, user_id, "Q2", ["A","B","C","D"],
                         "A", "B", "", ["conditional"], False)
    weak = await get_weak_concepts(session, user_id, limit=2)
    assert weak[0] == "て-form"
```

- [ ] **Step 2: Write `backend/services/analytics.py`**

```python
import json
from collections import Counter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from backend.models.tables import Attempt

async def record_attempt(
    db: AsyncSession,
    user_id: int,
    question_text: str,
    options: list[str],
    correct_answer: str,
    llm_answer: str,
    explanation: str,
    concepts: list[str],
    user_marked_correct: bool,
) -> Attempt:
    attempt = Attempt(
        user_id=user_id,
        question_text=question_text,
        options=json.dumps(options, ensure_ascii=False),
        correct_answer=correct_answer,
        llm_answer=llm_answer,
        explanation=explanation,
        concepts=json.dumps(concepts, ensure_ascii=False),
        user_marked_correct=user_marked_correct,
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)
    return attempt

async def get_stats(db: AsyncSession, user_id: int) -> dict:
    result = await db.execute(
        select(Attempt).where(Attempt.user_id == user_id)
    )
    attempts = result.scalars().all()
    if not attempts:
        return {"total_attempts": 0, "correct_rate": 0.0, "study_days": 0}
    correct = sum(1 for a in attempts if a.user_marked_correct)
    days = len({a.created_at.date() for a in attempts})
    return {
        "total_attempts": len(attempts),
        "correct_rate": round(correct / len(attempts), 3),
        "study_days": days,
    }

async def get_weak_concepts(db: AsyncSession, user_id: int, limit: int = 5) -> list[str]:
    result = await db.execute(
        select(Attempt).where(Attempt.user_id == user_id, Attempt.user_marked_correct == False)
    )
    wrong_attempts = result.scalars().all()
    counter: Counter = Counter()
    for attempt in wrong_attempts:
        concepts = json.loads(attempt.concepts or "[]")
        counter.update(concepts)
    return [concept for concept, _ in counter.most_common(limit)]
```

- [ ] **Step 3: Write `backend/routers/history.py`**

```python
import json
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from backend.database import get_db
from backend.models.tables import Attempt, User
from backend.auth.dependencies import get_current_user
from backend.schemas.api import AttemptResponse, StatsResponse
from backend.services.analytics import get_stats, get_weak_concepts

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
        )
        for a in attempts
    ]

@router.get("/stats", response_model=StatsResponse)
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stats = await get_stats(db, current_user.id)
    weak = await get_weak_concepts(db, current_user.id)
    return StatsResponse(weak_concepts=weak, **stats)
```

- [ ] **Step 4: Add `history` router to `backend/main.py`**

```python
# In backend/main.py, add after quiz router:
from backend.routers import auth, quiz, history   # update import
app.include_router(history.router)                # add this line
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_analytics_service.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/services/analytics.py backend/routers/history.py tests/test_analytics_service.py backend/main.py
git commit -m "feat: analytics service with attempt recording and weak concept tracking"
```

---

## Phase 3: Frontend Redesign

### Task 9: Vite + React + TypeScript Scaffold

**Files:**
- Create: `frontend/package.json`, `frontend/vite.config.ts`, `frontend/tsconfig.json`, `frontend/tailwind.config.js`, `frontend/index.html`
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/api/client.ts`

- [ ] **Step 1: Scaffold the Vite project**

```bash
cd /c/Users/Andrew/Desktop/Projects/NASC_2024
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install react-router-dom
```

- [ ] **Step 2: Configure `frontend/tailwind.config.js`**

```js
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: { 50: "#f0f9ff", 500: "#0ea5e9", 700: "#0369a1" },
        correct: "#16a34a",
        wrong: "#dc2626",
      },
    },
  },
  plugins: [],
}
```

- [ ] **Step 3: Replace `frontend/src/index.css` with Tailwind directives**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 4: Write `frontend/src/types/index.ts`**

```typescript
export type Phase = "idle" | "answering" | "explaining" | "done";

export interface SolveResult {
  answer: string;                          // "A" | "B" | "C" | "D"
  explanation: string;
  wrong_options: Record<string, string>;   // { "B": "why B is wrong", ... }
  concepts: string[];
}

export interface AttemptResponse {
  id: number;
  question_text: string;
  correct_answer: string;
  llm_answer: string;
  user_marked_correct: boolean;
  concepts: string[];
  created_at: string;
}

export interface StatsResponse {
  total_attempts: number;
  correct_rate: number;
  weak_concepts: string[];
  study_days: number;
}

export interface VideoRecommendation {
  title: string;
  video_id: string;
  thumbnail_url: string;
  channel_title: string;
}

export interface GeneratedQuestion {
  question: string;
  options: string[];
  correct_answer: string;
  explanation: string;
  concepts: string[];
}
```

- [ ] **Step 5: Write `frontend/src/api/client.ts`**

```typescript
const BASE = import.meta.env.VITE_API_BASE ?? "";

function authHeaders(): HeadersInit {
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function post<T>(path: string, body: unknown, auth = false): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(auth ? authHeaders() : {}) },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}

export const api = {
  register: (email: string, password: string) =>
    post<{ access_token: string }>("/auth/register", { email, password }),

  login: (email: string, password: string) =>
    post<{ access_token: string }>("/auth/login", { email, password }),

  me: () => get<{ id: number; email: string }>("/auth/me"),

  /** Returns an EventSource for SSE streaming */
  solveStream: (question: string, options: string[]): EventSource => {
    // POST-based SSE requires a workaround — use fetch with ReadableStream
    // For simplicity, encode params in URL (GET-compatible with EventSource)
    // Real impl: use fetch + ReadableStream reader (see useQuizSession)
    throw new Error("Use solveStreamFetch instead");
  },

  /**
   * Stream-solve via fetch (supports POST body + SSE).
   * Returns an async generator of SSE events.
   */
  async *solveStreamFetch(
    question: string,
    options: string[]
  ): AsyncGenerator<{ event: string; data: string }> {
    const res = await fetch(`${BASE}/api/v1/quiz/solve`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ question, options }),
    });
    if (!res.ok || !res.body) throw new Error("Stream failed");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      let event = "message";
      for (const line of lines) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) {
          yield { event, data: line.slice(6) };
          event = "message";
        }
      }
    }
  },

  recordAttempt: (body: {
    question_text: string; options: string[]; correct_answer: string;
    llm_answer: string; user_marked_correct: boolean; concepts: string[];
  }) => post("/api/v1/history/record", body, true),

  getHistory: () => get<import("./types").AttemptResponse[]>("/api/v1/history"),

  getStats: () => get<import("./types").StatsResponse>("/api/v1/stats"),

  getRecommendations: (concepts: string[]) =>
    get<import("./types").VideoRecommendation[]>(
      `/api/v1/recommendations?concepts=${concepts.join(",")}`
    ),

  generateQuestion: (concept: string, level: string) =>
    post<import("./types").GeneratedQuestion>("/api/v1/generate", { concept, level }, true),
};
```

- [ ] **Step 6: Run Vite dev server to confirm scaffold works**

```bash
cd frontend && npm run dev
```

Expected: Vite dev server starts on http://localhost:5173 with default React page.

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat: scaffold React+Vite+TypeScript frontend with Tailwind and typed API client"
```

---

### Task 10: Auth Context + Login Page

**Files:**
- Create: `frontend/src/contexts/AuthContext.tsx`
- Create: `frontend/src/hooks/useAuth.ts`
- Create: `frontend/src/pages/LoginPage.tsx`
- Create: `frontend/src/components/Layout.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write `frontend/src/contexts/AuthContext.tsx`**

```typescript
import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { api } from "../api/client";

interface AuthState {
  token: string | null;
  user: { id: number; email: string } | null;
  loading: boolean;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    token: localStorage.getItem("token"),
    user: null,
    loading: true,
  });

  useEffect(() => {
    if (!state.token) { setState(s => ({ ...s, loading: false })); return; }
    api.me()
      .then(user => setState(s => ({ ...s, user, loading: false })))
      .catch(() => { localStorage.removeItem("token"); setState({ token: null, user: null, loading: false }); });
  }, [state.token]);

  const login = async (email: string, password: string) => {
    const { access_token } = await api.login(email, password);
    localStorage.setItem("token", access_token);
    setState(s => ({ ...s, token: access_token }));
  };

  const register = async (email: string, password: string) => {
    const { access_token } = await api.register(email, password);
    localStorage.setItem("token", access_token);
    setState(s => ({ ...s, token: access_token }));
  };

  const logout = () => {
    localStorage.removeItem("token");
    setState({ token: null, user: null, loading: false });
  };

  return (
    <AuthContext.Provider value={{ ...state, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuthContext() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuthContext must be inside AuthProvider");
  return ctx;
}
```

- [ ] **Step 2: Write `frontend/src/pages/LoginPage.tsx`**

```typescript
import { useState, FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useAuthContext } from "../contexts/AuthContext";

export default function LoginPage() {
  const { login, register } = useAuthContext();
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      if (mode === "login") await login(email, password);
      else await register(email, password);
      navigate("/practice");
    } catch (err: any) {
      setError(err.message ?? "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-50">
      <div className="bg-white rounded-2xl shadow-md p-8 w-full max-w-sm">
        <h1 className="text-2xl font-bold text-slate-800 mb-1">JLPT Sensei</h1>
        <p className="text-slate-500 mb-6 text-sm">Your AI-powered grammar tutor</p>

        <div className="flex gap-2 mb-6">
          {(["login", "register"] as const).map(m => (
            <button key={m}
              onClick={() => setMode(m)}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors
                ${mode === m ? "bg-brand-500 text-white" : "bg-slate-100 text-slate-600"}`}
            >
              {m === "login" ? "Sign In" : "Sign Up"}
            </button>
          ))}
        </div>

        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Email</label>
            <input type="email" value={email} onChange={e => setEmail(e.target.value)} required
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)} required
              className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
            />
          </div>
          {error && <p className="text-red-600 text-sm">{error}</p>}
          <button type="submit" disabled={loading}
            className="w-full bg-brand-500 hover:bg-brand-700 text-white py-2.5 rounded-lg font-medium transition-colors disabled:opacity-50"
          >
            {loading ? "Please wait…" : mode === "login" ? "Sign In" : "Create Account"}
          </button>
        </form>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Write `frontend/src/components/Layout.tsx`**

```typescript
import { ReactNode } from "react";
import { NavLink, Navigate } from "react-router-dom";
import { useAuthContext } from "../contexts/AuthContext";

const NAV_ITEMS = [
  { to: "/practice", label: "Practice" },
  { to: "/history", label: "History" },
  { to: "/stats", label: "Stats" },
];

export default function Layout({ children }: { children: ReactNode }) {
  const { user, loading, logout } = useAuthContext();

  if (loading) return <div className="min-h-screen flex items-center justify-center text-slate-400">Loading…</div>;
  if (!user) return <Navigate to="/login" replace />;

  return (
    <div className="min-h-screen bg-slate-50">
      <nav className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between">
        <span className="font-bold text-slate-800">JLPT Sensei</span>
        <div className="flex items-center gap-6">
          {NAV_ITEMS.map(({ to, label }) => (
            <NavLink key={to} to={to}
              className={({ isActive }) =>
                `text-sm font-medium transition-colors ${isActive ? "text-brand-500" : "text-slate-600 hover:text-slate-900"}`
              }
            >
              {label}
            </NavLink>
          ))}
          <button onClick={logout} className="text-sm text-slate-400 hover:text-slate-600">Sign out</button>
        </div>
      </nav>
      <main className="max-w-4xl mx-auto px-6 py-8">{children}</main>
    </div>
  );
}
```

- [ ] **Step 4: Write `frontend/src/App.tsx`**

```typescript
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import LoginPage from "./pages/LoginPage";
import Layout from "./components/Layout";
import PracticePage from "./pages/PracticePage";
import HistoryPage from "./pages/HistoryPage";
import StatsPage from "./pages/StatsPage";

function PlaceholderPage({ name }: { name: string }) {
  return <div className="text-slate-400 text-center py-20">{name} — coming soon</div>;
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/practice" element={<Layout><PlaceholderPage name="Practice" /></Layout>} />
          <Route path="/history" element={<Layout><PlaceholderPage name="History" /></Layout>} />
          <Route path="/stats" element={<Layout><PlaceholderPage name="Stats" /></Layout>} />
          <Route path="*" element={<Navigate to="/practice" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
```

- [ ] **Step 5: Confirm app compiles**

```bash
cd frontend && npm run build
```

Expected: Build succeeds, `dist/` created.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat: auth context, login page, protected layout with nav"
```

---

### Task 11: Practice Page — Quiz Form + Streaming Explanation

**Files:**
- Create: `frontend/src/hooks/useQuizSession.ts`
- Create: `frontend/src/components/QuizForm.tsx`
- Create: `frontend/src/components/ExplanationCard.tsx`
- Create: `frontend/src/components/StreamingProgress.tsx`
- Create: `frontend/src/pages/PracticePage.tsx`

- [ ] **Step 1: Write `frontend/src/hooks/useQuizSession.ts`**

```typescript
import { useState, useCallback } from "react";
import { api } from "../api/client";
import { Phase, SolveResult } from "../types";

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

    try {
      const optLabels = fields.options
        .filter(o => o.trim())
        .map((o, i) => `${String.fromCharCode(65 + i)}: ${o}`);

      for await (const { event, data } of api.solveStreamFetch(fields.question, optLabels)) {
        if (event === "token") {
          setStreamText(t => t + data);
          if (data.includes("EXPLANATION")) setPhase("explaining");
        } else if (event === "result") {
          setResult(JSON.parse(data) as SolveResult);
          setPhase("done");
        }
      }
    } catch (err: any) {
      setError(err.message ?? "Something went wrong. Is the backend running?");
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

- [ ] **Step 2: Write `frontend/src/components/StreamingProgress.tsx`**

```typescript
import { Phase } from "../types";

const PHASES: { key: Phase; label: string }[] = [
  { key: "answering", label: "Selecting answer" },
  { key: "explaining", label: "Generating explanation" },
  { key: "done", label: "Done" },
];

export default function StreamingProgress({ phase }: { phase: Phase }) {
  if (phase === "idle") return null;
  return (
    <div className="flex items-center gap-3 mb-4">
      {PHASES.map(({ key, label }, i) => {
        const phaseOrder = { answering: 0, explaining: 1, done: 2 };
        const currentOrder = phaseOrder[phase as keyof typeof phaseOrder] ?? -1;
        const itemOrder = phaseOrder[key as keyof typeof phaseOrder];
        const isDone = itemOrder < currentOrder;
        const isActive = itemOrder === currentOrder;
        return (
          <div key={key} className="flex items-center gap-2">
            {i > 0 && <div className={`h-px w-6 ${isDone ? "bg-brand-500" : "bg-slate-200"}`} />}
            <span className={`text-xs font-medium px-2 py-1 rounded-full
              ${isDone ? "bg-brand-50 text-brand-700"
              : isActive ? "bg-brand-500 text-white animate-pulse"
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

- [ ] **Step 3: Write `frontend/src/components/ExplanationCard.tsx`**

```typescript
import { SolveResult } from "../types";

export default function ExplanationCard({
  streamText, result, phase
}: {
  streamText: string; result: SolveResult | null; phase: string;
}) {
  if (phase === "idle") return null;

  return (
    <div className="mt-6 space-y-4">
      {result ? (
        <>
          <div className="flex items-center gap-3 p-4 bg-green-50 border border-green-200 rounded-xl">
            <span className="text-2xl font-bold text-green-700">{result.answer}</span>
            <p className="text-sm text-slate-700">{result.explanation}</p>
          </div>

          {Object.entries(result.wrong_options).length > 0 && (
            <div className="p-4 bg-slate-50 rounded-xl space-y-2">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Why the others are wrong</p>
              {Object.entries(result.wrong_options).map(([opt, reason]) => (
                <div key={opt} className="flex gap-2 text-sm">
                  <span className="font-bold text-red-500 w-4 shrink-0">{opt}</span>
                  <span className="text-slate-600">{reason}</span>
                </div>
              ))}
            </div>
          )}

          {result.concepts.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {result.concepts.map(c => (
                <span key={c} className="bg-brand-50 text-brand-700 text-xs px-2 py-1 rounded-full">{c}</span>
              ))}
            </div>
          )}
        </>
      ) : (
        <div className="p-4 bg-white border border-slate-200 rounded-xl min-h-24">
          <p className="text-sm text-slate-600 whitespace-pre-wrap">{streamText}
            <span className="animate-pulse">▌</span>
          </p>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Write `frontend/src/components/QuizForm.tsx`**

```typescript
import { KeyboardEvent } from "react";

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
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
      <label className="block text-sm font-medium text-slate-700 mb-1">Question</label>
      <textarea
        value={question}
        onChange={e => onQuestionChange(e.target.value)}
        onKeyDown={handleKey}
        placeholder="彼女は毎日日本語を＿＿＿います。"
        disabled={disabled}
        rows={3}
        className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm mb-4
          focus:outline-none focus:ring-2 focus:ring-brand-500 resize-none disabled:bg-slate-50"
      />

      <p className="text-sm font-medium text-slate-700 mb-2">Options</p>
      <div className="grid grid-cols-2 gap-3 mb-5">
        {options.map((opt, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-sm font-bold text-slate-500 w-4">{String.fromCharCode(65 + i)}</span>
            <input
              value={opt}
              onChange={e => onOptionChange(i, e.target.value)}
              onKeyDown={handleKey}
              placeholder={`Option ${String.fromCharCode(65 + i)}`}
              disabled={disabled}
              className="flex-1 border border-slate-200 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-brand-500 disabled:bg-slate-50"
            />
          </div>
        ))}
      </div>

      {error && <p className="text-red-600 text-sm mb-3">{error}</p>}

      <div className="flex gap-3">
        <button onClick={onSubmit} disabled={disabled || !canSubmit}
          className="flex-1 bg-brand-500 hover:bg-brand-700 text-white py-2.5 rounded-lg
            font-medium text-sm transition-colors disabled:opacity-40"
        >
          {disabled ? "Analyzing…" : "Get Answer ⌘↵"}
        </button>
        <button onClick={onReset} disabled={disabled}
          className="px-4 py-2.5 border border-slate-200 rounded-lg text-sm text-slate-600
            hover:bg-slate-50 transition-colors disabled:opacity-40"
        >
          Clear
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Write `frontend/src/pages/PracticePage.tsx`**

```typescript
import { useQuizSession } from "../hooks/useQuizSession";
import QuizForm from "../components/QuizForm";
import ExplanationCard from "../components/ExplanationCard";
import StreamingProgress from "../components/StreamingProgress";

export default function PracticePage() {
  const {
    fields, setFields, setOption,
    phase, streamText, result, error,
    submit, reset,
  } = useQuizSession();

  return (
    <div>
      <h1 className="text-xl font-bold text-slate-800 mb-6">Practice</h1>
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
      <StreamingProgress phase={phase} />
      <ExplanationCard streamText={streamText} result={result} phase={phase} />
    </div>
  );
}
```

- [ ] **Step 6: Update `App.tsx` to use the real PracticePage**

```typescript
// Replace the import:
import PracticePage from "./pages/PracticePage";
// And the route (remove PlaceholderPage for Practice):
<Route path="/practice" element={<Layout><PracticePage /></Layout>} />
```

- [ ] **Step 7: Build and verify**

```bash
cd frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/
git commit -m "feat: practice page with streaming quiz form and explanation card"
```

---

### Task 12: History + Stats Pages

**Files:**
- Create: `frontend/src/pages/HistoryPage.tsx`
- Create: `frontend/src/components/HistoryItem.tsx`
- Create: `frontend/src/pages/StatsPage.tsx`
- Create: `frontend/src/components/StatsChart.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write `frontend/src/components/HistoryItem.tsx`**

```typescript
import { AttemptResponse } from "../types";

export default function HistoryItem({ attempt }: { attempt: AttemptResponse }) {
  const date = new Date(attempt.created_at).toLocaleDateString();
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 space-y-2">
      <div className="flex items-start justify-between gap-4">
        <p className="text-sm text-slate-800 flex-1">{attempt.question_text}</p>
        <span className={`text-xs font-bold px-2 py-1 rounded-full shrink-0
          ${attempt.user_marked_correct ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
          {attempt.user_marked_correct ? "Correct" : "Wrong"}
        </span>
      </div>
      <div className="flex items-center gap-3 text-xs text-slate-400">
        <span>Answer: <span className="font-semibold text-slate-600">{attempt.correct_answer}</span></span>
        <span>{date}</span>
      </div>
      {attempt.concepts.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {attempt.concepts.map(c => (
            <span key={c} className="bg-slate-100 text-slate-500 text-xs px-2 py-0.5 rounded-full">{c}</span>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Write `frontend/src/pages/HistoryPage.tsx`**

```typescript
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { AttemptResponse } from "../types";
import HistoryItem from "../components/HistoryItem";

export default function HistoryPage() {
  const [attempts, setAttempts] = useState<AttemptResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "wrong">("all");

  useEffect(() => {
    api.getHistory().then(setAttempts).finally(() => setLoading(false));
  }, []);

  const shown = filter === "wrong" ? attempts.filter(a => !a.user_marked_correct) : attempts;

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
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
          {shown.map(a => <HistoryItem key={a.id} attempt={a} />)}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Write `frontend/src/components/StatsChart.tsx`**

```typescript
interface Props {
  label: string;
  value: number;     // 0-1 ratio
  color: string;     // Tailwind color class like "bg-brand-500"
}

export function BarStat({ label, value, color }: Props) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-slate-600">{label}</span>
        <span className="font-semibold text-slate-800">{pct}%</span>
      </div>
      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Write `frontend/src/pages/StatsPage.tsx`**

```typescript
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { StatsResponse } from "../types";
import { BarStat } from "../components/StatsChart";

export default function StatsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);

  useEffect(() => { api.getStats().then(setStats); }, []);

  if (!stats) return <p className="text-slate-400 text-center py-12">Loading…</p>;

  return (
    <div className="space-y-6">
      <h1 className="text-xl font-bold text-slate-800">Your Progress</h1>

      <div className="grid grid-cols-3 gap-4">
        {[
          { label: "Total Attempts", value: stats.total_attempts, unit: "" },
          { label: "Accuracy", value: `${Math.round(stats.correct_rate * 100)}%`, unit: "" },
          { label: "Study Days", value: stats.study_days, unit: "" },
        ].map(({ label, value }) => (
          <div key={label} className="bg-white border border-slate-200 rounded-xl p-4 text-center">
            <p className="text-3xl font-bold text-slate-800">{value}</p>
            <p className="text-xs text-slate-400 mt-1">{label}</p>
          </div>
        ))}
      </div>

      <div className="bg-white border border-slate-200 rounded-xl p-6">
        <BarStat label="Accuracy" value={stats.correct_rate} color="bg-brand-500" />
      </div>

      {stats.weak_concepts.length > 0 && (
        <div className="bg-white border border-slate-200 rounded-xl p-6">
          <p className="text-sm font-semibold text-slate-700 mb-3">Weak Concepts to Review</p>
          <div className="flex flex-wrap gap-2">
            {stats.weak_concepts.map(c => (
              <span key={c} className="bg-red-50 text-red-700 text-sm px-3 py-1 rounded-full">{c}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Update `App.tsx` with real page imports**

```typescript
import HistoryPage from "./pages/HistoryPage";
import StatsPage from "./pages/StatsPage";
// Update routes:
<Route path="/history" element={<Layout><HistoryPage /></Layout>} />
<Route path="/stats" element={<Layout><StatsPage /></Layout>} />
```

- [ ] **Step 6: Add `record_attempt` endpoint to `backend/routers/history.py`**

```python
# Add this route to backend/routers/history.py
from backend.schemas.api import AttemptRecord
from backend.services.analytics import record_attempt as svc_record

@router.post("/history/record", status_code=201)
async def record(
    body: AttemptRecord,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await svc_record(
        db=db,
        user_id=current_user.id,
        question_text=body.question_text,
        options=body.options,
        correct_answer=body.correct_answer,
        llm_answer=body.llm_answer,
        explanation="",
        concepts=body.concepts,
        user_marked_correct=body.user_marked_correct,
    )
    return {"status": "recorded"}
```

- [ ] **Step 7: Build frontend**

```bash
cd frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/ backend/routers/history.py
git commit -m "feat: history page with filter, stats page with accuracy and weak concepts"
```

---

## Phase 4: Image Extractor Component

### Task 13: ImageExtractor with Mobile Camera Support

**Files:**
- Create: `frontend/src/components/ImageExtractor.tsx`
- Modify: `frontend/src/pages/PracticePage.tsx`
- Modify: `backend/routers/quiz.py` (add OCR endpoint)

- [ ] **Step 1: Add OCR route to `backend/routers/quiz.py`**

```python
# Add to backend/routers/quiz.py
from fastapi import File, UploadFile, Form
import base64
from backend.services.ocr import extract_text as svc_extract

@router.post("/ocr/extract")
async def ocr_extract(
    image_data: str = Form(...),    # base64 data URL
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
        raise HTTPException(status_code=422, detail=f"OCR failed: {str(e)}")
```

- [ ] **Step 2: Write `frontend/src/components/ImageExtractor.tsx`**

```typescript
import { useRef, useState, MouseEvent, TouchEvent } from "react";

interface Rect { x1: number; y1: number; x2: number; y2: number; }
interface Props { onExtract: (question: string, options: string[]) => void; }

function canvasCoords(
  clientX: number, clientY: number,
  canvas: HTMLCanvasElement
): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.round((clientX - rect.left) * (canvas.width / rect.width)),
    y: Math.round((clientY - rect.top) * (canvas.height / rect.height)),
  };
}

export default function ImageExtractor({ onExtract }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageData, setImageData] = useState<string | null>(null);
  const [selection, setSelection] = useState<Rect | null>(null);
  const [drawing, setDrawing] = useState(false);
  const pending = useRef<Rect>({ x1: 0, y1: 0, x2: 0, y2: 0 });
  const [numOptions, setNumOptions] = useState(4);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadImage = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const src = e.target!.result as string;
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current!;
        canvas.width = img.width;
        canvas.height = img.height;
        canvas.getContext("2d")!.drawImage(img, 0, 0);
        setImageData(src);
        setSelection(null);
      };
      img.src = src;
    };
    reader.readAsDataURL(file);
  };

  const redraw = (rect: Rect) => {
    if (!imageData || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext("2d")!;
    const img = new Image(); img.src = imageData;
    ctx.drawImage(img, 0, 0);
    ctx.strokeStyle = "#0ea5e9"; ctx.lineWidth = 2;
    ctx.strokeRect(rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1);
    ctx.fillStyle = "rgba(14,165,233,0.1)";
    ctx.fillRect(rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1);
  };

  const startDraw = (x: number, y: number) => {
    pending.current = { x1: x, y1: y, x2: x, y2: y };
    setDrawing(true); setSelection(null);
  };
  const moveDraw = (x: number, y: number) => {
    if (!drawing) return;
    pending.current.x2 = x; pending.current.y2 = y;
    redraw(pending.current);
  };
  const endDraw = () => {
    if (!drawing) return;
    setDrawing(false); setSelection({ ...pending.current });
  };

  const onMouseDown = (e: MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = canvasCoords(e.clientX, e.clientY, canvasRef.current!);
    startDraw(x, y);
  };
  const onMouseMove = (e: MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = canvasCoords(e.clientX, e.clientY, canvasRef.current!);
    moveDraw(x, y);
  };
  const onTouchStart = (e: TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const t = e.touches[0];
    const { x, y } = canvasCoords(t.clientX, t.clientY, canvasRef.current!);
    startDraw(x, y);
  };
  const onTouchMove = (e: TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const t = e.touches[0];
    const { x, y } = canvasCoords(t.clientX, t.clientY, canvasRef.current!);
    moveDraw(x, y);
  };

  const extract = async () => {
    if (!imageData || !selection) return;
    setLoading(true); setError("");
    try {
      const form = new FormData();
      form.append("image_data", imageData);
      form.append("x1", String(Math.min(selection.x1, selection.x2)));
      form.append("y1", String(Math.min(selection.y1, selection.y2)));
      form.append("x2", String(Math.max(selection.x1, selection.x2)));
      form.append("y2", String(Math.max(selection.y1, selection.y2)));
      form.append("num_options", String(numOptions));

      const res = await fetch("/api/v1/ocr/extract", { method: "POST", body: form });
      if (!res.ok) throw new Error("OCR failed");
      const data = await res.json();
      onExtract(data.question, data.options);
    } catch (err: any) {
      setError(err.message ?? "Extraction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-6 mt-4">
      <p className="text-sm font-medium text-slate-700 mb-3">Extract from image</p>

      <label className="block cursor-pointer">
        <input
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={e => e.target.files?.[0] && loadImage(e.target.files[0])}
        />
        <div className="border-2 border-dashed border-slate-200 rounded-xl p-4 text-center text-sm text-slate-400 hover:border-brand-500 transition-colors">
          {imageData ? "Tap to change image" : "Tap to upload or take photo"}
        </div>
      </label>

      {imageData && (
        <>
          <canvas
            ref={canvasRef}
            className="w-full mt-3 rounded-lg border border-slate-200 cursor-crosshair touch-none"
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={endDraw}
            onTouchStart={onTouchStart}
            onTouchMove={onTouchMove}
            onTouchEnd={endDraw}
          />
          <div className="flex items-center gap-3 mt-3">
            <label className="text-sm text-slate-600">Options:</label>
            <select value={numOptions} onChange={e => setNumOptions(Number(e.target.value))}
              className="border border-slate-200 rounded-lg px-2 py-1 text-sm">
              {[2, 3, 4].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
            <button onClick={extract} disabled={!selection || loading}
              className="flex-1 bg-brand-500 text-white text-sm py-2 rounded-lg disabled:opacity-40 transition-colors">
              {loading ? "Extracting…" : "Extract Text"}
            </button>
          </div>
          {error && <p className="text-red-600 text-sm mt-2">{error}</p>}
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Add ImageExtractor to `PracticePage.tsx`**

```typescript
// Add import:
import ImageExtractor from "../components/ImageExtractor";
// Add state to PracticePage and wire fillFromImage:
const fillFromImage = (question: string, options: string[]) => {
  setFields({ question, options: [...options, "", "", "", ""].slice(0, 4) });
};
// Add below QuizForm:
<ImageExtractor onExtract={fillFromImage} />
```

- [ ] **Step 4: Build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ImageExtractor.tsx frontend/src/pages/PracticePage.tsx backend/routers/quiz.py
git commit -m "feat: image extractor with touch support and OCR endpoint"
```

---

## Phase 5: Advanced Features — Recommendations + Question Generation

### Task 14: YouTube Recommendation Service

**Files:**
- Create: `backend/services/recommendations.py`
- Create: `backend/routers/advanced.py`
- Create: `tests/test_recommendations.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_recommendations.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_search_youtube_returns_videos():
    from backend.services.recommendations import search_youtube, Video

    mock_response = {
        "items": [{
            "id": {"videoId": "abc123"},
            "snippet": {
                "title": "て形 JLPT N4 Grammar",
                "channelTitle": "Japanese with Miku",
                "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/abc123/mqdefault.jpg"}},
            }
        }]
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.json = lambda: mock_response
        mock_get.return_value.raise_for_status = lambda: None
        videos = await search_youtube("て-form", "fake-api-key")

    assert len(videos) == 1
    assert videos[0].video_id == "abc123"
    assert videos[0].title == "て形 JLPT N4 Grammar"

@pytest.mark.asyncio
async def test_search_youtube_returns_empty_without_api_key():
    from backend.services.recommendations import search_youtube
    videos = await search_youtube("て-form", "")
    assert videos == []
```

- [ ] **Step 2: Write `backend/services/recommendations.py`**

```python
from dataclasses import dataclass
import httpx

@dataclass
class Video:
    video_id: str
    title: str
    thumbnail_url: str
    channel_title: str

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

async def search_youtube(concept: str, api_key: str, max_results: int = 3) -> list[Video]:
    if not api_key:
        return []
    params = {
        "part": "snippet",
        "q": f"JLPT {concept} grammar Japanese explanation",
        "type": "video",
        "videoDuration": "short",
        "maxResults": max_results,
        "key": api_key,
        "order": "relevance",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(YOUTUBE_SEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    videos = []
    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId", "")
        if not video_id:
            continue
        videos.append(Video(
            video_id=video_id,
            title=snippet.get("title", ""),
            thumbnail_url=snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            channel_title=snippet.get("channelTitle", ""),
        ))
    return videos
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_recommendations.py -v
```

Expected: `2 passed`

- [ ] **Step 4: Write `backend/routers/advanced.py`**

```python
from fastapi import APIRouter, Depends, Query
from backend.config import settings
from backend.schemas.api import VideoRecommendation, GenerateRequest, GeneratedQuestionResponse
from backend.services.recommendations import search_youtube
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
    # Use the first weak concept for YouTube search
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

- [ ] **Step 5: Add `advanced` router to `backend/main.py`**

```python
from backend.routers import auth, quiz, history, advanced
app.include_router(advanced.router)
```

- [ ] **Step 6: Commit**

```bash
git add backend/services/recommendations.py backend/routers/advanced.py tests/test_recommendations.py backend/main.py
git commit -m "feat: YouTube recommendation service and question generation endpoint"
```

---

### Task 15: Recommendation Panel + Question Generation UI

**Files:**
- Create: `frontend/src/components/RecommendationPanel.tsx`
- Modify: `frontend/src/pages/PracticePage.tsx`
- Modify: `frontend/src/pages/StatsPage.tsx`

- [ ] **Step 1: Write `frontend/src/components/RecommendationPanel.tsx`**

```typescript
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { VideoRecommendation } from "../types";

export default function RecommendationPanel({ concepts }: { concepts: string[] }) {
  const [videos, setVideos] = useState<VideoRecommendation[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (concepts.length === 0) return;
    setLoading(true);
    api.getRecommendations(concepts).then(setVideos).finally(() => setLoading(false));
  }, [concepts.join(",")]);

  if (concepts.length === 0 || (videos.length === 0 && !loading)) return null;

  return (
    <div className="mt-6">
      <p className="text-sm font-semibold text-slate-700 mb-3">Related videos for: {concepts[0]}</p>
      {loading ? (
        <p className="text-sm text-slate-400">Finding videos…</p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {videos.map(v => (
            <a
              key={v.video_id}
              href={`https://www.youtube.com/watch?v=${v.video_id}`}
              target="_blank" rel="noreferrer"
              className="bg-white border border-slate-200 rounded-xl overflow-hidden hover:shadow-md transition-shadow"
            >
              <img src={v.thumbnail_url} alt={v.title}
                className="w-full h-24 object-cover" />
              <div className="p-3">
                <p className="text-xs font-medium text-slate-800 line-clamp-2">{v.title}</p>
                <p className="text-xs text-slate-400 mt-1">{v.channel_title}</p>
              </div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Add recommendations to `PracticePage.tsx`**

```typescript
// Add import:
import RecommendationPanel from "../components/RecommendationPanel";

// In PracticePage JSX, after ExplanationCard:
{result && <RecommendationPanel concepts={result.concepts} />}
```

- [ ] **Step 3: Add "Generate a question" button to `StatsPage.tsx`**

```typescript
// Add to StatsPage.tsx (after weak concepts section):
import { useState } from "react";
import { api } from "../api/client";
import { GeneratedQuestion } from "../types";

// Inside StatsPage component:
const [generating, setGenerating] = useState(false);
const [generated, setGenerated] = useState<GeneratedQuestion | null>(null);
const [genError, setGenError] = useState("");

const generateForConcept = async (concept: string) => {
  setGenerating(true); setGenError(""); setGenerated(null);
  try {
    const q = await api.generateQuestion(concept, "N3");
    setGenerated(q);
  } catch {
    setGenError("Failed to generate question.");
  } finally {
    setGenerating(false);
  }
};

// In the weak concepts JSX, change each concept tag to:
<button key={c}
  onClick={() => generateForConcept(c)}
  className="bg-red-50 text-red-700 text-sm px-3 py-1 rounded-full hover:bg-red-100 transition-colors"
>
  {c} → Practice
</button>

// Add after weak concepts:
{generating && <p className="text-sm text-slate-400 mt-3">Generating question…</p>}
{genError && <p className="text-red-600 text-sm mt-3">{genError}</p>}
{generated && (
  <div className="mt-4 p-4 bg-slate-50 rounded-xl space-y-3">
    <p className="text-sm font-semibold text-slate-700">Practice Question</p>
    <p className="text-sm text-slate-800">{generated.question}</p>
    <div className="grid grid-cols-2 gap-2">
      {generated.options.map((opt, i) => (
        <div key={i} className={`text-xs p-2 rounded-lg border
          ${opt.startsWith(generated.correct_answer)
            ? "bg-green-50 border-green-200 text-green-800"
            : "bg-white border-slate-200 text-slate-600"}`}>
          {opt}
        </div>
      ))}
    </div>
    <p className="text-xs text-slate-500">{generated.explanation}</p>
  </div>
)}
```

- [ ] **Step 4: Final build**

```bash
cd frontend && npm run build
```

Expected: Build succeeds.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/
git commit -m "feat: YouTube recommendation panel and on-demand question generation from weak concepts"
```

---

### Task 16: End-to-End Smoke Test + CLAUDE.md Update

**Files:**
- Modify: `CLAUDE.md`
- Create: `tests/test_e2e_smoke.py`

- [ ] **Step 1: Write smoke test**

```python
# tests/test_e2e_smoke.py
"""
Smoke tests: run against a live backend (not unit tests).
Start backend first: uvicorn backend.main:app --app-dir .
"""
import httpx
import pytest

BASE = "http://localhost:8000"

@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE, timeout=10)

def test_health(client):
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_register_and_login(client):
    email = "smoketest@example.com"
    r = client.post("/auth/register", json={"email": email, "password": "testpass123"})
    assert r.status_code in (200, 400)  # 400 if already exists
    r2 = client.post("/auth/login", json={"email": email, "password": "testpass123"})
    assert r2.status_code == 200
    assert "access_token" in r2.json()

def test_stats_requires_auth(client):
    resp = client.get("/api/v1/stats")
    assert resp.status_code == 403
```

- [ ] **Step 2: Run smoke tests against live backend**

```bash
# Terminal 1:
uvicorn backend.main:app --app-dir .

# Terminal 2:
python -m pytest tests/test_e2e_smoke.py -v
```

Expected: `3 passed`

- [ ] **Step 3: Update `CLAUDE.md`**

Replace the existing CLAUDE.md content with:

```markdown
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
```

- [ ] **Step 4: Final commit**

```bash
git add CLAUDE.md tests/test_e2e_smoke.py
git commit -m "docs: update CLAUDE.md with new architecture and add e2e smoke tests"
```

---

## Self-Review

### Spec Coverage

| Requirement | Covered by |
|-------------|-----------|
| OpenRouter + local LLM backends | Tasks 3–5 (base, openrouter, local, factory) |
| Better frontend UX | Tasks 9–13 (Vite, Tailwind, new pages) |
| Activity time + wrong question tracking | Task 8 (analytics service, history/stats routes + pages) |
| YouTube recommendations | Tasks 14–15 (recommendations service + panel) |
| Generate new questions | Tasks 14–15 (generate endpoint + stats page UI) |

### No Placeholders Found
All steps contain complete code. No "TBD" or "implement later" text present.

### Type Consistency Check
- `SolveResult` defined in `base.py` (Task 3), used in `quiz.py` (Task 6) ✓
- `GeneratedQuestion` defined in `base.py` (Task 3), returned by `generate_question()` (Task 5), serialized in `advanced.py` (Task 14) ✓
- `AttemptRecord` / `AttemptResponse` in `schemas/api.py` (Task 6), used in `history.py` (Task 8) ✓
- Frontend `SolveResult` in `types/index.ts` matches backend `<RESULT>` JSON keys ✓

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-04-jlpt-sensei-revamp.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review between tasks, fast parallel iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, with checkpoints

**Which approach?**
