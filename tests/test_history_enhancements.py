import json
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from backend.main import app
from backend.database import get_db
from backend.models.tables import Base, User
from backend.auth.jwt_handler import create_access_token
from backend.services.analytics import record_attempt


@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    Session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with Session() as session:
        yield session
    await engine.dispose()


@pytest.fixture
async def test_user(db_session):
    user = User(email="testuser@example.com", hashed_password="hashed")
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def auth_client(db_session, test_user):
    token = create_access_token(test_user.id)

    # Override get_db dependency to use the in-memory session
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        client.headers["Authorization"] = f"Bearer {token}"
        yield client

    app.dependency_overrides.clear()


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
