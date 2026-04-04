# tests/test_models.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from backend.models.tables import Base, User, Attempt

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
