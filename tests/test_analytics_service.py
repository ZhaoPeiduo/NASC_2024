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
    assert attempts[0].user_marked_correct is True


@pytest.mark.asyncio
async def test_get_stats_calculates_correctly(db_session):
    session, user_id = db_session
    # 2 correct, 1 wrong
    for i, correct in enumerate([True, True, False]):
        await record_attempt(session, user_id, f"Q{i}", ["A","B","C","D"],
                             "A", "A" if correct else "B", "", [], correct)
    stats = await get_stats(session, user_id)
    assert stats["total_attempts"] == 3
    assert abs(stats["correct_rate"] - 2/3) < 0.001


@pytest.mark.asyncio
async def test_get_weak_concepts_returns_most_missed(db_session):
    session, user_id = db_session
    # 3 wrong attempts with "て-form", 1 wrong with "conditional"
    for _ in range(3):
        await record_attempt(session, user_id, "Q", ["A","B","C","D"],
                             "A", "B", "", ["て-form"], False)
    await record_attempt(session, user_id, "Q2", ["A","B","C","D"],
                         "A", "B", "", ["conditional"], False)
    weak = await get_weak_concepts(session, user_id, limit=2)
    assert weak[0] == "て-form"
    assert len(weak) == 2
