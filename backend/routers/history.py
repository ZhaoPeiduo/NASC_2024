import json
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db
from backend.models.tables import Attempt, User
from backend.auth.dependencies import get_current_user
from backend.schemas.api import AttemptResponse, AttemptRecord, StatsResponse
from backend.services.analytics import get_stats, get_weak_concepts, record_attempt

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


@router.post("/history/record", status_code=201)
async def record(
    body: AttemptRecord,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await record_attempt(
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


@router.get("/stats", response_model=StatsResponse)
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stats = await get_stats(db, current_user.id)
    weak = await get_weak_concepts(db, current_user.id)
    return StatsResponse(weak_concepts=weak, **stats)
