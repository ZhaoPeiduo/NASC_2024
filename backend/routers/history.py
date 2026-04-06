import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.database import get_db
from backend.models.tables import Attempt, User
from backend.auth.dependencies import get_current_user
from backend.schemas.api import (
    AttemptResponse, AttemptRecord, StatsResponse, WeakConceptsResponse,
)
from backend.services.analytics import get_stats, get_weak_concepts, record_attempt
from backend.services.llm.factory import get_llm_provider

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
            options=json.loads(a.options or "[]"),
            explanation=a.explanation or "",
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
        explanation=body.explanation,
        concepts=body.concepts,
        user_marked_correct=body.user_marked_correct,
    )
    return {"status": "recorded"}


@router.get("/history/weak-concepts", response_model=WeakConceptsResponse)
async def history_weak_concepts(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    concepts = await get_weak_concepts(db, current_user.id, limit=limit)
    return WeakConceptsResponse(concepts=concepts)


@router.post("/history/{attempt_id}/explain")
async def explain_attempt(
    attempt_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Attempt).where(
            Attempt.id == attempt_id,
            Attempt.user_id == current_user.id,
        )
    )
    attempt = result.scalar_one_or_none()
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    if attempt.explanation:
        return {"explanation": attempt.explanation}

    provider = get_llm_provider()
    options_text = "\n".join(json.loads(attempt.options or "[]"))
    prompt = (
        f"Japanese grammar question:\n{attempt.question_text}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Correct answer: {attempt.correct_answer}\n"
        f"User's answer: {attempt.llm_answer}\n\n"
        "In 2-3 sentences, explain why the correct answer is right and why the user's answer is wrong. "
        "Be concise and educational."
    )
    explanation = await provider.complete(prompt)

    attempt.explanation = explanation
    await db.commit()
    return {"explanation": explanation}


@router.get("/stats", response_model=StatsResponse)
async def get_user_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stats = await get_stats(db, current_user.id)
    weak = await get_weak_concepts(db, current_user.id)
    return StatsResponse(weak_concepts=weak, **stats)
