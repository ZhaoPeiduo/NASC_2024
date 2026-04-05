import json

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.auth.dependencies import get_current_user
from backend.models.tables import User
from backend.schemas.api import (
    UploadPracticeResponse, PracticeQuestion,
    AnalyzeRequest, AnalyzeResponse, AnalysisItem,
    BatchRecordRequest,
)
from backend.services.practice import (
    parse_grammar_csv,
    get_wrong_history_sample,
    analyze_wrong_answers,
)
from backend.services.analytics import record_attempt
from backend.services.llm.factory import get_llm_provider

router = APIRouter(prefix="/api/v1/practice")


@router.post("/upload", response_model=UploadPracticeResponse)
async def upload_csv(
    file: UploadFile = File(...),
    include_history: bool = Form(False),
    history_count: int = Form(5),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    content = (await file.read()).decode("utf-8", errors="replace")
    questions = parse_grammar_csv(content)
    if not questions:
        raise HTTPException(status_code=422, detail="No valid questions found in CSV")

    if include_history and history_count > 0:
        history_qs = await get_wrong_history_sample(db, current_user.id, history_count)
        questions = questions + history_qs  # history appended at end

    return UploadPracticeResponse(
        questions=[PracticeQuestion(**q) for q in questions],
        total=len(questions),
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    body: AnalyzeRequest,
    current_user: User = Depends(get_current_user),
):
    provider = get_llm_provider()
    wrong_dicts = [item.model_dump() for item in body.wrong_items]
    analyses = await analyze_wrong_answers(wrong_dicts, provider)
    return AnalyzeResponse(
        analyses=[AnalysisItem(**a) for a in analyses]
    )


@router.post("/record-batch", status_code=201)
async def record_batch(
    body: BatchRecordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    for item in body.attempts:
        await record_attempt(
            db=db,
            user_id=current_user.id,
            question_text=item.question_text,
            options=item.options,
            correct_answer=item.correct_answer,
            llm_answer=item.user_answer,
            explanation="",
            concepts=[],
            user_marked_correct=item.user_marked_correct,
        )
    return {"recorded": len(body.attempts)}
