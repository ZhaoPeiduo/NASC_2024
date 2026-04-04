import json
from fastapi import APIRouter, Depends, Form, HTTPException
from sse_starlette.sse import EventSourceResponse

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
            yield {"event": "token", "data": token}

        result = parse_result_block(full_text)
        if result:
            yield {
                "event": "result",
                "data": json.dumps(
                    {
                        "answer": result.answer,
                        "explanation": result.explanation,
                        "wrong_options": result.wrong_options,
                        "concepts": result.concepts,
                    }
                ),
            }
        yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@router.post("/ocr/extract")
async def ocr_extract(
    image_data: str = Form(...),
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
        raise HTTPException(status_code=422, detail=f"OCR failed: {e}")
