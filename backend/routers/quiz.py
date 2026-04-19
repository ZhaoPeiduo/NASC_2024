import base64
import io
import json
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sse_starlette.sse import EventSourceResponse
from PIL import Image

from backend.schemas.api import SolveRequest, OcrBatchResponse, PracticeQuestion
from backend.services.llm.factory import get_llm_provider
from backend.services.quiz import solve_question, parse_result_block
from backend.services.ocr import extract_text
from backend.auth.dependencies import get_current_user, get_current_user_optional
from backend.models.tables import User

router = APIRouter(prefix="/api/v1")

_STRUCTURE_PROMPT = """You are parsing Japanese grammar question text extracted via OCR from a screenshot.
The raw text may contain noise or imperfect line breaks.

Raw OCR text:
{raw_text}

Output a JSON object with exactly these keys:
- "question": the question stem (Japanese text)
- "options": array of exactly 4 strings, each prefixed like ["A: ...", "B: ...", "C: ...", "D: ..."]
- "answer": one of "A", "B", "C", "D" (guess "A" if not determinable)

Output ONLY the JSON object. No other text, no markdown fences."""


async def _structure_ocr_with_llm(raw_text: str, provider) -> dict | None:
    """Ask the LLM to parse raw OCR text into structured question format."""
    try:
        response = await provider.complete(_STRUCTURE_PROMPT.format(raw_text=raw_text))
        cleaned = response.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)
        if not all(k in parsed for k in ("question", "options", "answer")):
            return None
        if len(parsed["options"]) != 4:
            return None
        return parsed
    except Exception:
        return None


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


@router.post("/quiz/ocr/batch", response_model=OcrBatchResponse)
async def ocr_batch(
    images: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Accept multiple screenshot uploads, OCR each, LLM-structure into questions.
    Failed extractions are silently skipped.
    """
    provider = get_llm_provider()
    questions: list[PracticeQuestion] = []

    for upload in images:
        img_bytes = await upload.read()
        try:
            img = Image.open(io.BytesIO(img_bytes))
            w, h = img.size
            b64 = "data:image/png;base64," + base64.b64encode(img_bytes).decode()
            ocr_result = extract_text(b64, 0, 0, w, h, 4)
            raw_text = ocr_result.get("question", "") + " " + " ".join(ocr_result.get("options", []))
            structured = await _structure_ocr_with_llm(raw_text.strip(), provider)
            if structured:
                questions.append(PracticeQuestion(
                    question=structured["question"],
                    options=structured["options"],
                    correct_answer=structured["answer"],
                ))
        except Exception:
            continue

    return OcrBatchResponse(questions=questions)
