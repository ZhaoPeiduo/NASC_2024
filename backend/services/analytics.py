import json
from collections import Counter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
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


async def generate_and_cache_explanation(
    db: AsyncSession,
    attempt: Attempt,
    provider,
) -> str:
    options_text = "\n".join(json.loads(attempt.options or "[]"))
    prompt = (
        f"Japanese grammar question:\n{attempt.question_text}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Correct answer: {attempt.correct_answer}\n"
        # In quiz mode, llm_answer stores the user's selected option
        f"User's answer: {attempt.llm_answer}\n\n"
        "In 2-3 sentences, explain why the correct answer is right and why the user's answer is wrong. "
        "Be concise and educational."
    )
    explanation = await provider.complete(prompt)
    attempt.explanation = explanation
    await db.commit()
    return explanation


async def get_weak_concepts(
    db: AsyncSession, user_id: int, limit: int = 10
) -> list[str]:
    result = await db.execute(
        select(Attempt).where(
            Attempt.user_id == user_id,
            Attempt.user_marked_correct == False,  # noqa: E712
        )
    )
    wrong_attempts = result.scalars().all()
    counter: Counter = Counter()
    for attempt in wrong_attempts:
        concepts = json.loads(attempt.concepts or "[]")
        counter.update(concepts)
    return [concept for concept, _ in counter.most_common(limit)]
