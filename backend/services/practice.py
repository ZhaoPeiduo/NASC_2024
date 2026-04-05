import csv
import io
import json
import re

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.tables import Attempt
from backend.services.llm.base import LLMProvider


def parse_grammar_csv(content: str) -> list[dict]:
    """Parse CSV with columns Question, Options, Answer.
    Options format: 'a.text b.text c.text [d.text]'
    Answer format: 'a'/'b'/'c'/'d'
    Returns list of {question, options, correct_answer, from_history} dicts.
    """
    reader = csv.DictReader(io.StringIO(content))
    questions = []
    for row in reader:
        question = row.get("Question", "").strip()
        if not question:
            continue
        opts_raw = row.get("Options", "").strip()
        matches = re.findall(r"([a-d])\.(.+?)(?=\s[a-d]\.|$)", opts_raw)
        options = [f"{letter.upper()}: {text.strip()}" for letter, text in matches]
        if not options:
            continue
        answer = row.get("Answer", "").strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            continue
        questions.append({
            "question": question,
            "options": options,
            "correct_answer": answer,
            "from_history": False,
        })
    return questions


async def get_wrong_history_sample(
    db: AsyncSession, user_id: int, n: int
) -> list[dict]:
    """Return up to n randomly sampled wrong attempts from history."""
    result = await db.execute(
        select(Attempt)
        .where(Attempt.user_id == user_id, Attempt.user_marked_correct == False)  # noqa: E712
        .order_by(func.random())
        .limit(n)
    )
    attempts = result.scalars().all()
    return [
        {
            "question": a.question_text,
            "options": json.loads(a.options or "[]"),
            "correct_answer": a.correct_answer,
            "from_history": True,
        }
        for a in attempts
    ]


async def analyze_wrong_answers(
    wrong_items: list[dict], provider: LLMProvider
) -> list[dict]:
    """Batch-analyze wrong answers via LLM. Returns list of analysis dicts."""
    if not wrong_items:
        return []

    lines = []
    for i, item in enumerate(wrong_items, 1):
        opts = ", ".join(item["options"])
        lines.append(
            f'{i}. Q: {item["question"]}\n'
            f'   Options: {opts}\n'
            f'   Correct: {item["correct_answer"]}, User chose: {item["user_answer"]}'
        )

    prompt = (
        "You are a JLPT grammar teacher. For each wrong answer below, write 1-2 sentences "
        "explaining why the correct answer is right and why the user's choice was wrong.\n\n"
        + "\n\n".join(lines)
        + "\n\nRespond ONLY with a JSON array (no markdown fences):\n"
        '[{"question_index": 1, "explanation": "..."}, ...]'
    )

    raw = await provider.complete(prompt)
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    parsed: list[dict] = []
    if match:
        try:
            candidate = json.loads(match.group(0))
            if isinstance(candidate, list):
                parsed = candidate
        except json.JSONDecodeError:
            parsed = []

    results = []
    for i, item in enumerate(wrong_items, 1):
        explanation = next(
            (p["explanation"] for p in parsed if p.get("question_index") == i), ""
        )
        results.append({
            "question": item["question"],
            "correct_answer": item["correct_answer"],
            "user_answer": item["user_answer"],
            "explanation": explanation,
        })
    return results
