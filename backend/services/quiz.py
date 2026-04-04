import json
import re
from typing import AsyncIterator, Optional
from backend.services.llm.base import SolveResult

RESULT_PATTERN = re.compile(r"<RESULT>(.*?)</RESULT>", re.DOTALL)


def parse_result_block(text: str) -> Optional[SolveResult]:
    match = RESULT_PATTERN.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(1))
        return SolveResult(
            answer=data["answer"],
            explanation=data.get("explanation", ""),
            wrong_options=data.get("wrong_options", {}),
            concepts=data.get("concepts", []),
        )
    except (json.JSONDecodeError, KeyError):
        return None


async def solve_question(
    question: str,
    options: list[str],
    provider_stream: AsyncIterator[str],
) -> AsyncIterator[str]:
    """Pass-through: yields raw tokens from the LLM stream."""
    async for token in provider_stream:
        yield token
