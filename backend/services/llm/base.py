from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class SolveResult:
    answer: str                       # "A", "B", "C", or "D"
    explanation: str                  # Why the correct answer is right
    wrong_options: dict[str, str]     # {"B": "why B is wrong", ...}
    concepts: list[str]               # ["て-form", "conditional"]


@dataclass
class GeneratedQuestion:
    question: str
    options: list[str]                # ["A: ...", "B: ...", "C: ...", "D: ..."]
    correct_answer: str               # "A"/"B"/"C"/"D"
    explanation: str
    concepts: list[str]


class LLMProvider(ABC):
    @abstractmethod
    async def stream_solve(
        self,
        question: str,
        options: list[str],
    ) -> AsyncIterator[str]:
        """Yield raw text tokens. Final token sequence includes a JSON block:
        <RESULT>{"answer":"A","explanation":"...","wrong_options":{...},"concepts":[...]}</RESULT>
        """
        ...

    @abstractmethod
    async def generate_question(
        self,
        concept: str,
        level: str,
    ) -> GeneratedQuestion:
        """Return a fully formed question with answer and explanation."""
        ...
