from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class SolveResult:
    answer: str
    explanation: str
    wrong_options: dict[str, str]
    concepts: list[str]


@dataclass
class GeneratedQuestion:
    question: str
    options: list[str]
    correct_answer: str
    explanation: str
    concepts: list[str]


class LLMProvider(ABC):
    @abstractmethod
    async def stream_solve(
        self,
        question: str,
        options: list[str],
    ) -> AsyncIterator[str]:
        """Yield raw text tokens ending with <RESULT>{json}</RESULT>."""
        ...

    @abstractmethod
    async def generate_question(
        self,
        concept: str,
        level: str,
    ) -> GeneratedQuestion:
        """Return a fully formed question with answer and explanation."""
        ...

    @abstractmethod
    async def complete(self, prompt: str) -> str:
        """Non-streaming completion for structured tasks (analysis, recommendations)."""
        ...
