import json
import httpx
from typing import AsyncIterator
from backend.services.llm.base import LLMProvider, SolveResult, GeneratedQuestion

SOLVE_SYSTEM_PROMPT = """You are a JLPT Japanese grammar teacher.
Given a multiple-choice grammar question, select the correct answer and explain:
1. Why the correct answer is right
2. Why each wrong option is incorrect
3. List the grammar concepts tested

Always end your response with this exact block (fill in real values):
<RESULT>{"answer":"A","explanation":"...","wrong_options":{"B":"...","C":"...","D":"..."},"concepts":["..."]}</RESULT>"""

GENERATE_SYSTEM_PROMPT = """You are a JLPT grammar question creator.
Generate one multiple-choice grammar question at the specified JLPT level.
Respond ONLY with valid JSON, no other text."""


class OpenRouterProvider(LLMProvider):
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    async def _stream_chat(self, messages: list[dict]) -> AsyncIterator[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://jlpt-sensei.app",
        }
        body = {"model": self.model, "messages": messages, "stream": True}

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{self.BASE_URL}/chat/completions",
                headers=headers,
                json=body,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def stream_solve(self, question: str, options: list[str]) -> AsyncIterator[str]:
        options_text = "\n".join(options)
        user_msg = f"問題: {question}\n選択肢:\n{options_text}"
        messages = [
            {"role": "system", "content": SOLVE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        async for token in self._stream_chat(messages):
            yield token

    async def generate_question(self, concept: str, level: str) -> GeneratedQuestion:
        user_msg = (
            f"Generate a JLPT {level} grammar question about the concept: {concept}\n\n"
            f'Respond with JSON: {{"question":"...","options":["A: ...","B: ...","C: ...","D: ..."],'
            f'"correct_answer":"A","explanation":"...","concepts":["{concept}"]}}'
        )
        messages = [
            {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        full = ""
        async for token in self._stream_chat(messages):
            full += token
        data = json.loads(full.strip())
        return GeneratedQuestion(**data)

    async def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        full = ""
        async for token in self._stream_chat(messages):
            full += token
        return full
