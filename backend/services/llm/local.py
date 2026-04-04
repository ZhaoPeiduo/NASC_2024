"""
Local HuggingFace inference provider.
Requires: torch, transformers, accelerate (GPU recommended).
Falls back to CPU if no GPU available but will be very slow.
"""
import json
import asyncio
from typing import AsyncIterator
from backend.services.llm.base import LLMProvider, GeneratedQuestion

SOLVE_SYSTEM = """あなたはJLPT日本語文法の先生です。
問題を分析し、正しい答えを選んで理由を説明してください。
最後に必ずこのブロックで終わってください:
<RESULT>{"answer":"A","explanation":"...","wrong_options":{"B":"...","C":"...","D":"..."},"concepts":["..."]}</RESULT>"""


class LocalModelProvider(LLMProvider):
    def __init__(self, model_name: str) -> None:
        # Lazy imports — only triggered when local provider is selected
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def _generate_sync(self, prompt: str, max_new_tokens: int = 512) -> str:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
            )
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    async def stream_solve(self, question: str, options: list[str]) -> AsyncIterator[str]:
        options_text = "\n".join(options)
        prompt = (
            f"{SOLVE_SYSTEM}\n\n"
            f"問題: {question}\n選択肢:\n{options_text}\n\n回答:"
        )
        text = await asyncio.get_event_loop().run_in_executor(
            None, self._generate_sync, prompt
        )
        chunk_size = 4
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]
            await asyncio.sleep(0)

    async def generate_question(self, concept: str, level: str) -> GeneratedQuestion:
        prompt = (
            f"JLPT {level}レベルの{concept}に関する文法問題を1問作成してください。\n"
            f'JSON形式で回答: {{"question":"...","options":["A:...","B:...","C:...","D:..."],'
            f'"correct_answer":"A","explanation":"...","concepts":["{concept}"]}}'
        )
        text = await asyncio.get_event_loop().run_in_executor(
            None, self._generate_sync, prompt, 300
        )
        data = json.loads(text.strip())
        return GeneratedQuestion(**data)
