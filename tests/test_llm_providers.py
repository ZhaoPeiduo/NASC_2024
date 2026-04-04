# tests/test_llm_providers.py
import inspect
import pytest
from backend.services.llm.base import LLMProvider, SolveResult, GeneratedQuestion


def test_solve_result_has_required_fields():
    result = SolveResult(
        answer="A",
        explanation="AはXXXです",
        wrong_options={"B": "BはXXX", "C": "CはXXX", "D": "DはXXX"},
        concepts=["て-form"],
    )
    assert result.answer == "A"
    assert len(result.concepts) == 1
    assert "B" in result.wrong_options


def test_llm_provider_is_abstract():
    assert inspect.isabstract(LLMProvider)


def test_generated_question_fields():
    q = GeneratedQuestion(
        question="問題",
        options=["A: あ", "B: い", "C: う", "D: え"],
        correct_answer="A",
        explanation="Aです",
        concepts=["て-form"],
    )
    assert q.correct_answer == "A"
    assert len(q.options) == 4


import json
import pytest
from unittest.mock import AsyncMock, patch
from backend.services.llm.openrouter import OpenRouterProvider

SOLVE_PROMPT_RESPONSE = (
    "AはXXXです。て形は継続を表します。\n"
    '<RESULT>{"answer":"A","explanation":"AはXXXです","wrong_options":{"B":"BはYYY","C":"CはZZZ","D":"DはWWW"},"concepts":["て-form"]}</RESULT>'
)


@pytest.mark.asyncio
async def test_openrouter_stream_solve_yields_tokens():
    provider = OpenRouterProvider(api_key="fake-key", model="openai/gpt-4o-mini")

    async def fake_stream(*args, **kwargs):
        for char in SOLVE_PROMPT_RESPONSE:
            yield char

    with patch.object(provider, "_stream_chat", side_effect=fake_stream):
        tokens = []
        async for token in provider.stream_solve(
            "問題文", ["A: 勉強して", "B: 勉強した", "C: 勉強する", "D: 勉強し"]
        ):
            tokens.append(token)

    full_text = "".join(tokens)
    assert "AはXXXです" in full_text
    assert "<RESULT>" in full_text


from backend.services.llm.factory import get_llm_provider
from backend.services.llm.openrouter import OpenRouterProvider


def test_factory_returns_openrouter_when_configured(monkeypatch):
    monkeypatch.setattr("backend.services.llm.factory.settings.llm_provider", "openrouter")
    monkeypatch.setattr("backend.services.llm.factory.settings.openrouter_api_key", "sk-test")
    monkeypatch.setattr("backend.services.llm.factory.settings.openrouter_model", "openai/gpt-4o-mini")
    # Clear lru_cache so monkeypatched settings take effect
    get_llm_provider.cache_clear()
    provider = get_llm_provider()
    assert isinstance(provider, OpenRouterProvider)
    get_llm_provider.cache_clear()  # clean up after test


def test_factory_raises_for_unknown_provider(monkeypatch):
    monkeypatch.setattr("backend.services.llm.factory.settings.llm_provider", "unknown")
    get_llm_provider.cache_clear()
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_llm_provider()
    get_llm_provider.cache_clear()
