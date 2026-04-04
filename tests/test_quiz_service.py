# tests/test_quiz_service.py
import pytest
from backend.services.quiz import solve_question, parse_result_block


async def fake_stream():
    response = (
        "AはXXXです。て形は継続を表します。\n"
        '<RESULT>{"answer":"A","explanation":"AはXXXです","wrong_options":{"B":"BはYYY","C":"CはZZZ","D":"DはWWW"},"concepts":["て-form"]}</RESULT>'
    )
    for char in response:
        yield char


@pytest.mark.asyncio
async def test_solve_question_yields_tokens():
    tokens = []
    async for token in solve_question(
        question="彼女は毎日日本語を＿＿＿います。",
        options=["A: 勉強して", "B: 勉強した", "C: 勉強する", "D: 勉強し"],
        provider_stream=fake_stream(),
    ):
        tokens.append(token)

    full = "".join(tokens)
    assert "AはXXXです" in full
    assert "<RESULT>" in full


def test_parse_result_block_extracts_data():
    text = 'some text <RESULT>{"answer":"B","explanation":"Bです","wrong_options":{"A":"Aは違う"},"concepts":["conditional"]}</RESULT>'
    result = parse_result_block(text)
    assert result is not None
    assert result.answer == "B"
    assert result.concepts == ["conditional"]
    assert "A" in result.wrong_options


def test_parse_result_block_returns_none_when_missing():
    result = parse_result_block("no result block here")
    assert result is None
