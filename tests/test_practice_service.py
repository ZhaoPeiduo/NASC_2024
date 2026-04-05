import pytest
from backend.services.practice import parse_grammar_csv


def test_parse_basic_row():
    csv = "Question,Options,Answer\n今日は___寒い。,a.とても b.たいして c.かなり,b\n"
    qs = parse_grammar_csv(csv)
    assert len(qs) == 1
    assert qs[0]["question"] == "今日は___寒い。"
    assert qs[0]["correct_answer"] == "B"
    assert qs[0]["options"] == ["A: とても", "B: たいして", "C: かなり"]
    assert qs[0]["from_history"] is False


def test_parse_four_options():
    csv = "Question,Options,Answer\nTest Q,a.aaa b.bbb c.ccc d.ddd,d\n"
    qs = parse_grammar_csv(csv)
    assert len(qs) == 1
    assert qs[0]["correct_answer"] == "D"
    assert len(qs[0]["options"]) == 4


def test_parse_skips_empty_question():
    csv = "Question,Options,Answer\n,a.x b.y,a\n"
    qs = parse_grammar_csv(csv)
    assert qs == []


def test_parse_multiple_rows():
    csv = (
        "Question,Options,Answer\n"
        "Q1,a.aaa b.bbb c.ccc,a\n"
        "Q2,a.xxx b.yyy c.zzz,c\n"
    )
    qs = parse_grammar_csv(csv)
    assert len(qs) == 2
    assert qs[1]["correct_answer"] == "C"
