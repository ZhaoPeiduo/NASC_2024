import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.auth.dependencies import get_current_user
from backend.models.tables import User


def _fake_user():
    u = User()
    u.id = 1
    u.email = "test@example.com"
    return u


def _mock_jwt():
    app.dependency_overrides[get_current_user] = lambda: _fake_user()


def _clear_jwt():
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_ocr_batch_returns_questions():
    _mock_jwt()
    try:
        from PIL import Image
        import io

        img = Image.new("RGB", (10, 10), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        structured = {
            "question": "彼女は日本語を＿＿います。",
            "options": ["A: 話して", "B: 話した", "C: 話す", "D: 話せ"],
            "answer": "A",
        }

        with patch("backend.routers.quiz.extract_text", return_value={"question": "raw text", "options": []}):
            with patch("backend.routers.quiz._structure_ocr_with_llm", new_callable=AsyncMock, return_value=structured):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/quiz/ocr/batch",
                        files={"images": ("test.png", buf.getvalue(), "image/png")},
                    )

        assert response.status_code == 200
        data = response.json()
        assert "questions" in data
        assert len(data["questions"]) == 1
        assert data["questions"][0]["question"] == "彼女は日本語を＿＿います。"
    finally:
        _clear_jwt()


@pytest.mark.asyncio
async def test_ocr_batch_skips_failed_structure():
    _mock_jwt()
    try:
        from PIL import Image
        import io

        img = Image.new("RGB", (10, 10), color=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        with patch("backend.routers.quiz.extract_text", return_value={"question": "garbage", "options": []}):
            with patch("backend.routers.quiz._structure_ocr_with_llm", new_callable=AsyncMock, return_value=None):
                async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                    response = await client.post(
                        "/api/v1/quiz/ocr/batch",
                        files={"images": ("test.png", buf.getvalue(), "image/png")},
                    )

        assert response.status_code == 200
        assert response.json()["questions"] == []
    finally:
        _clear_jwt()
