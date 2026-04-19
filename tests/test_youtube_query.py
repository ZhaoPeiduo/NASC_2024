import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_generate_tutorial_query_returns_llm_string():
    from backend.services.youtube_query import generate_tutorial_query

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value="JLPT N4 te-form permission grammar")

    result = await generate_tutorial_query(
        question="彼女は毎日日本語を＿＿＿います。",
        correct_answer="A",
        user_answer="B",
        concepts=["て-form", "継続"],
        provider=mock_provider,
    )
    assert result == "JLPT N4 te-form permission grammar"
    mock_provider.complete.assert_called_once()


@pytest.mark.asyncio
async def test_generate_tutorial_query_falls_back_on_garbage_response():
    from backend.services.youtube_query import generate_tutorial_query

    mock_provider = MagicMock()
    # LLM returns a multi-line garbage response
    mock_provider.complete = AsyncMock(return_value="Here is a query:\nJLPT grammar\nFor your reference")

    result = await generate_tutorial_query(
        question="question text",
        correct_answer="A",
        user_answer="C",
        concepts=["は vs が"],
        provider=mock_provider,
    )
    # Fallback uses first concept
    assert "は vs が" in result


@pytest.mark.asyncio
async def test_get_random_japanese_song_returns_video():
    from backend.services.youtube_query import get_random_japanese_song
    from backend.services.recommendations import Video

    fake_video = Video(
        video_id="xyz789",
        title="夜に駆ける - YOASOBI",
        thumbnail_url="https://img.youtube.com/vi/xyz789/mqdefault.jpg",
        channel_title="YOASOBI Official",
    )

    with patch("backend.services.youtube_query.search_youtube", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [fake_video]
        result = await get_random_japanese_song(api_key="fake-key")

    assert result is not None
    assert result.video_id == "xyz789"


@pytest.mark.asyncio
async def test_get_random_japanese_song_returns_none_when_no_results():
    from backend.services.youtube_query import get_random_japanese_song

    with patch("backend.services.youtube_query.search_youtube", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = []
        result = await get_random_japanese_song(api_key="fake-key")

    assert result is None


@pytest.mark.asyncio
async def test_search_tutorials_deduplicates_by_concept():
    from backend.services.youtube_query import search_tutorials_for_wrong_answers
    from backend.services.recommendations import Video

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value="JLPT N4 grammar")

    fake_video = Video(
        video_id="v1", title="Tutorial", thumbnail_url="https://img.youtube.com/vi/v1/mqdefault.jpg",
        channel_title="Channel",
    )

    wrong_items = [
        {"question": "q1", "correct_answer": "A", "user_answer": "B", "concepts": ["て-form"]},
        {"question": "q2", "correct_answer": "C", "user_answer": "A", "concepts": ["て-form"]},  # duplicate concept
        {"question": "q3", "correct_answer": "B", "user_answer": "D", "concepts": ["は vs が"]},
    ]

    with patch("backend.services.youtube_query.search_youtube", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = [fake_video]
        results = await search_tutorials_for_wrong_answers(
            wrong_items=wrong_items, api_key="fake-key", provider=mock_provider
        )

    # Only 2 unique concepts, not 3 items
    assert len(results) == 2
    assert results[0]["concepts"] == ["て-form"]
    assert results[1]["concepts"] == ["は vs が"]
