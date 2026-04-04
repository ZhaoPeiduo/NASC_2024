import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_search_youtube_returns_videos():
    from backend.services.recommendations import search_youtube, Video

    mock_response = {
        "items": [{
            "id": {"videoId": "abc123"},
            "snippet": {
                "title": "て形 JLPT N4 Grammar",
                "channelTitle": "Japanese with Miku",
                "thumbnails": {"medium": {"url": "https://img.youtube.com/vi/abc123/mqdefault.jpg"}},
            }
        }]
    }

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value.json = lambda: mock_response
        mock_get.return_value.raise_for_status = lambda: None
        videos = await search_youtube("て-form", "fake-api-key")

    assert len(videos) == 1
    assert videos[0].video_id == "abc123"
    assert videos[0].title == "て形 JLPT N4 Grammar"


@pytest.mark.asyncio
async def test_search_youtube_returns_empty_without_api_key():
    from backend.services.recommendations import search_youtube
    videos = await search_youtube("て-form", "")
    assert videos == []
