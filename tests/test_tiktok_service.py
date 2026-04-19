import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_fake_video(video_id="abc123", desc="Test video", nickname="TestUser"):
    v = MagicMock()
    v.as_dict = {
        "id": video_id,
        "desc": desc,
        "author": {"nickname": nickname},
    }
    v.bytes = AsyncMock(return_value=b"fakevideobytes")
    return v


@pytest.mark.asyncio
async def test_get_random_tiktok_video_trending():
    from backend.services.tiktok_service import get_random_tiktok_video

    fake_video = _make_fake_video("v1", "Japanese lesson", "JapaneseTeacher")

    mock_api = MagicMock()
    mock_api.__aenter__ = AsyncMock(return_value=mock_api)
    mock_api.__aexit__ = AsyncMock(return_value=False)
    mock_api.create_sessions = AsyncMock()

    async def fake_trending_videos(count=30):
        yield fake_video

    mock_api.trending = MagicMock()
    mock_api.trending.videos = fake_trending_videos

    with patch("backend.services.tiktok_service.TikTokApi", return_value=mock_api):
        result = await get_random_tiktok_video(topic="trending", ms_token="tok")

    assert result is not None
    assert result.video_id == "v1"
    assert result.title == "Japanese lesson"
    assert result.author == "JapaneseTeacher"


@pytest.mark.asyncio
async def test_get_random_tiktok_video_returns_none_on_empty():
    from backend.services.tiktok_service import get_random_tiktok_video

    mock_api = MagicMock()
    mock_api.__aenter__ = AsyncMock(return_value=mock_api)
    mock_api.__aexit__ = AsyncMock(return_value=False)
    mock_api.create_sessions = AsyncMock()

    async def fake_trending_videos(count=30):
        return
        yield

    mock_api.trending = MagicMock()
    mock_api.trending.videos = fake_trending_videos

    with patch("backend.services.tiktok_service.TikTokApi", return_value=mock_api):
        result = await get_random_tiktok_video(topic="trending", ms_token="tok")

    assert result is None


@pytest.mark.asyncio
async def test_fetch_video_bytes():
    from backend.services.tiktok_service import fetch_video_bytes

    fake_video = _make_fake_video("v2")

    mock_api = MagicMock()
    mock_api.__aenter__ = AsyncMock(return_value=mock_api)
    mock_api.__aexit__ = AsyncMock(return_value=False)
    mock_api.create_sessions = AsyncMock()
    mock_api.video = MagicMock(return_value=fake_video)

    with patch("backend.services.tiktok_service.TikTokApi", return_value=mock_api):
        data = await fetch_video_bytes(video_id="v2", ms_token="tok")

    assert data == b"fakevideobytes"


@pytest.mark.asyncio
async def test_get_random_tiktok_video_hashtag():
    from backend.services.tiktok_service import get_random_tiktok_video

    fake_video = _make_fake_video("v3", "Anime scene", "AnimeChannel")

    mock_api = MagicMock()
    mock_api.__aenter__ = AsyncMock(return_value=mock_api)
    mock_api.__aexit__ = AsyncMock(return_value=False)
    mock_api.create_sessions = AsyncMock()

    mock_hashtag = MagicMock()

    async def fake_hashtag_videos(count=30):
        yield fake_video

    mock_hashtag.videos = fake_hashtag_videos
    mock_api.hashtag = MagicMock(return_value=mock_hashtag)

    with patch("backend.services.tiktok_service.TikTokApi", return_value=mock_api):
        result = await get_random_tiktok_video(topic="anime", ms_token="tok")

    assert result is not None
    assert result.video_id == "v3"
    mock_api.hashtag.assert_called_once_with(name="anime")
