import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_get_media_recommendations_parses_llm_json():
    from backend.services.media_recommendations import get_media_recommendations

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value='''
    {
      "songs": [{"title": "千本桜", "artist": "黒うさP"}],
      "anime": [{"title": "進撃の巨人", "scene": "第1話 誓いのシーン"}],
      "articles": [{"title": "て形の使い方", "keywords": "JLPT N4 て形 文法"}]
    }
    ''')

    result = await get_media_recommendations("て-form", mock_provider)
    assert len(result["songs"]) == 1
    assert result["songs"][0]["title"] == "千本桜"
    assert len(result["anime"]) == 1
    assert len(result["articles"]) == 1


@pytest.mark.asyncio
async def test_get_media_recommendations_handles_bad_json():
    from backend.services.media_recommendations import get_media_recommendations

    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(return_value="not json at all")

    result = await get_media_recommendations("て-form", mock_provider)
    assert result == {"songs": [], "anime": [], "articles": []}
