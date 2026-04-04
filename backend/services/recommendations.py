from dataclasses import dataclass
import httpx


@dataclass
class Video:
    video_id: str
    title: str
    thumbnail_url: str
    channel_title: str


YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


async def search_youtube(concept: str, api_key: str, max_results: int = 3) -> list[Video]:
    if not api_key:
        return []
    params = {
        "part": "snippet",
        "q": f"JLPT {concept} grammar Japanese explanation",
        "type": "video",
        "videoDuration": "short",
        "maxResults": max_results,
        "key": api_key,
        "order": "relevance",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(YOUTUBE_SEARCH_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    videos = []
    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId", "")
        if not video_id:
            continue
        videos.append(Video(
            video_id=video_id,
            title=snippet.get("title", ""),
            thumbnail_url=snippet.get("thumbnails", {}).get("medium", {}).get("url", ""),
            channel_title=snippet.get("channelTitle", ""),
        ))
    return videos
