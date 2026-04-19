import random
from dataclasses import dataclass

from TikTokApi import TikTokApi

TOPIC_HASHTAG_MAP: dict[str, str | None] = {
    "trending": None,
    "日本語": "日本語",
    "anime": "anime",
    "grammar": "日本語文法",
}


@dataclass
class TikTokVideoMeta:
    video_id: str
    title: str
    author: str


async def get_random_tiktok_video(
    topic: str,
    ms_token: str,
    count: int = 30,
) -> TikTokVideoMeta | None:
    """Fetch a random TikTok video matching `topic`. Returns None on empty results."""
    hashtag = TOPIC_HASHTAG_MAP.get(topic.lower(), None)
    videos: list[TikTokVideoMeta] = []

    async with TikTokApi() as api:
        tokens = [ms_token] if ms_token else []
        await api.create_sessions(
            ms_tokens=tokens,
            num_sessions=1,
            sleep_after=3,
            browser="chromium",
        )

        if hashtag is None:
            async for video in api.trending.videos(count=count):
                d = video.as_dict
                videos.append(TikTokVideoMeta(
                    video_id=str(d.get("id", "")),
                    title=d.get("desc", "")[:120],
                    author=d.get("author", {}).get("nickname", ""),
                ))
        else:
            async for video in api.hashtag(name=hashtag).videos(count=count):
                d = video.as_dict
                videos.append(TikTokVideoMeta(
                    video_id=str(d.get("id", "")),
                    title=d.get("desc", "")[:120],
                    author=d.get("author", {}).get("nickname", ""),
                ))

    if not videos:
        return None
    return random.choice(videos)


async def fetch_video_bytes(video_id: str, ms_token: str) -> bytes:
    """Download raw bytes for a TikTok video by ID."""
    async with TikTokApi() as api:
        tokens = [ms_token] if ms_token else []
        await api.create_sessions(
            ms_tokens=tokens,
            num_sessions=1,
            sleep_after=3,
            browser="chromium",
        )
        video = api.video(id=video_id)
        return await video.bytes()
