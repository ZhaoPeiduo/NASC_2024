from fastapi import APIRouter, Depends, Query

from backend.config import settings
from backend.schemas.api import (
    VideoRecommendation, GenerateRequest, GeneratedQuestionResponse,
    MediaRecommendRequest, MediaRecommendResponse, SongRec, AnimeRec, ArticleRec,
)
from backend.services.recommendations import search_youtube
from backend.services.media_recommendations import get_media_recommendations
from backend.services.llm.factory import get_llm_provider
from backend.auth.dependencies import get_current_user
from backend.models.tables import User

router = APIRouter(prefix="/api/v1")


@router.get("/recommendations", response_model=list[VideoRecommendation])
async def get_recommendations(
    concepts: str = Query(..., description="Comma-separated concept list"),
    current_user: User = Depends(get_current_user),
):
    concept_list = [c.strip() for c in concepts.split(",") if c.strip()]
    if not concept_list:
        return []
    videos = await search_youtube(concept_list[0], settings.youtube_api_key)
    return [
        VideoRecommendation(
            title=v.title,
            video_id=v.video_id,
            thumbnail_url=v.thumbnail_url,
            channel_title=v.channel_title,
        )
        for v in videos
    ]


@router.post("/recommendations/media", response_model=MediaRecommendResponse)
async def media_recommendations(
    body: MediaRecommendRequest,
    current_user: User = Depends(get_current_user),
):
    provider = get_llm_provider()
    data = await get_media_recommendations(body.concept, provider)
    return MediaRecommendResponse(
        songs=[SongRec(**s) for s in data.get("songs", [])],
        anime=[AnimeRec(**a) for a in data.get("anime", [])],
        articles=[ArticleRec(**ar) for ar in data.get("articles", [])],
    )


@router.post("/generate", response_model=GeneratedQuestionResponse)
async def generate_question(
    body: GenerateRequest,
    current_user: User = Depends(get_current_user),
):
    provider = get_llm_provider()
    q = await provider.generate_question(concept=body.concept, level=body.level)
    return GeneratedQuestionResponse(
        question=q.question,
        options=q.options,
        correct_answer=q.correct_answer,
        explanation=q.explanation,
        concepts=q.concepts,
    )
