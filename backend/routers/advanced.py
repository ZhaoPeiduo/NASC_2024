from fastapi import APIRouter, Depends, HTTPException, Query

from backend.config import settings
from backend.schemas.api import (
    VideoRecommendation, GenerateRequest, GeneratedQuestionResponse,
    MediaRecommendRequest, MediaRecommendResponse, SongRec, AnimeRec, ArticleRec,
    WrongAnswerRecsRequest, WrongAnswerRecsResponse, VideoTutorialSet,
)
from backend.services.recommendations import search_youtube
from backend.services.media_recommendations import get_media_recommendations
from backend.services.llm.factory import get_llm_provider
from backend.services.youtube_query import (
    search_tutorials_for_wrong_answers,
    get_random_japanese_song,
)
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


@router.post("/recommendations/for-wrong-answers", response_model=WrongAnswerRecsResponse)
async def recs_for_wrong_answers(
    body: WrongAnswerRecsRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Given a list of wrong quiz answers, return YouTube tutorial videos
    for up to 3 unique grammar concepts found in those answers.
    Returns an empty list when YOUTUBE_API_KEY is not configured.
    """
    if not settings.youtube_api_key:
        return WrongAnswerRecsResponse(recommendations=[])

    provider = get_llm_provider()
    raw_items = [
        {
            "question": item.question,
            "correct_answer": item.correct_answer,
            "user_answer": item.user_answer,
            "concepts": item.concepts,
        }
        for item in body.wrong_items
    ]
    results = await search_tutorials_for_wrong_answers(
        wrong_items=raw_items,
        api_key=settings.youtube_api_key,
        provider=provider,
    )
    recommendations = [
        VideoTutorialSet(
            question_snippet=r["question_snippet"],
            concepts=r["concepts"],
            search_query=r["search_query"],
            videos=[
                VideoRecommendation(
                    title=v.title,
                    video_id=v.video_id,
                    thumbnail_url=v.thumbnail_url,
                    channel_title=v.channel_title,
                )
                for v in r["videos"]
            ],
        )
        for r in results
    ]
    return WrongAnswerRecsResponse(recommendations=recommendations)


@router.get("/youtube/random-song", response_model=VideoRecommendation)
async def random_japanese_song(
    concept: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """
    Return a random Japanese song video from YouTube.
    Optional `concept` query param biases toward grammar-relevant music.
    Returns 404 when YouTube API key is not configured or no results found.
    """
    if not settings.youtube_api_key:
        raise HTTPException(status_code=404, detail="YouTube API not configured")

    video = await get_random_japanese_song(
        api_key=settings.youtube_api_key,
        concept=concept,
    )
    if video is None:
        raise HTTPException(status_code=404, detail="No video found")

    return VideoRecommendation(
        title=video.title,
        video_id=video.video_id,
        thumbnail_url=video.thumbnail_url,
        channel_title=video.channel_title,
    )
