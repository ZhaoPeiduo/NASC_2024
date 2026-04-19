import random

from backend.services.llm.base import LLMProvider
from backend.services.recommendations import search_youtube, Video

TUTORIAL_PROMPT = """You are a JLPT grammar expert.
A student answered a Japanese grammar question incorrectly.
Generate a focused YouTube search query (in English) to find a tutorial explaining the specific grammar rule they missed.

Question: {question}
Correct answer: {correct_answer}
Student's answer: {user_answer}
Grammar concepts: {concepts}

Return ONLY the search query (3-8 words). Example: "JLPT N4 te-form permission grammar"
No explanation. No quotes. Just the query."""

# Curated grammar-themed song queries for random discovery
SONG_QUERIES = [
    "日本語 JLPT Japanese learning song grammar",
    "Japanese pop song everyday conversation 日常",
    "anime opening Japanese language N4 N5",
    "J-pop Japanese song 2024 popular",
    "YOASOBI Official Music Video Japanese",
    "Kenshi Yonezu Japanese music popular",
    "Japanese city pop classics 70s 80s",
    "anime ending theme Japanese language",
    "Japanese traditional folk song grammar story",
    "J-rock popular Japanese band song",
]


async def generate_tutorial_query(
    question: str,
    correct_answer: str,
    user_answer: str,
    concepts: list[str],
    provider: LLMProvider,
) -> str:
    """Ask the LLM to produce a focused YouTube search query for this wrong answer."""
    prompt = TUTORIAL_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        user_answer=user_answer,
        concepts=", ".join(concepts) if concepts else "unknown",
    )
    raw = await provider.complete(prompt)
    query = raw.strip().strip('"').strip("'").splitlines()[0].strip()
    # Fallback: use first concept if LLM returns something suspicious
    if len(query) > 120 or not query or query.endswith(":") or query.lower().startswith("here"):
        fallback = concepts[0] if concepts else "JLPT grammar Japanese"
        query = f"JLPT {fallback} grammar explanation"
    return query


async def search_tutorials_for_wrong_answers(
    wrong_items: list[dict],
    api_key: str,
    provider: LLMProvider,
    max_per_item: int = 2,
    max_items: int = 3,
) -> list[dict]:
    """
    For up to `max_items` unique concepts in `wrong_items`:
      1. Generate a targeted YouTube search query via LLM.
      2. Call YouTube API.
      3. Return structured results.

    Deduplicates by first concept to conserve YouTube API quota.
    """
    seen_concepts: set[str] = set()
    results: list[dict] = []

    for item in wrong_items:
        if len(results) >= max_items:
            break
        concept_key = item["concepts"][0] if item.get("concepts") else item["correct_answer"]
        if concept_key in seen_concepts:
            continue
        seen_concepts.add(concept_key)

        query = await generate_tutorial_query(
            question=item["question"],
            correct_answer=item["correct_answer"],
            user_answer=item["user_answer"],
            concepts=item.get("concepts", []),
            provider=provider,
        )
        videos = await search_youtube(query, api_key, max_results=max_per_item)
        results.append({
            "question_snippet": item["question"][:80],
            "concepts": item.get("concepts", []),
            "search_query": query,
            "videos": videos,
        })

    return results


async def get_random_japanese_song(
    api_key: str,
    concept: str | None = None,
) -> Video | None:
    """
    Return a random Japanese song video from YouTube.
    If `concept` is given, bias toward grammar-relevant music.
    """
    if concept:
        query = f"Japanese song {concept} 日本語 lyrics"
    else:
        query = random.choice(SONG_QUERIES)

    videos = await search_youtube(query, api_key, max_results=10)
    if not videos:
        return None
    return random.choice(videos)
