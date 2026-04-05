import json
import re

from backend.services.llm.base import LLMProvider

MEDIA_PROMPT = """You are a JLPT study resource curator.
For the Japanese grammar point '{concept}', suggest real, famous Japanese media that naturally uses this grammar.

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "songs": [{{"title": "...", "artist": "..."}}],
  "anime": [{{"title": "...", "scene": "..."}}],
  "articles": [{{"title": "...", "keywords": "..."}}]
}}

Rules:
- Suggest 2 songs, 2 anime scenes, 2 articles
- Songs and anime must be real and famous
- Article keywords should be Japanese search terms useful for JLPT study
- All titles in Japanese where applicable"""


async def get_media_recommendations(concept: str, provider: LLMProvider) -> dict:
    prompt = MEDIA_PROMPT.format(concept=concept)
    raw = await provider.complete(prompt)

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"songs": [], "anime": [], "articles": []}
    try:
        data = json.loads(match.group(0))
        return {
            "songs":    data.get("songs", []),
            "anime":    data.get("anime", []),
            "articles": data.get("articles", []),
        }
    except json.JSONDecodeError:
        return {"songs": [], "anime": [], "articles": []}
