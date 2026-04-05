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

    # Try direct parse first, then regex fallback
    def _extract(text: str) -> dict | None:
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
        return None

    data = _extract(raw)
    if data is None:
        return {"songs": [], "anime": [], "articles": []}
    return {
        "songs":    data.get("songs", []),
        "anime":    data.get("anime", []),
        "articles": data.get("articles", []),
    }
