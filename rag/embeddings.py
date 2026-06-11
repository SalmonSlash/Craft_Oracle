"""Embedding backend. Uses an OpenAI-compatible endpoint (DeepInfra bge-m3 by
default) so no model is downloaded locally — important on a low-disk machine."""
from openai import OpenAI
import config

_client = OpenAI(base_url=config.EMBED_BASE_URL, api_key=config.EMBED_API_KEY or "missing")


BATCH = 100  # cap inputs per request (web fallback can produce hundreds of paragraphs)


def embed_texts(texts):
    """Embed a list of strings -> list of vectors, in input order."""
    if not texts:
        return []
    out = []
    for i in range(0, len(texts), BATCH):
        resp = _client.embeddings.create(model=config.EMBED_MODEL, input=texts[i : i + BATCH])
        # the API guarantees order via `index`, not the position in `data`
        out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return out


def embed_query(text):
    return embed_texts([text])[0]
