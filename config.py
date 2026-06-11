"""Central configuration. All values come from environment / .env so the same
code runs identically on a laptop and on a Hugging Face Space."""
import os
from dotenv import load_dotenv

load_dotenv()

# LLM (OpenRouter, OpenAI-compatible)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash")

# Embeddings (bge-m3 via OpenRouter — same key as the LLM; OpenAI-compatible, no local model)
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://openrouter.ai/api/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "baai/bge-m3")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))

# Vector store (Qdrant: local file by default, Cloud if QDRANT_URL set)
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_local")
COLLECTION = os.getenv("COLLECTION", "crafting")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "12"))
RERANK_TOP = int(os.getenv("RERANK_TOP", "4"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# App
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
WEB_SEARCH = os.getenv("WEB_SEARCH", "true").lower() == "true"  # default ON (DuckDuckGo, no key)
# Domain allow-list for the web-search fallback (the "narrow-domain loop").
# Keeps answers on-source and is a prompt-injection containment boundary.
INCLUDE_DOMAINS = [d.strip() for d in os.getenv("INCLUDE_DOMAINS", "minecraft.wiki").split(",") if d.strip()]
