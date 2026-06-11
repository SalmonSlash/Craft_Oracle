"""End-to-end RAG pipeline.

Flow:  embed query -> Qdrant search -> (web fallback if weak) -> paragraph
rerank -> build cited context -> LLM answer with citations (LangChain LCEL).

The generation step is a LangChain Expression Language (LCEL) chain:
    prompt | ChatOpenAI | StrOutputParser
"""
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from rag.embeddings import embed_query
from rag.store import Store
from rag.rerank import paragraph_rerank
from rag.web_search import web_search
from openai import OpenAI

_tr = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def _reasoning_kwargs():
    """Reasoning-capable models (DeepSeek V4/R1) otherwise block the first token
    while they think. We suppress the reasoning trace (`exclude`) AND pin a
    latency-sorted provider — OpenRouter's default routing randomly lands on a
    provider that still thinks for ~12s; latency-sort consistently picks one that
    honours the no-reasoning flag, giving ~1s first-token. No-op for plain chat
    models (e.g. deepseek-chat) which don't reason."""
    m = config.LLM_MODEL.lower()
    reasoning_model = ("r1" in m or "reason" in m
                       or ("deepseek" in m and "chat" not in m))
    if reasoning_model:
        return {"reasoning": {"exclude": True}, "provider": {"sort": "latency"}}
    return {}


_TRANSLATE_PROMPT = (
    "You are a Minecraft expert translator. The user asks about a Minecraft item, "
    "block, mob, or mechanic in another language. Identify what it refers to and "
    "rewrite the question in English using the OFFICIAL Minecraft Wiki name. "
    "IMPORTANT: any English word already in the question is almost certainly the "
    "item name (e.g. Target, Loom, Bell) — keep it EXACTLY, never replace it. "
    "Thai colloquial names map like: เครื่องปรุงยา/เครื่องต้มยา=Brewing Stand, "
    "เตาหลอม/เตาเผา=Furnace, โต๊ะคราฟ=Crafting Table, คบไฟ=Torch, "
    "หีบ/กล่อง=Chest, ดาบเหล็ก=Iron Sword. Return ONLY the English question."
)


def _translate(q):
    """Translate a non-English question to English for cross-lingual retrieval.

    Maps colloquial/local names to the official Minecraft Wiki item name so the
    embedding query lands on the right recipe (e.g. เครื่องปรุงยา -> Brewing Stand,
    not "potion")."""
    if all(ord(c) < 128 for c in q):
        return q
    try:
        r = _tr.chat.completions.create(
            model=config.LLM_MODEL, temperature=0, extra_body=_reasoning_kwargs(),
            messages=[{"role": "user", "content": _TRANSLATE_PROMPT + "\n\n" + q}],
        )
        return (r.choices[0].message.content or "").strip() or q
    except Exception:
        return q


def _kw_score(source, query):
    """Fraction of the item-name's words that appear as WHOLE words in the query.
    (Substring matching would treat 'bed' as present in 'bedrock' and wrongly
    force Bed to source [1] for a Bedrock question.)"""
    if source.startswith("http"):
        return 0.0
    name = source.replace(".md", "").replace("_smelting", "").replace("_", " ").lower()
    words = name.split()
    qwords = set(re.findall(r"[a-z0-9]+", query.lower()))
    return sum(1 for w in words if w in qwords) / max(len(words), 1)

SYSTEM = (
    "You are a Minecraft wiki assistant. Answer questions about Minecraft — crafting, "
    "smelting, brewing/potions, trading, mobs, items, and game mechanics — using ONLY "
    "the provided sources. Cite claims with [n] referring to the source number. "
    "Give item/ingredient names in both English and Thai, e.g. 'Cobblestone (หินก้อนกรวด)'. "
    "Format the answer with Markdown — tables for ingredient lists or brewing/trading "
    "info, bullet points for steps, short bold headings where helpful. Do NOT draw the "
    "3x3 crafting-grid layout yourself (it is shown separately); just give the ingredients "
    "with their counts. "
    "The user may use a colloquial name or a different language than the sources — "
    "match it to the item(s) in the sources and answer about them directly; do not "
    "refuse only because the wording differs. "
    "If the sources don't contain the answer, say you don't have enough information "
    "instead of guessing. Reply in the same language as the question."
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM),
        ("human", "Question: {question}\n\nSources:\n{context}\n\nAnswer with citations:"),
    ]
)

_chain = None
_store = None


def _get_chain():
    global _chain
    if _chain is None:
        llm = ChatOpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL,
            temperature=0.2,
            timeout=60,
            extra_body=_reasoning_kwargs() or None,
        )
        _chain = _prompt | llm | StrOutputParser()
    return _chain


def _get_store():
    global _store
    if _store is None:
        _store = Store()
        _store.ensure()
    return _store


def _retrieve(question):
    """Shared retrieval: translate -> vector search (translated + raw, merged) ->
    web fallback if weak -> paragraph rerank -> exact-word override. Returns
    (reranked_passages, used_web)."""
    en = _translate(question)  # translate to English for cross-lingual retrieval
    hits = _get_store().search(embed_query(en), config.TOP_K)

    # also retrieve on the RAW question (bge-m3 is multilingual) and merge — a
    # safety net for when translation drops/misreads an item name (e.g. 'Target'),
    # keeping the best score per source.
    if en != question:
        merged = {}
        for h in hits + _get_store().search(embed_query(question), config.TOP_K):
            if h["source"] not in merged or h["score"] > merged[h["source"]]["score"]:
                merged[h["source"]] = h
        hits = sorted(merged.values(), key=lambda h: h["score"], reverse=True)[: config.TOP_K]

    used_web = False
    top_score = hits[0]["score"] if hits else 0.0
    if not hits or top_score < config.SCORE_THRESHOLD:
        web = web_search(en)
        if web:
            hits = hits + web
            used_web = True

    reranked = paragraph_rerank(en, hits, config.RERANK_TOP)

    # exact word match: if a retrieved item's name is fully present in the query,
    # force it to source [1] (e.g. 'ดาบเหล็ก' -> Iron Sword, not Diamond Sword)
    best = max(hits, key=lambda h: _kw_score(h["source"], en), default=None)
    if best and _kw_score(best["source"], en) >= 0.99:
        if not reranked or reranked[0]["source"] != best["source"]:
            # promote the exact-name match to source [1], reusing the focused
            # paragraph the reranker already picked for it (fall back to the raw
            # chunk only if that source has no reranked passage yet)
            promoted = next((p for p in reranked if p["source"] == best["source"]), None)
            if promoted is None:
                promoted = {"text": best["text"], "source": best["source"], "score": 1.0}
            reranked = [promoted] + [p for p in reranked if p["source"] != best["source"]]
            reranked = reranked[: config.RERANK_TOP]

    return reranked, used_web


def _build_context(reranked):
    return "\n\n".join(
        f"[{i + 1}] ({h['source']}) {h['text']}" for i, h in enumerate(reranked)
    )


def _meta(reranked, used_web):
    return {
        "sources": [
            {"n": i + 1, "source": h["source"], "score": round(h["score"], 3)}
            for i, h in enumerate(reranked)
        ],
        "debug": reranked,
        "web": used_web,
    }


_NO_SOURCES = "ไม่พบข้อมูลที่เกี่ยวข้องในแหล่งอ้างอิง (no relevant sources found)."


def answer(question):
    """Non-streaming answer (used by the eval harness)."""
    t0 = time.time()
    reranked, used_web = _retrieve(question)
    if not reranked:
        ans = _NO_SOURCES
    else:
        ans = _get_chain().invoke(
            {"question": question, "context": _build_context(reranked)}
        )
    out = _meta(reranked, used_web)
    out["answer"] = ans
    out["latency"] = round(time.time() - t0, 2)
    return out


def answer_stream(question):
    """Streaming answer for the UI. Returns (meta, token_generator); the generator
    yields answer-text chunks so the UI can render them live (st.write_stream)."""
    reranked, used_web = _retrieve(question)
    meta = _meta(reranked, used_web)
    if not reranked:
        return meta, iter([_NO_SOURCES])
    gen = _get_chain().stream(
        {"question": question, "context": _build_context(reranked)}
    )
    return meta, gen
