"""Paragraph-level cosine reranking.

Retrieved chunks are often long and noisy. We split each chunk into paragraphs,
embed them, and re-score against the query — so the LLM gets focused passages
instead of diluted text. This is the single biggest precision win in the pipeline.
"""
import numpy as np
from rag.embeddings import embed_texts, embed_query


def _split_paragraphs(text):
    parts = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
    return parts or [text.strip()]


def _norm(m):
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


def paragraph_rerank(query, hits, top):
    """Re-rank retrieved hits at paragraph granularity. Returns top passages."""
    if not hits:
        return []
    qv = _norm(np.array(embed_query(query)))

    paragraphs, owners = [], []
    for i, h in enumerate(hits):
        for para in _split_paragraphs(h["text"]):
            paragraphs.append(para)
            owners.append(i)

    pv = _norm(np.array(embed_texts(paragraphs)))
    sims = pv @ qv

    ranked = sorted(zip(sims, paragraphs, owners), key=lambda x: -x[0])
    out, seen = [], set()
    for sim, para, idx in ranked:
        key = para  # dedup on the full paragraph (a 60-char prefix collapsed distinct rows)
        if key in seen:
            continue
        seen.add(key)
        out.append({"text": para, "source": hits[idx]["source"], "score": float(sim)})
        if len(out) >= top:
            break
    return out
