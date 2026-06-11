# Engineering decisions

A short writeup of the non-obvious choices — the part that separates "I followed
a RAG tutorial" from "I have shipped and debugged retrieval systems." Several of
these come from running a much larger Thai RAG assistant in production.

## 1. Paragraph-level reranking, not just top-k

Vector search returns whole chunks. Chunks are long and mix relevant sentences
with boilerplate, which **dilutes** the signal the LLM sees. Instead of feeding
the top-k chunks straight to the model, I split each retrieved chunk into
paragraphs, embed those, and re-score them against the query. The model then
receives a handful of focused passages.

*Effect:* fewer, sharper sources per answer and noticeably better citation
precision, at the cost of extra embedding calls per query — a trade-off worth
making for a quality-first assistant.

## 2. Citations are enforced, and "I don't know" is a valid answer

The system prompt forces every claim to carry a `[n]` citation and instructs the
model to **decline** when the retrieved sources don't support an answer. For a
retrieval system, confidently answering from parametric memory is the failure
mode — refusing is the correct behaviour, not a bug.

## 3. Score threshold gates the web-search fallback

Web search is expensive and noisy, so it should not fire on every query. The
pipeline only falls back to the web when the **top retrieval score is below a
threshold** — i.e. when the local corpus clearly doesn't cover the question.
This keeps cost and latency predictable and the answer grounded whenever possible.

## 4. API embeddings instead of a local model

Running `sentence-transformers` means shipping `torch` (multiple GB) and the
model weights. By calling an OpenAI-compatible embedding endpoint (bge-m3 via
OpenRouter — the **same key and provider as the LLM**), the app stays tiny,
installs in seconds, needs only one credential, and behaves **identically** on a
laptop and on a hosted Space. The trade-off — a network call per embedding — is
acceptable for this workload and removes a whole class of environment/disk problems.

## 5. Local vs Cloud vector store behind one switch

The same `Store` class talks to a local file-based Qdrant or to Qdrant Cloud,
selected purely by whether `QDRANT_URL` is set. Develop locally with zero
accounts; deploy on managed Qdrant by setting two env vars. No code changes.

## 6. Private by design

No logging, no analytics, no user counter. A shared password gates the hosted
demo. The goal of this project is to be **verifiable**, not to be a public
service — so it carries none of the data-handling or abuse surface of one.

## Known limits (honest)

- Reranking re-embeds paragraphs per query; for high traffic you'd cache or use a
  dedicated cross-encoder reranker.
- The LLM-as-judge faithfulness check is a coarse signal, not a substitute for a
  labelled eval set on a real corpus.
- Local Qdrant file mode is single-process; concurrent writers need the server
  or Cloud.

These are deliberate scope cuts for a demo, not oversights — the production
version handles them differently.
