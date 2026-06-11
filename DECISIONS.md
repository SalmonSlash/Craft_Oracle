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

## Hard problems & design calls

### Cross-lingual retrieval without translating the data

A Thai query like "ดาบเหล็ก" (iron sword) kept retrieving *Diamond Sword* — close
in embedding space, wrong answer. The tempting fix was to translate the whole
corpus to Thai or tag every document with Thai labels. I rejected that: it
doubles the corpus, drifts out of sync, and still leaves the matching fuzzy.
Instead the query is translated to the canonical English name and searched **two
ways at once** — the translated query *and* the raw Thai (bge-m3 is multilingual,
so the raw query is a safety net when translation drops a word) — then a
**whole-word** name check force-promotes an exact item match to the top source.

*Trade-off:* one extra translation call per non-English query, in exchange for the
data staying single-language and the disambiguation being explicit rather than
hoped-for.

### Knowing when not to build the router

The obvious next step looked like an intent router: classify each question as
greeting / recipe / other and branch on it. I prototyped it as rules and it fell
apart immediately — the length and keyword thresholds were magic numbers that
don't generalize (a stretched "หวัดดีฮาฟฟฟ" breaks a greeting check; "bedrock"
contains "bed"), and the only robust version was just another LLM call. The
deeper "proper" fix — resolving the item name straight to a file — was real but
solved a problem the system already handled.

*Decision:* cut both. Don't pre-classify intent from text at all; let retrieval
scores and the answer itself decide. The cheapest correct system is the one that
doesn't add a layer it can't make reliable.

### A wiki assistant, not a pretty dataset

Early on this was a curated recipe RAG — clean, narrow, easy to score. But a
"Minecraft wiki assistant" that only knows pre-baked recipes isn't a wiki. The
call was to keep an **exact corpus fast-path** for recipes (deterministic,
checkable, instant) and route everything else — brewing, mobs, lore, "can I
upgrade this?" — to a **live, domain-restricted read of minecraft.wiki**.

*Trade-off:* recipe answers stay exact and fast; the long tail is slower and only
as good as the page, but the assistant answers like the wiki instead of refusing
whatever wasn't pre-loaded.

### Streaming under a reasoning model

DeepSeek-V4-Flash was clearly smarter and cheaper than the baseline, but it's a
reasoning model: with streaming, the first visible token only arrives *after* all
the hidden thinking finishes — 5–15s of dead air, and wildly variable. The cause
turned out to be the gateway randomly routing to providers that still "think"
even with reasoning disabled. The fix was two flags: suppress the reasoning trace
and **pin a latency-sorted provider**, which consistently lands on one that
honours the request — dropping first-token to ~1s.

*Lesson:* the model wasn't the bottleneck, the routing was. Measure where the time
actually goes before swapping the model.

### Measuring honestly

Two small choices kept the evaluation trustworthy. First, a case that times out
or rate-limits is **not** counted as a wrong answer — infra errors are tracked
separately and excluded from the quality score, so a flaky provider can't quietly
tank the pass-rate. Second, every prompt change is re-run against the **boundary
cases**, not just the case it was meant to fix: tightening the prompt so it stops
hedging on colloquial item names also nudged the "is this craftable?" answers,
which only surfaced because the out-of-domain and non-craftable cases were
checked before shipping. A number you can't trust is worse than no number.

## Known limits (honest)

- Reranking re-embeds paragraphs per query; for high traffic you'd cache or use a
  dedicated cross-encoder reranker.
- The LLM-as-judge faithfulness check is a coarse signal, not a substitute for a
  labelled eval set on a real corpus.
- Local Qdrant file mode is single-process; concurrent writers need the server
  or Cloud.

These are deliberate scope cuts for a demo, not oversights — the production
version handles them differently.
