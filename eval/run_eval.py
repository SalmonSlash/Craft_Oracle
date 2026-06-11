"""Evaluation harness.

Runs a golden question set through the pipeline and reports keyword pass-rate,
LLM-as-judge faithfulness, and latency (avg / p90), with a normal-vs-edge breakdown.

golden.jsonl format (one JSON per line):
  {"question": "...", "must_contain": [...], "kind": "normal"|"edge", "note": "..."}
Each element of must_contain is either:
  - a string  -> must appear in the answer, or
  - a list     -> at least one of them must appear (alternatives, e.g. variants)

Usage:  python -m eval.run_eval     (from project root, after ingest)
"""
import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKERS = 12  # run cases in parallel (cap concurrency to avoid rate-limits)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")  # so Thai prints don't crash on Windows
except Exception:
    pass

import config
from rag.pipeline import answer, _reasoning_kwargs


def kw_ok(ans, must):
    a = ans.lower()
    for k in must:
        if isinstance(k, list):
            if not any(str(x).lower() in a for x in k):
                return False
        elif str(k).lower() not in a:
            return False
    return True


def percentile(vals, q):
    if not vals:
        return 0.0
    s = sorted(vals)
    return round(s[int(round((q / 100) * (len(s) - 1)))], 2)


def judge(question, ans):
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
            model=config.LLM_MODEL, temperature=0, timeout=60,
            extra_body=_reasoning_kwargs() or None,  # don't let the judge "think"
        )
        prompt = (
            f"Question: {question}\nAnswer: {ans}\n\n"
            "Is the answer non-evasive and consistent with a retrieval-grounded "
            "response? Reply with only 'yes' or 'no'."
        )
        return llm.invoke(prompt).content.strip().lower().startswith("y")
    except Exception:
        return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "golden.jsonl"), encoding="utf-8") as fh:
        cases = [json.loads(l) for l in fh if l.strip()]

    def run_case(c):
        try:
            r = answer(c["question"])
            ok = kw_ok(r["answer"], c.get("must_contain", []))
            f = judge(c["question"], r["answer"])
            return c, r, ok, f
        except Exception as e:
            # one case failing (timeout, rate-limit, 500) must not kill the run
            return c, {"answer": f"<error: {e}>", "latency": None}, False, None

    lats, rows = [], []
    agg = {"normal": [0, 0], "edge": [0, 0]}  # kind -> [passed, ran]
    passed = faithful = judged = errored = 0
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fut in as_completed(ex.submit(run_case, c) for c in cases):
            c, r, ok, f = fut.result()
            done += 1
            err = r["latency"] is None  # run_case marks an errored case with latency=None
            kind = c.get("kind", "normal")
            agg.setdefault(kind, [0, 0])
            if err:
                errored += 1  # infra error: kept out of the quality scores
            else:
                lats.append(r["latency"])
                passed += int(ok)
                agg[kind][0] += int(ok)
                agg[kind][1] += 1
                if f is not None:  # only count cases the judge actually rated
                    judged += 1
                    faithful += int(f)
            rows.append((kind, c["question"], ok, f, r["latency"], err))
            status = "ERR " if err else ("OK  " if ok else "FAIL")
            lat_s = "—" if err else f"{r['latency']}s"
            print(f"[{done}/{len(cases)}] {status} {lat_s}  {c['question'][:42]}",
                  file=sys.stderr, flush=True)

    n = len(cases)
    ran = n - errored
    md = [
        "# Eval Results\n",
        f"- Test cases: **{n}**" + (f"  (errored: **{errored}**, excluded from scores)" if errored else ""),
        f"- Keyword pass-rate: **{passed}/{ran} ({round(100 * passed / ran) if ran else 0}%)**",
        f"  - normal: **{agg['normal'][0]}/{agg['normal'][1]}**"
        + (f"  | edge: **{agg['edge'][0]}/{agg['edge'][1]}**" if agg.get("edge", [0, 0])[1] else ""),
        f"- Faithful (LLM-judge): **{faithful}/{judged}**"
        + (f"  ({ran - judged} judge-failed)" if judged < ran else ""),
        f"- Latency: avg **{round(sum(lats) / len(lats), 2) if lats else 0}s**, p90 **{percentile(lats, 90)}s**\n",
        "| Kind | Question | Keyword | Faithful | Latency |",
        "|---|---|---|---|---|",
    ]
    for kind, q, ok, f, l, err in rows:
        kw = "ERR" if err else ("pass" if ok else "FAIL")
        fa = "n/a" if (err or f is None) else ("yes" if f else "no")
        lat = "—" if err else f"{l}s"
        md.append(f"| {kind} | {q[:46]} | {kw} | {fa} | {lat} |")

    report = "\n".join(md)
    with open(os.path.join(here, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)


if __name__ == "__main__":
    main()
