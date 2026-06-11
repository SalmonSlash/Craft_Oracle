"""Streamlit demo UI.

Design choices that match the project goals:
- No logging, no analytics, no user counter — nothing is stored.
- Optional password gate (APP_PASSWORD) so only people you share it with get in.
- A "Behind the scenes" panel exposes the retrieval pipeline (chunks + scores),
  turning the hidden complexity into the thing recruiters actually see.
"""
import hmac
import os
import time

import streamlit as st
import streamlit.components.v1 as components

import config
from rag.pipeline import answer_stream

REFUSAL_MARKERS = ("ไม่มีข้อมูล", "ไม่พบ", "don't have", "do not have",
                   "not enough", "no relevant", "ไม่เพียงพอ")


def is_refusal(text):
    t = text.lower()
    return any(m.lower() in t for m in REFUSAL_MARKERS)


def grid_html(source):
    """Render a corpus recipe's crafting grid as a clean square HTML table
    (deterministic — LLMs are unreliable at copying spatial grids). Empty cells
    stay empty; the grid is padded to a square so it reads like a real crafting
    table (e.g. a 2-row recipe shows as 3x3 with the bottom row empty)."""
    path = os.path.join("data", "recipes", source)
    if not os.path.exists(path):
        return None
    lines = open(path, encoding="utf-8").read().splitlines()
    idx = next((i for i, l in enumerate(lines) if l.startswith("Grid (rows")), None)
    if idx is None:
        return None
    rows = []
    for l in lines[idx + 1:]:
        if not l.strip():
            break
        rows.append([("" if c.strip() == "(empty)" else c.strip()) for c in l.split("|")])
    rows = [r for r in rows if any(c for c in r)]  # drop fully-empty rows
    if not rows:
        return None
    n = max(max(len(r) for r in rows), len(rows))  # square side (usually 3)
    cell = ("border:1px solid #3a3f4b;width:44px;height:44px;text-align:center;"
            "vertical-align:middle;font-size:10px;line-height:1.1;padding:2px;"
            "color:#cfd3dc;")
    html = ["<table style='border-collapse:collapse;margin:4px 0'>"]
    for i in range(n):
        html.append("<tr>")
        for j in range(n):
            c = rows[i][j] if i < len(rows) and j < len(rows[i]) else ""
            bg = "#21262d" if c else "#161a20"
            html.append(f"<td style='{cell}background:{bg}'>{c}</td>")
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)

st.set_page_config(page_title="Craft Oracle", layout="centered")
st.markdown(
    "<style>"
    "#MainMenu{visibility:hidden;}footer{visibility:hidden;}"
    "[data-testid='stToolbar']{display:none;}[data-testid='stDecoration']{display:none;}"
    # de-emphasise the technical expander so it stops competing with the answer
    ".streamlit-expanderHeader,[data-testid='stExpander'] summary"
    "{font-size:12px;color:#6b7280;}"
    "</style>",
    unsafe_allow_html=True,
)


def kill_autocomplete():
    """Stop the browser from showing the previous-questions dropdown on the input."""
    components.html(
        "<script>"
        "const f=()=>window.parent.document.querySelectorAll('input')"
        ".forEach(i=>{i.setAttribute('autocomplete','off');"
        "i.setAttribute('autocorrect','off');i.setAttribute('spellcheck','false');});"
        "f();setTimeout(f,300);setTimeout(f,1000);"
        "</script>",
        height=0,
    )


def gate():
    """Simple shared-password gate. Empty APP_PASSWORD = open (local dev)."""
    if not config.APP_PASSWORD:
        return
    if st.session_state.get("authed"):
        return
    pw = st.text_input("Access password", type="password")
    if pw and hmac.compare_digest(pw, config.APP_PASSWORD):
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Wrong password")
    st.stop()


gate()

st.title("Craft Oracle")
st.markdown(
    "<p style='font-size:18px;line-height:1.5;color:#e6e8ec;margin:-6px 0 2px'>"
    "<b>Minecraft wiki assistant</b> — crafting, smelting, brewing, trading, mobs, "
    "and more, grounded in the Minecraft Wiki.</p>"
    "<p style='font-size:14px;color:#9aa0aa;margin:0 0 14px'>"
    "Thai &amp; English &middot; No data is stored.</p>",
    unsafe_allow_html=True,
)

q = st.text_input(
    "Your question",
    placeholder="เช่น: คราฟ furnace ยังไง? / brewing a potion of healing? / Creeper คืออะไร?",
)
kill_autocomplete()

if q:
    t0 = time.time()
    with st.spinner("Retrieving…"):
        meta, stream = answer_stream(q)

    st.markdown("### Answer")
    try:
        full = st.write_stream(stream)  # streams tokens live, returns the full text
        stream_ok = True
    except Exception:
        full, stream_ok = "", False
        st.error("ขออภัย ระบบขัดข้องชั่วคราว ลองส่งคำถามใหม่อีกครั้งครับ")
    latency = round(time.time() - t0, 2)

    # only show the grid + source when we actually answered from a corpus recipe
    answered = stream_ok and bool(full.strip()) and bool(meta["sources"]) and not is_refusal(full)
    if answered:
        gh = grid_html(meta["sources"][0]["source"])
        if gh:
            st.markdown("**Crafting grid**")
            st.markdown(gh, unsafe_allow_html=True)

        src = meta["sources"][0]["source"]
        if src.startswith("http"):
            url = src
            name = src.rstrip("/").split("/")[-1].replace("_", " ")
        else:
            base = src.replace(".md", "").replace("_smelting", "")
            url = "https://minecraft.wiki/w/" + base
            name = base.replace("_", " ")
        st.caption(f"ที่มา: [{name}]({url})")

    with st.expander("Behind the scenes"):
        st.caption(
            f"Latency {latency}s · web-search fallback: {meta['web']} · "
            f"passages after rerank: {len(meta['debug'])} · "
            "embed (bge-m3) -> Qdrant search -> paragraph rerank -> cited answer (LangChain LCEL)"
        )
        for i, h in enumerate(meta["debug"]):
            st.caption(f"[{i + 1}] cosine={round(h['score'], 3)} — {h['source']}: {h['text'][:240]}")
