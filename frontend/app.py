"""Streamlit front end: compare the three strategies, ask questions, read the benchmark results."""

import os
from pathlib import Path

import httpx
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
RESULTS = Path(os.environ.get("RESULTS_DIR", "results"))
STRATEGIES = ["vector", "hybrid", "hybrid_rerank"]
LABELS = {"vector": "Vector only", "hybrid": "Hybrid (BM25 + vector, RRF)", "hybrid_rerank": "Hybrid + rerank"}


def post(path: str, payload: dict) -> dict:
    response = httpx.post(f"{BACKEND_URL}{path}", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def show_timings(body: dict) -> None:
    parts = [f"{name} {ms:.0f} ms" for name, ms in body["timings_ms"].items()]
    if body.get("tokens"):
        parts += [f"{name} tokens {count}" for name, count in body["tokens"].items()]
    st.caption(" | ".join(parts))


st.set_page_config(page_title="Clausewise — policy search", layout="wide")
st.title("Policy search: vector vs hybrid vs rerank")
compare_tab, ask_tab, bench_tab = st.tabs(["Compare", "Ask", "Benchmark"])

with compare_tab:
    query = st.text_input("Question", "Who must approve an expense of INR 20,000?")
    k = st.slider("Results per strategy", 1, 10, 5)
    if st.button("Search", type="primary") and query.strip():
        for column, strategy in zip(st.columns(3), STRATEGIES, strict=True):
            with column:
                st.subheader(LABELS[strategy])
                try:
                    body = post("/search", {"query": query, "strategy": strategy, "k": k})
                except httpx.HTTPError as exc:
                    st.error(f"Request failed: {exc}")
                    continue
                show_timings(body)
                for rank, hit in enumerate(body["hits"], start=1):
                    chunk = hit["chunk"]
                    st.markdown(
                        f"**{rank}. {chunk['chunk_id']}** | score {hit['score']:.3f}  \n"
                        f"dense #{hit['dense_rank'] or '-'} | bm25 #{hit['bm25_rank'] or '-'}"
                    )
                    st.caption(chunk["text"])

with ask_tab:
    question = st.text_input("Question", "How quickly must a 500 litre diesel spill be reported to the regulator?", key="ask")
    strategy = st.radio("Retrieval", STRATEGIES, index=2, format_func=LABELS.get, horizontal=True)
    if st.button("Ask", type="primary") and question.strip():
        try:
            with st.spinner("Waiting for the model..."):
                body = post("/ask", {"query": question, "strategy": strategy})
        except httpx.HTTPError as exc:
            st.error(f"Request failed: {exc}")
        else:
            show = {"answered": st.success, "conflict": st.warning, "refused": st.info}.get(body["status"], st.error)
            show(f"Status: {body['status']}")
            st.write(body["answer"])
            st.markdown("**Citations:** " + (", ".join(body["citations"]) or "none"))
            for note in body.get("guardrail_notes", []):
                st.caption(f"Guardrail: {note}")
            show_timings(body)
            with st.expander("Evidence given to the model"):
                for hit in body["hits"]:
                    st.markdown(f"**{hit['chunk']['chunk_id']}** {hit['chunk']['text']}")

with bench_tab:
    report = RESULTS / "benchmark.md"
    if not report.exists():
        st.info("No results yet. Run the evaluation first: `docker compose run --rm evaluation`.")
    else:
        if (RESULTS / "pareto.png").exists():
            st.image(str(RESULTS / "pareto.png"), width=520)
        st.markdown(report.read_text(encoding="utf-8"))
        if (RESULTS / "robustness.md").exists():
            st.markdown((RESULTS / "robustness.md").read_text(encoding="utf-8"))
