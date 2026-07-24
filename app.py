import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from knowledge_assistant.pipeline import ask, build_store, ingest
from knowledge_assistant.rag import get_generator

load_dotenv()
st.set_page_config(page_title="Knowledge Assistant", layout="wide")
st.title("Knowledge Assistant (RAG)")

with st.sidebar:
    st.subheader("Konfiguration")
    emb = st.selectbox("Embedding", ["hashing", "default", "openai"],
                       help="hashing = offline ohne Download, nur zum Testen. "
                            "default = all-MiniLM-L6-v2, laedt beim ersten Lauf.")
    llm = st.selectbox("Generator", ["extractive", "anthropic", "openai"],
                       help="extractive = offline, zitiert woertlich")
    key = ""
    model = None
    if llm != "extractive":
        default_model = "claude-sonnet-4-5" if llm == "anthropic" else "gpt-4o-mini"
        model = st.text_input("Modell", default_model)
        env_key = "ANTHROPIC_API_KEY" if llm == "anthropic" else "OPENAI_API_KEY"
        key = st.text_input("API Key", type="password", value=os.getenv(env_key, ""))
    mode = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"],
                        help="hybrid = Dense + BM25 via RRF")
    k = st.slider("Top-K Chunks", 1, 10, 5)

try:
    store = build_store(emb, "chroma_db")
except Exception as e:
    st.error(f"Embedding-Backend nicht verfuegbar: {e}")
    st.stop()

with st.sidebar:
    st.divider()
    st.metric("Chunks im Index", store.count())
    for s in store.sources():
        st.caption(f"- {s}")
    if st.button("Index leeren"):
        store.reset()
        st.rerun()

tab_ask, tab_ingest = st.tabs(["Fragen", "Dokumente"])

with tab_ingest:
    files = st.file_uploader("PDF / TXT / MD", type=["pdf", "txt", "md"],
                             accept_multiple_files=True)
    if files and st.button("Indexieren", type="primary"):
        with tempfile.TemporaryDirectory() as td:
            for f in files:
                (Path(td) / f.name).write_bytes(f.getvalue())
            with st.spinner("Chunking und Embedding..."):
                r = ingest(td, store)
        st.success(f"{r.files} Datei(en), {r.pages} Seiten, {r.chunks} Chunks "
                   f"({r.duration_s:.1f}s)")
        for e in r.errors:
            st.warning(e)

with tab_ask:
    srcs = store.sources()
    src = st.selectbox("Quelle einschraenken", ["(alle)"] + srcs)
    q = st.text_input("Frage", placeholder="Was ist das Schluesselenzym des Calvin-Zyklus?")

    if q and st.button("Fragen", type="primary"):
        if store.count() == 0:
            st.warning("Index ist leer. Erst Dokumente hochladen.")
            st.stop()
        kwargs = {}
        if llm != "extractive":
            if not key:
                st.error("API Key fehlt.")
                st.stop()
            kwargs = {"model": model, "api_key": key}
        try:
            gen = get_generator(llm, **kwargs)
        except Exception as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Suche und Antwort..."):
            a = ask(q, store, gen, k=k, mode=mode,
                    source=None if src == "(alle)" else src)

        st.markdown(a.text)
        if not a.grounded:
            st.error("Antwort ist nicht sauber belegt.")
        for w in a.warnings:
            st.warning(w)

        if a.sentence_reports:
            st.caption("Satzweise Belegpruefung")
            for r in a.sentence_reports:
                icon = "OK" if r["ok"] else "UNGEDECKT"
                st.markdown(f"`{icon}` support={r['support']:.2f} "
                            f"cites={r['cites']} - {r['sentence'][:120]}")

        st.divider()
        st.caption("Abgerufene Kontextbloecke")
        for i, h in enumerate(a.hits, 1):
            used = "zitiert" if i in a.cited else "nicht zitiert"
            with st.expander(f"[{i}] {h.source} - Seite {h.page} - "
                             f"score {h.score} - {used}"):
                st.text(h.text)
