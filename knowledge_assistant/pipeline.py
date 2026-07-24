import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .chunking import chunk_pages
from .embeddings import get_backend, EmbeddingBackend
from .loader import load_file
from .rag import (Answer, Generator, build_context, check_grounding,
                  NO_ANSWER)
from .retriever import HybridRetriever
from .store import Hit, VectorStore

# Gemessen, nicht geraten: siehe calibrate.py und README-Tabelle.
# Bestes F1 aus Recall(beantwortbar) und Refusal(nicht beantwortbar) auf dem
# Beispiel-Fragenset. Der Wert ist backend-spezifisch und MUSS nach einem
# Wechsel des Embeddings neu kalibriert werden.
MIN_SCORE = 0.17


@dataclass
class IngestReport:
    files: int
    pages: int
    chunks: int
    errors: List[str]
    duration_s: float


def ingest(path, store: VectorStore, reset: bool = False,
           target: int = None, overlap: int = None) -> IngestReport:
    t0 = time.time()
    if reset:
        store.reset()

    p = Path(path)
    errors, pages, files = [], [], 0

    paths = sorted(p.iterdir()) if p.is_dir() else [p]
    for f in paths:
        if f.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        files += 1
        try:
            pages.extend(load_file(f))
        except Exception as e:
            errors.append(str(e))

    kw = {}
    if target:
        kw["target"] = target
    if overlap is not None:
        kw["overlap"] = overlap
    chunks = chunk_pages(pages, **kw)
    store.add(chunks)
    return IngestReport(files, len(pages), len(chunks), errors, time.time() - t0)


def retrieve(question: str, store: VectorStore, k: int = 5,
             source: Optional[str] = None, mode: str = "hybrid",
             retriever: HybridRetriever = None) -> List[Hit]:
    """Nur Retrieval, ohne Generierung. Wird von der Evaluation genutzt."""
    r = retriever or HybridRetriever(store)
    return r.search(question, k=k, source=source, mode=mode)


def ask(question: str, store: VectorStore, generator: Generator,
        k: int = 5, source: Optional[str] = None,
        min_score: float = MIN_SCORE, mode: str = "hybrid",
        retriever: HybridRetriever = None) -> Answer:
    hits = retrieve(question, store, k=k, source=source, mode=mode,
                    retriever=retriever)
    relevant = [h for h in hits if h.score >= min_score]

    if not relevant:
        warn = [] if hits else ["Index ist leer oder Filter ergab nichts."]
        if hits:
            warn.append(f"Kein Treffer ueber Schwellwert {min_score} "
                        f"(bester: {hits[0].score}).")
        return Answer(question, NO_ANSWER, hits, [], warn, True, [])

    context = build_context(relevant)
    try:
        text = generator.generate(question, context)
    except Exception as e:
        return Answer(question, f"Generierungsfehler: {e}", relevant, [],
                      [str(e)], False, [])

    cited, warnings, grounded, reports = check_grounding(text, relevant)
    return Answer(question, text, relevant, cited, warnings, grounded, reports)


def build_store(embedding: str = None, persist_dir: str = "chroma_db",
                **kw) -> VectorStore:
    backend: EmbeddingBackend = get_backend(embedding, **kw)
    return VectorStore(backend, persist_dir)
