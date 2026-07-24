"""ChromaDB-Wrapper.

Embeddings werden ausserhalb berechnet und explizit uebergeben, damit das
Backend austauschbar bleibt und der Index weiss, womit er gebaut wurde.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import chromadb

from .chunking import Chunk
from .embeddings import EmbeddingBackend

DEFAULT_DIR = "chroma_db"
COLLECTION = "documents"


@dataclass
class Hit:
    chunk_id: str
    text: str
    source: str
    page: int
    score: float          # absolute Aehnlichkeit: 1.0 = identisch, 0.0 = unaehnlich
    rrf_score: float = 0.0   # nur im Hybrid-Modus gesetzt; bestimmt die Reihenfolge


class VectorStore:
    def __init__(self, backend: EmbeddingBackend, persist_dir: str = DEFAULT_DIR):
        self.backend = backend
        self.persist_dir = str(persist_dir)
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        # Name enthaelt das Backend: verhindert, dass Vektoren aus
        # unterschiedlichen Modellen in derselben Collection landen.
        self.collection = self.client.get_or_create_collection(
            name=f"{COLLECTION}_{backend.name}",
            metadata={"hnsw:space": "cosine", "backend": backend.name},
        )

    def add(self, chunks: List[Chunk], batch: int = 64) -> int:
        if not chunks:
            return 0
        added = 0
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            vecs = self.backend.embed([c.text for c in part])
            self.collection.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=vecs,
                documents=[c.text for c in part],
                metadatas=[c.metadata() for c in part],
            )
            added += len(part)
        return added

    def search(self, query: str, k: int = 5,
               source: Optional[str] = None) -> List[Hit]:
        if self.collection.count() == 0:
            return []
        qv = self.backend.embed([query])[0]
        where = {"source": source} if source else None
        res = self.collection.query(
            query_embeddings=[qv],
            n_results=min(k, self.collection.count()),
            where=where,
        )
        hits = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0],
            res["metadatas"][0], res["distances"][0]
        ):
            hits.append(Hit(cid, doc, meta.get("source", "?"),
                            int(meta.get("page", 0)),
                            round(max(0.0, 1.0 - dist), 4)))
        return hits

    def count(self) -> int:
        return self.collection.count()

    def sources(self) -> List[str]:
        if self.collection.count() == 0:
            return []
        got = self.collection.get(include=["metadatas"])
        return sorted({m.get("source", "?") for m in got["metadatas"]})

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=f"{COLLECTION}_{self.backend.name}",
            metadata={"hnsw:space": "cosine", "backend": self.backend.name},
        )
