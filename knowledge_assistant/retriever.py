"""Hybrid-Retrieval: Dense (ChromaDB) + BM25, fusioniert per RRF.

Der BM25-Index wird beim ersten Zugriff aus der Collection gebaut und im
Speicher gehalten. Fuer die Groessenordnung dieses Projekts (bis ~50k Chunks)
ist das unproblematisch; darueber gehoert der Index persistiert oder in eine
Suchmaschine ausgelagert. Diese Grenze ist bewusst und dokumentiert.
"""
from typing import List, Optional

from .bm25 import BM25Index, reciprocal_rank_fusion
from .store import Hit, VectorStore

RRF_K = 60
CANDIDATE_MULTIPLIER = 4   # pro Kanal mehr Kandidaten holen als am Ende gebraucht


class HybridRetriever:
    def __init__(self, store: VectorStore, rrf_k: int = RRF_K):
        self.store = store
        self.rrf_k = rrf_k
        self._bm25: Optional[BM25Index] = None
        self._built_for = -1

    def _ensure_bm25(self):
        count = self.store.count()
        if self._bm25 is not None and self._built_for == count:
            return
        got = self.store.collection.get(include=["documents"])
        pairs = list(zip(got["ids"], got["documents"]))
        self._bm25 = BM25Index().build(pairs)
        self._built_for = count

    def search(self, query: str, k: int = 5, source: Optional[str] = None,
               mode: str = "hybrid") -> List[Hit]:
        if self.store.count() == 0:
            return []

        n_cand = min(k * CANDIDATE_MULTIPLIER, self.store.count())
        dense = self.store.search(query, k=n_cand, source=source)

        if mode == "dense":
            return dense[:k]

        self._ensure_bm25()
        lexical = self._bm25.search(query, k=n_cand)

        if mode == "bm25":
            ids = [cid for cid, _ in lexical[:k]]
            return self._hydrate(ids, dense, source)

        fused = reciprocal_rank_fusion(
            [[h.chunk_id for h in dense], [cid for cid, _ in lexical]],
            k=self.rrf_k,
        )
        ordered = sorted(fused.items(), key=lambda x: -x[1])
        hits = self._hydrate([cid for cid, _ in ordered], dense, source, limit=k)

        # RRF liefert eine Reihenfolge, aber KEIN absolutes Relevanzmass: der
        # beste Treffer hat per Konstruktion immer den hoechsten RRF-Wert, auch
        # wenn er inhaltlich nichts mit der Frage zu tun hat. Wuerde man darauf
        # schwellwerten, waere die Ablehnung irrelevanter Fragen unmoeglich.
        # Deshalb bleibt score die absolute Dense-Aehnlichkeit; die RRF-Position
        # steckt in rrf_score und bestimmt nur die Reihenfolge.
        dense_by_id = {h.chunk_id: h.score for h in dense}
        rrf_by_id = dict(ordered)
        for h in hits:
            h.rrf_score = round(rrf_by_id.get(h.chunk_id, 0.0), 5)
            h.score = dense_by_id.get(h.chunk_id, h.score)
        return hits

    def _hydrate(self, ids: List[str], dense: List[Hit],
                 source: Optional[str], limit: int = None) -> List[Hit]:
        """Chunk-IDs zu Hit-Objekten aufloesen, Metadaten aus der Collection."""
        known = {h.chunk_id: h for h in dense}
        missing = [i for i in ids if i not in known]

        if missing:
            got = self.store.collection.get(ids=missing,
                                            include=["documents", "metadatas"])
            for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
                known[cid] = Hit(cid, doc, meta.get("source", "?"),
                                 int(meta.get("page", 0)), 0.0)

        out = []
        for cid in ids:
            h = known.get(cid)
            if h is None:
                continue
            if source and h.source != source:
                continue
            out.append(h)
            if limit and len(out) >= limit:
                break
        return out
