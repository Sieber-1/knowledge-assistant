"""BM25 (Okapi) als lexikalischer Retrieval-Kanal.

Warum zusaetzlich zu Dense-Retrieval: Embeddings sind gut bei Umschreibungen und
Synonymen, aber schwach bei exakten Begriffen, Eigennamen, Zahlen und
Abkuerzungen. "RuBisCO" oder "Paragraph 14 Absatz 2" landen im Vektorraum nahe
bei allem moeglichen; BM25 trifft sie exakt.

Bewusst ohne externe Abhaengigkeit implementiert (kein rank_bm25), damit die
Formel im Code sichtbar und pruefbar bleibt.

Formel (Okapi BM25):
    score(q,d) = sum_t IDF(t) * (f(t,d) * (k1+1)) /
                 (f(t,d) + k1 * (1 - b + b * |d|/avgdl))
    IDF(t)     = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

k1 steuert die Saettigung der Termfrequenz (hoeher = Wiederholung zaehlt mehr),
b die Laengennormalisierung (1.0 = voll, 0.0 = keine).
"""
import math
from collections import Counter
from typing import Dict, List, Tuple

from .embeddings import tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    def __init__(self, k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.tfs: List[Counter] = []
        self.lengths: List[int] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0

    def build(self, docs: List[Tuple[str, str]]) -> "BM25Index":
        """docs: Liste von (chunk_id, text)."""
        self.doc_ids, self.tfs, self.lengths = [], [], []
        self.df = Counter()

        for cid, text in docs:
            toks = tokenize(text)
            tf = Counter(toks)
            self.doc_ids.append(cid)
            self.tfs.append(tf)
            self.lengths.append(len(toks))
            for t in tf:
                self.df[t] += 1

        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        return self

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        q_terms = tokenize(query)
        if not q_terms or self.n_docs == 0:
            return []

        scores: Dict[str, float] = {}
        for i, tf in enumerate(self.tfs):
            dl = self.lengths[i] or 1
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += self._idf(t) * (f * (self.k1 + 1)) / denom
            if s > 0:
                scores[self.doc_ids[i]] = s

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return ranked


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """RRF: score(d) = sum over rankings of 1/(k + rank(d)).

    Fusioniert Ranglisten ohne Score-Normalisierung. Das ist der Grund, warum
    RRF hier gegenueber gewichteter Score-Addition bevorzugt wird: Cosine-Scores
    (0..1) und BM25-Scores (unbeschraenkt) sind nicht vergleichbar, ihre Raenge
    schon. k=60 ist der Wert aus der Originalarbeit von Cormack et al.
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused
