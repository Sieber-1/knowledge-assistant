"""Embedding-Backends.

- default : ChromaDB all-MiniLM-L6-v2 (ONNX). Laedt beim ersten Lauf ~80 MB.
            Das ist der Standardweg fuer echte Nutzung.
- openai  : text-embedding-3-small ueber API.
- hashing : lokaler TF-Hashing-Vektor, keine Downloads, keine Kosten.
            NUR fuer Tests und Demos ohne Netz. Kein semantisches Verstaendnis —
            findet Wortueberlappung, keine Synonyme. Siehe README.
"""
import hashlib
import math
import os
import re
from typing import List

Vector = List[float]


class EmbeddingBackend:
    name = "base"
    dim = 0

    def embed(self, texts: List[str]) -> List[Vector]:
        raise NotImplementedError


# --- offline fallback ------------------------------------------------------

_TOKEN = re.compile(r"[a-zA-ZaeoeueAEOEUEss0-9\u00e4\u00f6\u00fc\u00c4\u00d6\u00dc\u00df]+")

GERMAN_STOPWORDS = {
    "der", "die", "das", "und", "oder", "ein", "eine", "einer", "eines", "dem",
    "den", "des", "ist", "sind", "war", "waren", "im", "in", "an", "auf", "fuer",
    "mit", "von", "zu", "zum", "zur", "bei", "aus", "als", "auch", "sich", "nicht",
    "the", "of", "and", "a", "to", "is", "in", "for", "on", "with", "that", "it",
}


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _TOKEN.findall(text)]
    return [t for t in toks if t not in GERMAN_STOPWORDS and len(t) > 2]


class HashingBackend(EmbeddingBackend):
    """Feature hashing + sublinear TF, L2-normalisiert.

    Cosine-Aehnlichkeit darauf entspricht ungefaehr gewichteter
    Wortueberlappung. Kein Ersatz fuer echte Embeddings.
    """
    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _hash(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim

    def embed(self, texts: List[str]) -> List[Vector]:
        out = []
        for t in texts:
            vec = [0.0] * self.dim
            counts = {}
            for tok in tokenize(t):
                counts[tok] = counts.get(tok, 0) + 1
            for tok, c in counts.items():
                vec[self._hash(tok)] += 1.0 + math.log(c)
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


# --- real backends ---------------------------------------------------------

class DefaultBackend(EmbeddingBackend):
    """all-MiniLM-L6-v2 via ChromaDB. 384 Dimensionen, laeuft lokal."""
    name = "default"
    dim = 384

    def __init__(self):
        from chromadb.utils import embedding_functions
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed(self, texts: List[str]) -> List[Vector]:
        return [list(v) for v in self._ef(texts)]


class OpenAIBackend(EmbeddingBackend):
    name = "openai"
    dim = 1536

    def __init__(self, model="text-embedding-3-small", api_key=None):
        from openai import OpenAI
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY fehlt.")
        self.client = OpenAI(api_key=key)
        self.model = model

    def embed(self, texts: List[str]) -> List[Vector]:
        out = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i + 100]
            r = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in r.data])
        return out


def get_backend(name: str = None, **kw) -> EmbeddingBackend:
    name = (name or os.getenv("EMBEDDING_BACKEND", "default")).lower()
    if name == "default":
        return DefaultBackend()
    if name == "openai":
        return OpenAIBackend(**kw)
    if name == "hashing":
        return HashingBackend(**kw)
    raise ValueError(f"Unbekanntes Embedding-Backend: {name}")
