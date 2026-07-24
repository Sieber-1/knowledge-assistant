"""Evaluation des Retrieval und der Antwortqualitaet.

Metriken:
  Recall@k  Anteil der Fragen, bei denen mindestens ein erwarteter Chunk unter
            den Top-k liegt. Das ist die Obergrenze fuer die Antwortqualitaet:
            was nicht abgerufen wird, kann nicht beantwortet werden.
  MRR       Mean Reciprocal Rank. 1/Rang des ersten korrekten Treffers,
            gemittelt. Bestraft korrekte Treffer auf Platz 5 gegenueber Platz 1.
  Refusal   Anteil der Fragen ohne Antwort im Korpus, die korrekt abgelehnt
            werden. Ohne diese Metrik optimiert man ein System, das immer
            irgendetwas antwortet.

Ground Truth ist eine JSON-Datei; Format siehe eval/questions.example.json.
Ein Eintrag gilt als getroffen, wenn Quelle UND (falls angegeben) Seite
uebereinstimmen. Bei answerable=false wird erwartet, dass das System ablehnt.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .pipeline import ask, retrieve
from .rag import NO_ANSWER
from .retriever import HybridRetriever
from .store import Hit, VectorStore


@dataclass
class Question:
    question: str
    answerable: bool = True
    expected_source: Optional[str] = None
    expected_page: Optional[int] = None
    expected_text: Optional[str] = None
    note: str = ""

    @staticmethod
    def from_dict(d: dict) -> "Question":
        return Question(
            question=d["question"],
            answerable=d.get("answerable", True),
            expected_source=d.get("expected_source"),
            expected_page=d.get("expected_page"),
            expected_text=d.get("expected_text"),
            note=d.get("note", ""),
        )


def load_questions(path) -> List[Question]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["questions"] if isinstance(data, dict) else data
    qs = [Question.from_dict(d) for d in items]
    for q in qs:
        if q.answerable and not q.expected_source:
            raise ValueError(
                f"Frage ohne erwartete Quelle: {q.question!r}. "
                f"Entweder expected_source setzen oder answerable=false."
            )
    return qs


def _is_match(hit: Hit, q: Question) -> bool:
    """Treffer, wenn Quelle (und ggf. Seite) passen UND — falls angegeben — der
    Beleg-Text im Chunk enthalten ist.

    Die Textpruefung ist der eigentlich aussagekraeftige Teil: bei wenigen
    Dokumenten ist die blosse Datei-Zuordnung trivial und jede Konfiguration
    erreicht Recall 1.0. Erst die Frage "steht die Antwort wirklich in diesem
    Chunk?" macht Chunk-Groesse und Ueberlappung messbar.
    """
    if hit.source != q.expected_source:
        return False
    if q.expected_page is not None and hit.page != q.expected_page:
        return False
    if q.expected_text:
        return _normalize(q.expected_text) in _normalize(hit.text)
    return True


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def _first_hit_rank(hits: List[Hit], q: Question) -> Optional[int]:
    for i, h in enumerate(hits, 1):
        if _is_match(h, q):
            return i
    return None


@dataclass
class RetrievalMetrics:
    n_answerable: int = 0
    recall_at: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    misses: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"n={self.n_answerable}"]
        for k in sorted(self.recall_at):
            parts.append(f"R@{k}={self.recall_at[k]:.2f}")
        parts.append(f"MRR={self.mrr:.3f}")
        return "  ".join(parts)


def evaluate_retrieval(questions: List[Question], store: VectorStore,
                       ks: List[int] = (1, 3, 5, 10),
                       mode: str = "hybrid") -> RetrievalMetrics:
    answerable = [q for q in questions if q.answerable]
    if not answerable:
        return RetrievalMetrics()

    retriever = HybridRetriever(store)
    max_k = max(ks)
    ranks, misses = [], []

    for q in answerable:
        hits = retrieve(q.question, store, k=max_k, mode=mode,
                        retriever=retriever)
        r = _first_hit_rank(hits, q)
        ranks.append(r)
        if r is None:
            misses.append(q.question)

    m = RetrievalMetrics(n_answerable=len(answerable), misses=misses)
    for k in ks:
        hit = sum(1 for r in ranks if r is not None and r <= k)
        m.recall_at[k] = hit / len(ranks)
    m.mrr = sum((1.0 / r) for r in ranks if r is not None) / len(ranks)
    return m


@dataclass
class AnswerMetrics:
    n: int = 0
    grounded_rate: float = 0.0
    refusal_correct: float = 0.0
    n_unanswerable: int = 0
    false_answers: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"n={self.n}  grounded={self.grounded_rate:.2f}  "
                f"refusal={self.refusal_correct:.2f} (n={self.n_unanswerable})")


def evaluate_answers(questions: List[Question], store: VectorStore,
                     generator, k: int = 5, mode: str = "hybrid") -> AnswerMetrics:
    retriever = HybridRetriever(store)
    grounded, refused_ok, false_answers = 0, 0, []
    unanswerable = [q for q in questions if not q.answerable]

    for q in questions:
        a = ask(q.question, store, generator, k=k, mode=mode, retriever=retriever)
        refused = NO_ANSWER in a.text

        if q.answerable:
            if a.grounded and not refused:
                grounded += 1
        else:
            if refused:
                refused_ok += 1
            else:
                false_answers.append(q.question)

    n_ans = len(questions) - len(unanswerable)
    return AnswerMetrics(
        n=len(questions),
        grounded_rate=(grounded / n_ans) if n_ans else 0.0,
        refusal_correct=(refused_ok / len(unanswerable)) if unanswerable else 0.0,
        n_unanswerable=len(unanswerable),
        false_answers=false_answers,
    )


def compare_modes(questions: List[Question], store: VectorStore,
                  ks: List[int] = (1, 3, 5)) -> Dict[str, RetrievalMetrics]:
    return {m: evaluate_retrieval(questions, store, ks, mode=m)
            for m in ("dense", "bm25", "hybrid")}
