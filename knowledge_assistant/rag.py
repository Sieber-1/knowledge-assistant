"""Antwortgenerierung mit Quellenbindung.

Kernidee gegen Halluzination: das Modell bekommt nummerierte Kontextbloecke und
muss jede Aussage mit [1], [2] ... belegen. Anschliessend wird geprueft, ob die
zitierten Nummern ueberhaupt existieren und ob die Antwort ohne Belege
auskommt — beides wird gemeldet, nicht stillschweigend akzeptiert.
"""
import os
import re
from dataclasses import dataclass, field
from typing import List

from .embeddings import tokenize
from .store import Hit

SYSTEM_PROMPT = """Du beantwortest Fragen ausschliesslich auf Basis der \
bereitgestellten Kontextauszuege.

Regeln:
- Nutze NUR Informationen aus dem Kontext. Kein Vorwissen, keine Ergaenzungen.
- Belege jede Aussage mit der Quellennummer in eckigen Klammern, z.B. [2].
- Wenn der Kontext die Frage nicht beantwortet, schreibe genau:
  "Die Dokumente enthalten dazu keine Information."
  Rate nicht und fuelle keine Luecken.
- Wenn der Kontext sich widerspricht, benenne den Widerspruch mit beiden Quellen.
- Antworte knapp und auf Deutsch, sofern die Frage nicht englisch gestellt ist."""

NO_ANSWER = "Die Dokumente enthalten dazu keine Information."


@dataclass
class Answer:
    question: str
    text: str
    hits: List[Hit]
    cited: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    grounded: bool = True
    sentence_reports: List[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.text,
            "grounded": self.grounded,
            "warnings": self.warnings,
            "sentence_checks": self.sentence_reports,
            "sources": [
                {"n": i + 1, "source": h.source, "page": h.page,
                 "score": h.score, "cited": (i + 1) in self.cited}
                for i, h in enumerate(self.hits)
            ],
        }


def build_context(hits: List[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] Quelle: {h.source}, Seite {h.page}\n{h.text}")
    return "\n\n---\n\n".join(blocks)


SUPPORT_THRESHOLD = 0.35   # Anteil der Inhaltswoerter, der im Beleg vorkommen muss


def sentence_support(sentence: str, chunk_text: str) -> float:
    """Anteil der Inhaltswoerter des Satzes, die im belegenden Chunk vorkommen.

    Das ist eine lexikalische Naeherung an Entailment, kein NLI-Modell. Sie
    erkennt zuverlaessig den harten Fall — eine Aussage, deren Begriffe im
    zitierten Chunk gar nicht auftauchen — und verfehlt den weichen Fall, in dem
    dieselben Woerter etwas anderes behaupten (Verneinung, vertauschte
    Zuordnung). Siehe README, Abschnitt Grenzen.
    """
    s_toks = set(tokenize(sentence))
    if not s_toks:
        return 1.0
    c_toks = set(tokenize(chunk_text))
    return len(s_toks & c_toks) / len(s_toks)


def check_grounding(text: str, hits: List[Hit],
                    threshold: float = SUPPORT_THRESHOLD) -> tuple:
    """Zitate formal UND inhaltlich pruefen.

    Rueckgabe: (zitierte Nummern, Warnungen, grounded, Satzberichte)
    """
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    warnings = []
    valid = [c for c in cited if 1 <= c <= len(hits)]
    invalid = [c for c in cited if c not in valid]

    if invalid:
        warnings.append(f"Zitat verweist auf nicht existierende Quelle: {invalid}")
    if not cited and NO_ANSWER not in text and len(text.strip()) > 40:
        warnings.append("Antwort enthaelt keine Quellenangabe.")

    # Inhaltliche Pruefung: jeder Satz gegen die von ihm zitierten Chunks.
    reports = []
    unsupported = 0
    uncited_sentences = 0

    if NO_ANSWER not in text:
        # Trennung nur nach Satzende ODER nach einem darauf folgenden Zitat.
        # Ohne die Klammer im Lookbehind wuerde "... ab. [1] Das ..." das Zitat
        # dem naechsten Satz zuschlagen und den ersten als unbelegt melden.
        for sent in re.split(r"(?<=[.!?])\s+(?!\[)|(?<=\])\s+", text.strip()):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            nums = [int(n) for n in re.findall(r"\[(\d+)\]", sent)]
            nums = [n for n in nums if 1 <= n <= len(hits)]
            clean = re.sub(r"\[\d+\]", "", sent).strip()

            if not nums:
                uncited_sentences += 1
                reports.append({"sentence": clean, "cites": [],
                                "support": 0.0, "ok": False})
                continue

            best = max(sentence_support(clean, hits[n - 1].text) for n in nums)
            ok = best >= threshold
            if not ok:
                unsupported += 1
            reports.append({"sentence": clean, "cites": nums,
                            "support": round(best, 2), "ok": ok})

    if unsupported:
        warnings.append(
            f"{unsupported} Satz/Saetze werden vom zitierten Chunk nicht gestuetzt "
            f"(Schwellwert {threshold})."
        )
    if uncited_sentences:
        warnings.append(f"{uncited_sentences} Satz/Saetze ohne Quellenangabe.")

    grounded = (not invalid and not unsupported and not uncited_sentences
                and (bool(cited) or NO_ANSWER in text))
    return valid, warnings, grounded, reports


# --- Provider --------------------------------------------------------------

class Generator:
    name = "base"

    def generate(self, question: str, context: str) -> str:
        raise NotImplementedError


class AnthropicGenerator(Generator):
    name = "anthropic"

    def __init__(self, model="claude-sonnet-4-5", api_key=None):
        import anthropic
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY fehlt (oder --llm extractive).")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def generate(self, question: str, context: str) -> str:
        r = self.client.messages.create(
            model=self.model, max_tokens=1000, system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                       "content": f"Kontext:\n\n{context}\n\nFrage: {question}"}],
        )
        return r.content[0].text.strip()


class OpenAIGenerator(Generator):
    name = "openai"

    def __init__(self, model="gpt-4o-mini", api_key=None):
        from openai import OpenAI
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY fehlt (oder --llm extractive).")
        self.client = OpenAI(api_key=key)
        self.model = model

    def generate(self, question: str, context: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",
                       "content": f"Kontext:\n\n{context}\n\nFrage: {question}"}],
        )
        return r.choices[0].message.content.strip()


class ExtractiveGenerator(Generator):
    """Offline-Fallback: waehlt die Saetze mit der hoechsten Wortueberlappung
    zur Frage und gibt sie woertlich mit Quellennummer zurueck.

    Erfindet per Konstruktion nichts, formuliert aber auch nicht — das ist
    kein Ersatz fuer ein LLM, sondern eine Baseline, gegen die man das LLM
    vergleichen kann.
    """
    name = "extractive"

    def __init__(self, model="sentence-overlap", api_key=None, max_sentences=3):
        self.model = model
        self.max_sentences = max_sentences

    def generate(self, question: str, context: str) -> str:
        q = set(tokenize(question))
        if not q:
            return NO_ANSWER

        scored = []
        for block in context.split("\n\n---\n\n"):
            m = re.match(r"\[(\d+)\]", block)
            if not m:
                continue
            n = int(m.group(1))
            body = block.split("\n", 1)[1] if "\n" in block else ""
            for sent in re.split(r"(?<=[.!?])\s+", body.replace("\n", " ")):
                sent = sent.strip()
                if len(sent) < 25:
                    continue
                toks = set(tokenize(sent))
                if not toks:
                    continue
                overlap = len(q & toks) / len(q)
                if overlap > 0:
                    scored.append((overlap, n, sent))

        scored.sort(key=lambda x: -x[0])
        picked = scored[:self.max_sentences]
        if not picked or picked[0][0] < 0.15:
            return NO_ANSWER
        return " ".join(f"{s} [{n}]" for _, n, s in picked)


def get_generator(name: str = None, **kw) -> Generator:
    name = (name or os.getenv("LLM_PROVIDER", "anthropic")).lower()
    if name == "anthropic":
        return AnthropicGenerator(**kw)
    if name == "openai":
        return OpenAIGenerator(**kw)
    if name in ("extractive", "mock"):
        return ExtractiveGenerator(**{k: v for k, v in kw.items() if k != "api_key"})
    raise ValueError(f"Unbekannter Generator: {name}")
