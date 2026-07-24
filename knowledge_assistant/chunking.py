"""Chunking.

Strategie: Absatzgrenzen respektieren, dann auf Zielgroesse fuellen, mit
Ueberlappung zwischen benachbarten Chunks.

Warum nicht stumpf alle N Zeichen schneiden: ein Schnitt mitten im Satz
zerstoert den Embedding-Kontext und liefert Treffer, die als Zitat unbrauchbar
sind. Warum Ueberlappung: eine Antwort, die genau auf einer Chunk-Grenze liegt,
waere sonst in keinem einzigen Chunk vollstaendig.

Die Werte (800 Zeichen, 150 Overlap) sind ein Startpunkt, keine gemessene
Optimierung — siehe README.
"""
import re
from dataclasses import dataclass, asdict
from typing import List

from .loader import Page

TARGET_CHARS = 800
OVERLAP_CHARS = 150
MIN_CHARS = 100


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    page: int
    index: int
    text: str

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("text")
        d.pop("chunk_id")
        return d


def _split_paragraphs(text: str) -> List[str]:
    parts = re.split(r"\n\s*\n", text)
    out = []
    for p in parts:
        p = re.sub(r"[ \t]+", " ", p).strip()
        if not p:
            continue
        if len(p) <= TARGET_CHARS * 1.5:
            out.append(p)
            continue
        # Zu langer Absatz: an Satzenden weiterteilen.
        sentences = re.split(r"(?<=[.!?])\s+", p)
        buf = ""
        for s in sentences:
            if len(buf) + len(s) + 1 > TARGET_CHARS and buf:
                out.append(buf.strip())
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            out.append(buf.strip())
    return out


def chunk_pages(pages: List[Page],
                target: int = TARGET_CHARS,
                overlap: int = OVERLAP_CHARS) -> List[Chunk]:
    chunks: List[Chunk] = []
    counter = 0

    for page in pages:
        paragraphs = _split_paragraphs(page.text)
        buf = ""
        for para in paragraphs:
            if buf and len(buf) + len(para) + 2 > target:
                chunks.append(_mk(page, counter, buf))
                counter += 1
                tail = buf[-overlap:] if overlap else ""
                # Ueberlappung an Wortgrenze beginnen lassen.
                if " " in tail:
                    tail = tail[tail.index(" ") + 1:]
                buf = f"{tail} {para}".strip() if tail else para
            else:
                buf = f"{buf}\n\n{para}".strip() if buf else para

        if buf:
            if len(buf) < MIN_CHARS and chunks and chunks[-1].page == page.page:
                chunks[-1].text += "\n\n" + buf   # Rest an vorigen Chunk anhaengen
            else:
                chunks.append(_mk(page, counter, buf))
                counter += 1

    return chunks


def _mk(page: Page, idx: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{page.doc_id}-p{page.page}-c{idx}",
        doc_id=page.doc_id,
        source=page.source,
        page=page.page,
        index=idx,
        text=text.strip(),
    )
