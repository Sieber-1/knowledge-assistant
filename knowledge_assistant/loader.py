"""Dokumente laden: PDF, TXT, MD — seitenweise, damit Zitate eine Seitenzahl haben."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pdfplumber

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass
class Page:
    doc_id: str
    source: str
    page: int
    text: str


def load_file(path) -> List[Page]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Nicht unterstuetzt: {ext}")

    if ext == ".pdf":
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, p in enumerate(pdf.pages, 1):
                t = (p.extract_text() or "").strip()
                if t:
                    pages.append(Page(path.stem, path.name, i, t))
        if not pages:
            raise ValueError(
                f"{path.name}: kein Textlayer gefunden (vermutlich ein Scan)."
            )
        return pages

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"{path.name}: leer.")
    return [Page(path.stem, path.name, 1, text)]


def load_dir(folder) -> List[Page]:
    out = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in SUPPORTED:
            out.extend(load_file(p))
    return out
