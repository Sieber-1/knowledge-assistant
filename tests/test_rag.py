import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knowledge_assistant.chunking import chunk_pages, TARGET_CHARS   # noqa: E402
from knowledge_assistant.loader import Page                          # noqa: E402
from knowledge_assistant.pipeline import ask, ingest, build_store    # noqa: E402
from knowledge_assistant.bm25 import (BM25Index,                     # noqa: E402
                                      reciprocal_rank_fusion)
from knowledge_assistant.rag import (NO_ANSWER, build_context,       # noqa: E402
                                     check_grounding, sentence_support,
                                     ExtractiveGenerator)
from knowledge_assistant.retriever import HybridRetriever            # noqa: E402
from knowledge_assistant.store import Hit                            # noqa: E402


def test_chunk_respects_size():
    text = "\n\n".join(f"Absatz {i}. " + "Wort " * 40 for i in range(12))
    chunks = chunk_pages([Page("d", "d.pdf", 1, text)])
    assert len(chunks) > 1
    assert all(len(c.text) < TARGET_CHARS * 2.2 for c in chunks)
    assert all(c.page == 1 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_overlap_exists():
    text = "\n\n".join(f"Satz {i} mit etwas Text daran. " * 6 for i in range(8))
    chunks = chunk_pages([Page("d", "d.pdf", 1, text)])
    assert len(chunks) >= 2
    tail = chunks[0].text[-60:].split()
    assert any(w in chunks[1].text for w in tail if len(w) > 3)


def test_pages_not_merged():
    pages = [Page("d", "d.pdf", 1, "Seite eins Inhalt. " * 20),
             Page("d", "d.pdf", 2, "Seite zwei Inhalt. " * 20)]
    chunks = chunk_pages(pages)
    for c in chunks:
        if "eins" in c.text:
            assert "zwei" not in c.text


def test_grounding_detects_bad_citation():
    hits = [Hit("a", "text", "d.pdf", 1, 0.9)]
    cited, warns, grounded, _ = check_grounding("Aussage [3].", hits)
    assert not grounded
    assert any("nicht existierende" in w for w in warns)


def test_grounding_flags_missing_citation():
    hits = [Hit("a", "text", "d.pdf", 1, 0.9)]
    long_text = "Eine ausfuehrliche Antwort ohne jede Quellenangabe im Text."
    _, warns, grounded, _ = check_grounding(long_text, hits)
    assert not grounded
    assert any("keine Quellenangabe" in w for w in warns)


def test_grounding_accepts_valid():
    hits = [Hit("a", "Alpha Beta Gamma Delta", "d.pdf", 1, 0.9),
            Hit("b", "Epsilon Zeta Eta Theta", "d.pdf", 2, 0.8)]
    cited, warns, grounded, _ = check_grounding(
        "Alpha Beta Gamma Delta stehen hier [1]. Epsilon Zeta Eta Theta auch [2].",
        hits)
    assert grounded and cited == [1, 2] and warns == []


def test_extractive_refuses_when_irrelevant():
    hits = [Hit("a", "Der Calvin-Zyklus laeuft im Stroma ab und braucht ATP.",
                "d.pdf", 1, 0.5)]
    out = ExtractiveGenerator().generate("Wie hoch ist der Eiffelturm?",
                                         build_context(hits))
    assert out == NO_ANSWER


def test_end_to_end_hashing():
    tmp = tempfile.mkdtemp()
    try:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        (docs / "a.md").write_text(
            "# Chloroplasten\n\n"
            "Die Photosynthese findet in den Chloroplasten statt. "
            "Das zentrale Pigment ist Chlorophyll a.\n\n"
            "Der Calvin-Zyklus laeuft im Stroma ab und benoetigt ATP und NADPH "
            "als Energietraeger fuer die Fixierung von Kohlendioxid.\n",
            encoding="utf-8")

        store = build_store("hashing", str(Path(tmp) / "db"))
        rep = ingest(docs, store)
        assert rep.chunks > 0 and store.count() == rep.chunks

        a = ask("Wo findet die Photosynthese statt?", store,
                ExtractiveGenerator(), k=3)
        assert "Chloroplasten" in a.text
        assert a.cited and a.grounded

        b = ask("Wie funktioniert ein Dieselmotor?", store,
                ExtractiveGenerator(), k=3)
        assert b.text == NO_ANSWER
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bm25_exact_term_beats_dense_noise():
    idx = BM25Index().build([
        ("a", "Das Schluesselenzym des Calvin-Zyklus ist RuBisCO."),
        ("b", "Pflanzen wandeln Lichtenergie in chemische Energie um."),
        ("c", "Die Temperatur beeinflusst viele biologische Prozesse."),
    ])
    top = idx.search("RuBisCO", k=3)
    assert top and top[0][0] == "a"


def test_bm25_idf_favours_rare_terms():
    idx = BM25Index().build([
        ("a", "Wasser Wasser Wasser Photosynthese"),
        ("b", "Wasser Wasser Wasser RuBisCO"),
        ("c", "Wasser Wasser Wasser Wasser"),
    ])
    scores = dict(idx.search("Wasser RuBisCO", k=3))
    assert scores["b"] > scores["c"]


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([["x", "y", "z"], ["y", "x", "z"]])
    assert fused["y"] > fused["z"] and fused["x"] > fused["z"]


def test_sentence_support_detects_unsupported():
    chunk = "Der Calvin-Zyklus laeuft im Stroma der Chloroplasten ab."
    assert sentence_support("Der Calvin-Zyklus laeuft im Stroma ab.", chunk) > 0.6
    assert sentence_support("Der Eiffelturm steht in Paris.", chunk) < 0.2


def test_grounding_flags_unsupported_sentence():
    hits = [Hit("a", "Der Calvin-Zyklus laeuft im Stroma der Chloroplasten ab.",
                "d.pdf", 1, 0.9)]
    _, warns, grounded, reports = check_grounding(
        "Der Eiffelturm in Paris ist dreihundert Meter hoch gebaut worden [1].",
        hits)
    assert not grounded
    assert any("nicht gestuetzt" in w for w in warns)
    assert reports[0]["ok"] is False


def test_hybrid_score_stays_absolute():
    """RRF darf den Score nicht auf 1.0 normalisieren, sonst ist der
    Schwellwert wirkungslos und irrelevante Fragen werden nie abgelehnt."""
    tmp = tempfile.mkdtemp()
    try:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        (docs / "a.md").write_text(
            "Die Photosynthese findet in den Chloroplasten statt und "
            "wandelt Lichtenergie in chemische Energie um.\n",
            encoding="utf-8")
        store = build_store("hashing", str(Path(tmp) / "db"))
        ingest(docs, store, reset=True)
        r = HybridRetriever(store)
        good = r.search("Wo findet die Photosynthese statt?", k=3)
        bad = r.search("Wie repariere ich ein Fahrradschaltwerk?", k=3)
        assert good[0].score > bad[0].score
        assert bad[0].score < 0.17
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unanswerable_is_refused_end_to_end():
    tmp = tempfile.mkdtemp()
    try:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        (docs / "a.md").write_text(
            "Der Calvin-Zyklus laeuft im Stroma der Chloroplasten ab und "
            "benoetigt ATP sowie NADPH als Energietraeger.\n",
            encoding="utf-8")
        store = build_store("hashing", str(Path(tmp) / "db"))
        ingest(docs, store, reset=True)
        a = ask("Wie hoch ist der Eiffelturm?", store, ExtractiveGenerator(), k=3)
        assert a.text == NO_ANSWER
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} bestanden")
    sys.exit(1 if failed else 0)
