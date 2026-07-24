#!/usr/bin/env python3
"""Kalibriert den Relevanz-Schwellwert (MIN_SCORE) gegen ein Fragenset.

Der Schwellwert entscheidet, ab welcher Aehnlichkeit ein Treffer als Kontext
durchgereicht wird. Zu niedrig: das System beantwortet Fragen, zu denen nichts
im Korpus steht. Zu hoch: es lehnt beantwortbare Fragen ab.

Das Verfahren: Top-Score fuer jede beantwortbare und jede bewusst
unbeantwortbare Frage messen, dann den Wert suchen, der beide Fehler
ausbalanciert (hoechstes F1 aus Recall und Refusal).

Der Wert ist NICHT uebertragbar zwischen Embedding-Backends. Nach einem Wechsel
von hashing auf default neu laufen lassen.

    python calibrate.py --questions eval/questions.example.json --embedding default
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from knowledge_assistant.evaluate import load_questions
from knowledge_assistant.pipeline import build_store, ingest, retrieve
from knowledge_assistant.retriever import HybridRetriever


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--embedding", default="hashing",
                    choices=["default", "openai", "hashing"])
    ap.add_argument("--mode", default="hybrid",
                    choices=["dense", "bm25", "hybrid"])
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    questions = load_questions(args.questions)
    pos_q = [q for q in questions if q.answerable]
    neg_q = [q for q in questions if not q.answerable]

    if not neg_q:
        print("FEHLER: Ohne Fragen mit answerable=false laesst sich der "
              "Schwellwert nicht kalibrieren. Es fehlt die Gegenprobe.")
        return

    tmp = tempfile.mkdtemp()
    try:
        store = build_store(args.embedding, str(Path(tmp) / "db"))
        rep = ingest(args.docs, store, reset=True)
        print(f"Index: {rep.chunks} Chunks [{args.embedding}, {args.mode}]")
        print(f"Fragen: {len(pos_q)} beantwortbar, {len(neg_q)} nicht\n")

        r = HybridRetriever(store)
        pos = [retrieve(q.question, store, k=5, mode=args.mode,
                        retriever=r)[0].score for q in pos_q]
        neg = [retrieve(q.question, store, k=5, mode=args.mode,
                        retriever=r)[0].score for q in neg_q]

        print(f"Top-Score beantwortbar : min={min(pos):.3f} "
              f"median={sorted(pos)[len(pos)//2]:.3f} max={max(pos):.3f}")
        print(f"Top-Score nicht beantw.: min={min(neg):.3f} "
              f"median={sorted(neg)[len(neg)//2]:.3f} max={max(neg):.3f}")

        overlap = max(neg) >= min(pos)
        print(f"\nUeberlappung der Verteilungen: {'JA' if overlap else 'NEIN'}"
              f"{' - perfekte Trennung nicht moeglich' if overlap else ''}\n")

        hi = max(max(pos), max(neg))
        print(" thr    Recall  Refusal  F1")
        best = None
        for i in range(args.steps + 1):
            t = round(hi * i / args.steps, 3)
            rec = sum(1 for x in pos if x >= t) / len(pos)
            ref = sum(1 for x in neg if x < t) / len(neg)
            f1 = 0.0 if rec + ref == 0 else 2 * rec * ref / (rec + ref)
            if best is None or f1 > best[1]:
                best = (t, f1, rec, ref)
            print(f" {t:.3f}  {rec:.2f}    {ref:.2f}     {f1:.3f}")

        print(f"\nEmpfehlung: MIN_SCORE = {best[0]:.2f}  "
              f"(Recall {best[2]:.2f}, Refusal {best[3]:.2f})")
        print("In knowledge_assistant/pipeline.py eintragen.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
