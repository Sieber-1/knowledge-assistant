#!/usr/bin/env python3
"""Evaluation: Retrieval-Modi vergleichen, Chunk-Parameter sweepen.

    python evaluate.py --questions eval/questions.example.json --docs docs
    python evaluate.py --questions eval/questions.example.json --docs docs --sweep
    python evaluate.py --questions eval/questions.example.json --answers --llm extractive
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from knowledge_assistant.evaluate import (compare_modes, evaluate_answers,
                                          evaluate_retrieval, load_questions)
from knowledge_assistant.pipeline import build_store, ingest
from knowledge_assistant.rag import get_generator

SWEEP = [(400, 80), (600, 120), (800, 150), (1200, 200), (800, 0)]


def table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[i]).ljust(widths[i]) for i in range(len(r))))


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="RAG-Evaluation")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--embedding", default="hashing",
                    choices=["default", "openai", "hashing"])
    ap.add_argument("--sweep", action="store_true",
                    help="Chunk-Groessen durchprobieren")
    ap.add_argument("--answers", action="store_true",
                    help="Zusaetzlich Antwortqualitaet messen (kostet API-Calls)")
    ap.add_argument("--llm", default="extractive")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--json", help="Ergebnisse als JSON speichern")
    args = ap.parse_args()

    questions = load_questions(args.questions)
    n_ans = sum(1 for q in questions if q.answerable)
    print(f"{len(questions)} Fragen ({n_ans} beantwortbar, "
          f"{len(questions) - n_ans} bewusst nicht)\n")

    if len(questions) < 20:
        print(f"WARNUNG: {len(questions)} Fragen sind zu wenig fuer belastbare "
              f"Zahlen. Ab ~25-30 werden die Werte stabil.\n", file=sys.stderr)

    results = {}
    tmp = tempfile.mkdtemp()

    try:
        if args.sweep:
            print("=== Chunk-Parameter-Sweep (Modus: hybrid) ===\n")
            rows = []
            for target, overlap in SWEEP:
                db = Path(tmp) / f"db_{target}_{overlap}"
                store = build_store(args.embedding, str(db))
                rep = ingest(args.docs, store, reset=True,
                             target=target, overlap=overlap)
                m = evaluate_retrieval(questions, store, ks=(1, 3, 5))
                rows.append([f"{target}/{overlap}", rep.chunks,
                             f"{m.recall_at[1]:.2f}", f"{m.recall_at[3]:.2f}",
                             f"{m.recall_at[5]:.2f}", f"{m.mrr:.3f}"])
                results[f"chunk_{target}_{overlap}"] = {
                    "chunks": rep.chunks, "recall": m.recall_at, "mrr": m.mrr}
            table(rows, ["target/overlap", "chunks", "R@1", "R@3", "R@5", "MRR"])
            print()

        db = Path(tmp) / "db_main"
        store = build_store(args.embedding, str(db))
        rep = ingest(args.docs, store, reset=True)
        print(f"Index: {rep.chunks} Chunks aus {rep.files} Datei(en) "
              f"[{args.embedding}]\n")

        print("=== Retrieval-Modi ===\n")
        modes = compare_modes(questions, store, ks=(1, 3, 5, 10))
        rows = []
        for name, m in modes.items():
            rows.append([name, f"{m.recall_at[1]:.2f}", f"{m.recall_at[3]:.2f}",
                         f"{m.recall_at[5]:.2f}", f"{m.recall_at[10]:.2f}",
                         f"{m.mrr:.3f}"])
            results[f"mode_{name}"] = {"recall": m.recall_at, "mrr": m.mrr,
                                       "misses": m.misses}
        table(rows, ["mode", "R@1", "R@3", "R@5", "R@10", "MRR"])

        miss = modes["hybrid"].misses
        if miss:
            print(f"\nNicht gefunden (hybrid, k=10): {len(miss)}")
            for q in miss[:5]:
                print(f"  - {q}")

        if args.answers:
            print(f"\n=== Antwortqualitaet ({args.llm}) ===\n")
            gen = get_generator(args.llm)
            am = evaluate_answers(questions, store, gen, k=args.k)
            print(am.summary())
            results["answers"] = {
                "grounded_rate": am.grounded_rate,
                "refusal_correct": am.refusal_correct,
                "false_answers": am.false_answers,
            }
            if am.false_answers:
                print("\nFalsch beantwortet statt abgelehnt:")
                for q in am.false_answers:
                    print(f"  - {q}")

        if args.json:
            Path(args.json).write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\n-> {args.json}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
