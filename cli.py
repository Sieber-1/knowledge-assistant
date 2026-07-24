#!/usr/bin/env python3
import argparse
import json
import sys

from dotenv import load_dotenv

from knowledge_assistant.pipeline import ask, build_store, ingest
from knowledge_assistant.rag import get_generator


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Knowledge Assistant (RAG)")
    ap.add_argument("--db", default="chroma_db", help="Pfad zur Vektor-DB")
    ap.add_argument("--embedding", choices=["default", "openai", "hashing"],
                    help="hashing = offline, ohne Download")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_in = sub.add_parser("ingest", help="Dokumente indexieren")
    p_in.add_argument("path")
    p_in.add_argument("--reset", action="store_true")

    p_ask = sub.add_parser("ask", help="Frage stellen")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=5)
    p_ask.add_argument("--source", help="Nur in dieser Datei suchen")
    p_ask.add_argument("--llm", choices=["anthropic", "openai", "extractive"])
    p_ask.add_argument("--mode", default="hybrid",
                       choices=["dense", "bm25", "hybrid"],
                       help="Retrieval-Kanal (Standard: hybrid)")
    p_ask.add_argument("--model")
    p_ask.add_argument("--json", action="store_true")

    sub.add_parser("info", help="Status des Index")

    args = ap.parse_args()

    try:
        store = build_store(args.embedding, args.db)
    except Exception as e:
        print(f"Embedding-Fehler: {e}", file=sys.stderr)
        sys.exit(2)

    if args.cmd == "ingest":
        r = ingest(args.path, store, reset=args.reset)
        print(f"{r.files} Datei(en) - {r.pages} Seiten - {r.chunks} Chunks "
              f"({r.duration_s:.1f}s)")
        print(f"Index enthaelt jetzt {store.count()} Chunks "
              f"[{store.backend.name}]")
        for e in r.errors:
            print(f"  ! {e}", file=sys.stderr)
        sys.exit(1 if r.errors and r.chunks == 0 else 0)

    if args.cmd == "info":
        print(f"DB:       {args.db}")
        print(f"Backend:  {store.backend.name}")
        print(f"Chunks:   {store.count()}")
        for s in store.sources():
            print(f"  - {s}")
        return

    # ask
    try:
        gen = get_generator(args.llm, **({"model": args.model} if args.model else {}))
    except Exception as e:
        print(f"Generator-Fehler: {e}", file=sys.stderr)
        sys.exit(2)

    a = ask(args.question, store, gen, k=args.k, source=args.source,
            mode=args.mode)

    if args.json:
        print(json.dumps(a.to_dict(), indent=2, ensure_ascii=False))
        return

    print(f"\n{a.text}\n")
    print("Quellen:")
    for i, h in enumerate(a.hits, 1):
        mark = "*" if i in a.cited else " "
        print(f" {mark}[{i}] {h.source} S.{h.page}  score={h.score}")
    if a.sentence_reports:
        print("\nSatzpruefung:")
        for r in a.sentence_reports:
            mark = "ok" if r["ok"] else "!!"
            print(f" {mark} support={r['support']:.2f} cites={r['cites']} "
                  f"{r['sentence'][:60]}")

    if a.warnings:
        print("\nWarnungen:", file=sys.stderr)
        for w in a.warnings:
            print(f"  ! {w}", file=sys.stderr)


if __name__ == "__main__":
    main()
