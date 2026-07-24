# knowledge-assistant

# Knowledge Assistant (RAG)

Semantische Dokumentensuche ueber PDFs, TXT und Markdown. Hybrid-Retrieval
(Dense + BM25), Antworten mit Quellenangabe und satzweiser Belegpruefung.

## Schnellstart (offline, ohne API-Key und ohne Modell-Download)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python make_docs.py                                    # 3 Testdokumente
python tests/test_rag.py                               # 15 Unit-Tests
python cli.py --embedding hashing ingest docs --reset
python cli.py --embedding hashing ask "Was ist das Schluesselenzym des Calvin-Zyklus?" --llm extractive
streamlit run app.py
```

## Evaluation

```bash
python evaluate.py --questions eval/questions.example.json --docs docs --sweep
python evaluate.py --questions eval/questions.example.json --answers --llm extractive
python calibrate.py --questions eval/questions.example.json
```

### Gemessene Ergebnisse

32 Fragen (27 beantwortbar, 5 bewusst nicht beantwortbar), 3 Dokumente,
Backend `hashing`. Ein Treffer zaehlt nur, wenn der Beleg-Satz im abgerufenen
Chunk tatsaechlich enthalten ist — nicht schon bei richtiger Datei.

Chunk-Parameter (Modus hybrid):

| target/overlap | Chunks | R@1 | R@3 | MRR |
|---|---|---|---|---|
| 400/80 | 10 | 0.81 | 1.00 | 0.907 |
| **600/120** | 7 | **0.93** | 1.00 | **0.957** |
| 800/150 | 7 | 0.89 | 1.00 | 0.944 |
| 1200/200 | 6 | 0.89 | 1.00 | 0.944 |
| 800/**0** | 6 | 0.85 | 1.00 | 0.926 |

Zwei Befunde: 600/120 schlaegt den verbreiteten Startwert 800/150, und der
Wegfall der Ueberlappung (800/0) kostet messbar Genauigkeit. Der Default steht
weiterhin auf 800/150, weil ein Vorsprung von 0.04 auf 27 Fragen innerhalb der
Streuung liegt — bei ~3 Fragen Unterschied ist das kein belastbares Ergebnis.
Genau deshalb steht die Tabelle hier und nicht nur der Gewinner.

Retrieval-Modi:

| Modus | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|
| dense | 0.89 | 1.00 | 1.00 | 0.932 |
| bm25 | 0.89 | 1.00 | 1.00 | 0.944 |
| hybrid | 0.89 | 1.00 | 1.00 | 0.944 |

Hybrid bringt hier **keinen** Vorteil. Das ist zu erwarten: bei 7 Chunks und
einem Backend, das selbst lexikalisch arbeitet, haben beide Kanaele fast
dieselbe Sicht. Der Nutzen von BM25 zeigt sich erst bei groesseren Korpora und
echten Embeddings, wo Dense bei exakten Begriffen wie "RuBisCO" abfaellt —
`test_bm25_exact_term_beats_dense_noise` prueft genau diesen Fall isoliert.
Die Tabelle bleibt so stehen, weil ein negatives Messergebnis auch ein
Messergebnis ist.

Antwortqualitaet (`extractive`, k=5, MIN_SCORE=0.17):

| Metrik | Wert |
|---|---|
| grounded (beantwortbare Fragen) | 0.93 |
| korrekte Ablehnung (nicht beantwortbare) | 1.00 |

### Schwellwert-Kalibrierung

`MIN_SCORE` war urspruenglich auf 0.15 geraten. Die Messung ergab:

| MIN_SCORE | Recall | Refusal | F1 |
|---|---|---|---|
| 0.08 | 1.00 | 0.40 | 0.571 |
| 0.12 | 0.96 | 0.60 | 0.739 |
| 0.14 | 0.93 | 0.80 | 0.858 |
| **0.17** | **0.93** | **1.00** | **0.962** |
| 0.24 | 0.67 | 1.00 | 0.800 |
| 0.30 | 0.41 | 1.00 | 0.579 |

Mit dem geratenen Wert wurden 3 von 5 unbeantwortbaren Fragen faelschlich
beantwortet. Der Wert ist backend-spezifisch und nach jedem Wechsel des
Embeddings mit `calibrate.py` neu zu bestimmen.

## Richtiger Betrieb (echte Embeddings + LLM)

```bash
cp .env.example .env
python cli.py ingest docs --reset       # laedt beim ersten Lauf ~80 MB
python cli.py ask "Warum ist Ueberlappung noetig?" --llm anthropic
python calibrate.py --questions eval/questions.example.json --embedding default
```

## Architektur

```
loader.py      PDF/TXT/MD -> Seiten (Seitenzahl bleibt fuer Zitate erhalten)
chunking.py    absatz- und satzbewusst, mit Ueberlappung
embeddings.py  default (MiniLM) | openai | hashing (offline)
bm25.py        Okapi BM25 + Reciprocal Rank Fusion, ohne externe Abhaengigkeit
store.py       ChromaDB, Cosine-Distanz, persistent
retriever.py   Hybrid: Dense + BM25 ueber RRF fusioniert
rag.py         Prompt, Generatoren, satzweise Belegpruefung
pipeline.py    ingest(), retrieve(), ask()
evaluate.py    Recall@k, MRR, Refusal-Rate
calibrate.py   Schwellwert-Bestimmung
```

## Massnahmen gegen Halluzination

1. Systemprompt verbietet Vorwissen, feste Ablehnungsformel.
2. Zitierpflicht mit `[n]`.
3. **Satzweise Belegpruefung**: jeder Satz wird gegen den von ihm zitierten
   Chunk geprueft. Saetze ohne Zitat und Saetze, deren Inhaltswoerter im Beleg
   nicht vorkommen, setzen `grounded=False`.
4. Kalibrierter Relevanz-Schwellwert, gemessen gegen bewusst unbeantwortbare
   Fragen.
5. UI zeigt jeden Chunk im Volltext und den Support-Wert pro Satz.

## Bekannte Grenzen

- **Die Belegpruefung ist lexikalisch, kein NLI.** Sie erkennt den harten Fall
  (Begriffe kommen im Beleg gar nicht vor) und verfehlt den weichen: eine
  Verneinung oder vertauschte Zuordnung mit denselben Woertern passiert die
  Pruefung. Ein echtes Entailment-Modell waere der naechste Schritt.
- **`hashing` ist kein semantisches Embedding**, sondern gewichtete
  Wortueberlappung. Alle obigen Zahlen sind mit diesem Backend gemessen und
  **nicht auf `default` uebertragbar**. Fuer belastbare Aussagen ueber die
  Produktionskonfiguration muss die Evaluation mit `--embedding default`
  wiederholt werden.
- **27 beantwortbare Fragen sind wenig.** Unterschiede unter ~0.05 Recall sind
  Rauschen. Ab ~100 Fragen werden die Zahlen belastbar.
- **Das Fragenset ist synthetisch** und von derselben Person geschrieben wie die
  Dokumente. Das ist zirkulaer und misst eher Konsistenz als Realitaetstauglichkeit.
- **Kein Cross-Encoder-Reranking.** RRF ordnet nur um; ein Reranker wuerde die
  Top-k tatsaechlich neu bewerten.
- **BM25 haelt den Index im RAM**, neu gebaut bei Aenderung der Chunk-Zahl. Bis
  ~50k Chunks unproblematisch, darueber gehoert er persistiert.
- **Kein OCR.** Gescannte PDFs ohne Textlayer werden mit klarer Meldung abgelehnt.

## Entstehung

Erste Implementierung mit Claude erstellt, danach ueberarbeitet und gemessen.
Dabei gefundene und behobene Fehler:

- Satztrennung ordnete nachgestellte Zitate (`... ab. [1] Das ...`) dem falschen
  Satz zu; jeder erste Satz galt als unbelegt (`grounded` faelschlich 0.00).
- RRF normalisierte den Score auf den Bestwert, wodurch der Top-Treffer immer
  1.0 war und der Relevanz-Schwellwert wirkungslos wurde. Seitdem ist `score`
  die absolute Dense-Aehnlichkeit, `rrf_score` bestimmt nur die Reihenfolge.
- Evaluation mass zunaechst nur die Datei-Zuordnung; bei 3 Dokumenten erreichte
  jede Konfiguration Recall 1.00. Erst die Pruefung auf Chunk-Ebene
  (`expected_text`) machte die Chunk-Parameter unterscheidbar.


