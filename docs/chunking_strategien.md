# Chunking-Strategien im RAG-Kontext

## Warum Chunking noetig ist

Embedding-Modelle haben ein Kontextlimit. Ein 40-seitiges PDF passt nicht in
einen einzigen Vektor, und selbst wenn es passte, waere der resultierende Vektor
ein Durchschnitt ueber zu viele Themen und damit fuer die Suche unbrauchbar.

## Feste Groesse gegen semantische Grenzen

Die einfachste Strategie schneidet alle N Zeichen. Sie ist schnell und
vorhersagbar, zerstoert aber Saetze und Absaetze. Ein Treffer, der mitten im Satz
beginnt, ist als Zitat wertlos.

Die Alternative respektiert Absatz- und Satzgrenzen und fuellt bis zu einer
Zielgroesse auf. Das ist etwas langsamer und erzeugt ungleich lange Chunks.

## Ueberlappung

Ohne Ueberlappung kann eine Antwort, die genau auf einer Chunk-Grenze liegt, in
keinem Chunk vollstaendig enthalten sein. Ueblich sind 10 bis 20 Prozent der
Chunk-Groesse als Ueberlappung. Der Preis ist redundanter Speicher und die
Moeglichkeit, dass derselbe Inhalt mehrfach in den Top-K-Treffern auftaucht.

## Groessenwahl

Kleine Chunks von 200 bis 400 Zeichen liefern praezise Treffer, aber wenig
Kontext. Grosse Chunks von 1000 bis 2000 Zeichen geben dem Modell mehr Kontext,
verwaessern aber das Embedding. Ein haeufig genutzter Startwert liegt bei etwa
800 Zeichen mit 150 Zeichen Ueberlappung. Diese Werte sind ein Ausgangspunkt und
sollten gegen ein eigenes Evaluationsset geprueft werden.
