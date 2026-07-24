#!/usr/bin/env python3
"""Erzeugt Testdokumente in docs/ (PDF + Markdown)."""
from pathlib import Path
from textwrap import wrap

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

OUT = Path(__file__).parent / "docs"

PDF_DOCS = {
    "photosynthese_grundlagen.pdf": [
        ("Photosynthese: Grundlagen",
         """Die Photosynthese ist der Prozess, mit dem gruene Pflanzen Lichtenergie in
chemische Energie umwandeln. Sie findet in den Chloroplasten statt, genauer in
den Thylakoidmembranen und im Stroma.

Der Gesamtprozess laesst sich in zwei Phasen unterteilen: die Lichtreaktion und
die Dunkelreaktion, die auch als Calvin-Zyklus bezeichnet wird. Die
Lichtreaktion benoetigt direktes Licht, die Dunkelreaktion nicht.

Das zentrale Pigment ist Chlorophyll a. Es absorbiert vor allem Licht im blauen
Bereich um 430 Nanometer und im roten Bereich um 662 Nanometer. Gruenes Licht
wird ueberwiegend reflektiert, weshalb Blaetter gruen erscheinen."""),
        ("Lichtreaktion",
         """In der Lichtreaktion wird Wasser gespalten. Dabei entsteht Sauerstoff als
Nebenprodukt. Pro gespaltenem Wassermolekuel werden zwei Elektronen frei.

Die Energietraeger ATP und NADPH werden in dieser Phase gebildet und
anschliessend im Calvin-Zyklus verbraucht. Der Wirkungsgrad der Photosynthese
liegt unter natuerlichen Bedingungen bei etwa 1 bis 2 Prozent der eingestrahlten
Sonnenenergie.

Die optimale Temperatur fuer die Photosynthese liegt bei den meisten
C3-Pflanzen zwischen 20 und 30 Grad Celsius."""),
    ],
    "calvin_zyklus.pdf": [
        ("Der Calvin-Zyklus",
         """Der Calvin-Zyklus laeuft im Stroma der Chloroplasten ab. Er wurde nach Melvin
Calvin benannt, der dafuer 1961 den Nobelpreis fuer Chemie erhielt.

Das Schluesselenzym des Zyklus ist RuBisCO. Es gilt als das haeufigste Protein
der Erde. RuBisCO fixiert Kohlendioxid, indem es dieses an
Ribulose-1,5-bisphosphat bindet.

Fuer die Fixierung eines Molekuels Kohlendioxid werden drei Molekuele ATP und
zwei Molekuele NADPH benoetigt. Um ein Molekuel Glucose zu bilden, muss der
Zyklus sechsmal durchlaufen werden."""),
        ("Photorespiration",
         """RuBisCO hat eine Schwaeche: es kann statt Kohlendioxid auch Sauerstoff binden.
Dieser Vorgang heisst Photorespiration und verringert die Effizienz der
Photosynthese erheblich.

Bei hohen Temperaturen nimmt die Photorespiration zu. C4-Pflanzen wie Mais und
Zuckerrohr haben Mechanismen entwickelt, um dieses Problem zu umgehen. Sie
reichern Kohlendioxid vor der Fixierung an.

Anmerkung der Redaktion: In aelterer Literatur wird der Wirkungsgrad der
Photosynthese teilweise mit bis zu 6 Prozent angegeben. Diese Zahl bezieht sich
auf Laborbedingungen und nicht auf Freilandmessungen."""),
    ],
}

MD_DOC = """# Chunking-Strategien im RAG-Kontext

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
"""


def write_pdf(fn, sections):
    OUT.mkdir(exist_ok=True)
    c = canvas.Canvas(str(OUT / fn), pagesize=A4)
    w, h = A4
    for title, body in sections:
        y = h - 60
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y, title)
        y -= 28
        c.setFont("Helvetica", 10.5)
        for para in body.strip().split("\n\n"):
            for line in wrap(" ".join(para.split()), 92):
                c.drawString(50, y, line)
                y -= 14
            y -= 8
        c.showPage()
    c.save()
    print(f"-> docs/{fn}")


if __name__ == "__main__":
    for fn, secs in PDF_DOCS.items():
        write_pdf(fn, secs)
    OUT.mkdir(exist_ok=True)
    (OUT / "chunking_strategien.md").write_text(MD_DOC, encoding="utf-8")
    print("-> docs/chunking_strategien.md")
