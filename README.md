# master_EmbForLiTtext

## Attention:
Rekurrentes Netzwerk zur Klassifizierung von Textgruppen mit Attention.
Im Code anzupassen:
testdata: Pfad zu einer Datei mit Testdaten
traindata: Pfad zu einer Datei mit Trainingsdaten

# Format der Eingabe:
Je ein Segement pro Zeile, tokenisiert und durch \<tab\> getrennt. Zusätzlich eine Spalte für das Label.

tok1<tab>tok2<tab>tok3<tab>...<tab>tok200<tab>label
  
das<tab>ist<tab>ein<tab>...<tab>Test 1
das<tab>ist<tab>ein<tab>...<tab>Test 2

