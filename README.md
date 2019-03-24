# master_EmbForLiTtext

## Attention:
Rekurrentes Netzwerk zur Klassifizierung von Textgruppen mit Attention.
Im Code anzupassen:
testdata: Pfad zu einer Datei mit Testdaten
traindata: Pfad zu einer Datei mit Trainingsdaten

### Format der Eingabe:
Je ein Segement pro Zeile, tokenisiert und durch \<tab\> getrennt. <br/>Zusätzlich eine Spalte für das Label.

tok1\<tab\>tok2\<tab\>tok3\<tab\>...\<tab\>tok200\<tab\>label<br/><br/>
  
das\<tab\>ist\<tab\>ein\<tab\>...\<tab\>Test 1<br/>
das\<tab\>ist\<tab\>ein\<tab\>...\<tab\>Test 2<br/>

## Zeta

run.py für den vollen Umfang der Parametersuche
run_batch_adv.py für die Standardeinstellung (schneller)

Zu übergebende Parameter:

    case = id des testcase
    focus = Eigenschaft der Gegengruppe in den Metadaten
    reihenname = Eigenschaft der Gegengruppe in den Metadaten
    counter = Eigenschaft der Gegengruppe in den Metadaten
    meta = Pfad zu Metadaten
    dtm_path = Pfad zu sequentieller DTM (Eine Spalte mit Counter für Segmente a 250 Token)
    dtm_single_path = Pfad zu DTM Ordner
    seg_size = Segmentgröße für Zeta
    logaddition = 0.21
    ft_model = fasttext model
    ft_model_trained = fasttext model
    w2v_model = w2v model
    stoplist = Liste mit Stopwords
    t_mode = False
    metric = (euclidean|cosine|manhattan)
    random_state = 
    run_num = id der ausgabedatei
    meth = (MS|Brich|AP) für Mean Shift, Birch und Afinity Propagation
    calc = (True|False) True: mean als finaler Rechenschritt, False: min
    
    
