# Tracking the circulation of Jean-Martin Charcot’s medical discourse: first observations

This repository was created in order to make available the OCRed corpus of the [Charcot Collection](https://patrimoine.sorbonne-universite.fr/collection/Fonds-Charcot)  and Python script for identifying relevant medical concepts in them according to three different weighting measures.

---

## Corpus

(*courtesy of the Digital Heritage Library of Sorbonne University*)

The Charcot Collection corpus (`corpus`) is divided into two sub-corpora:

1. `corpus_Charcot`: Charcot's writings as a co-author
2. `corpus_Autres`: writings of Charcot's collaborators and successors only

---

## Concepts

* the list of relevant concepts present in the Charcot's medical discourse, selected from the [index](https://patrimoine.sorbonne-universite.fr/viewer/3468/?offset=1#page=501&viewer=picture&o=&n=0&q=) of an edition of the complete works of Charcot
* `concepts/concepts.xml`: the list of mentioned concepts converted from the PDF to the XML file using the converter [Teinte](https://obtic.huma-num.fr/teinte/)

---

## Script

The `script/tfidf_bm25_bert.py` Python script generates the CSV file(s) with the scores for each concept across the two corpora, based on the TF-IDF, BM-25 and BERTScore metrics. By discarding generic terms, like *cerveau* (brain), *os* (bone), etc., we have retained a restricted list of terms or expressions popularised by Charcot, such as *hystérie* (hysteria), *sclérose latérale amyotrophique* (amyotrophic lateral sclerosis), etc. 

---

## Cite

Ljudmila Petkovic, Motasem Alrahabi, Glenn Roe, *Tracking the circulation of Jean-Martin Charcot’s medical discours: first observations*, 2023, Github, https://github.com/ljpetkovic/Charcot_circulations

---

## Licence

![68747470733a2f2f692e6372656174697665636f6d6d6f6e732e6f72672f6c2f62792f322e302f38387833312e706e67](https://user-images.githubusercontent.com/56683417/115237678-2150d080-a11d-11eb-903e-5a26587e12e1.png)

