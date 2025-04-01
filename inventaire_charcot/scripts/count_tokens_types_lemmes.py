import json
from collections import defaultdict

# Charger les données depuis le fichier JSON
with open("tei_metadata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Initialiser les compteurs pour chaque corpus et globalement
totals = {
    "corpus_Charcot": {"tokens": 0, "types": 0, "lemmas": 0},
    "corpus_Autres": {"tokens": 0, "types": 0, "lemmas": 0},
    "total": {"tokens": 0, "types": 0, "lemmas": 0}
}

# Ajouter les compteurs individuels pour chaque corpus
for entry in data:
    corpus = entry["corpus"]
    if corpus in totals:
        totals[corpus]["tokens"] += entry["token_count"]
        totals[corpus]["types"] += entry["word_type_count"]
        totals[corpus]["lemmas"] += entry["lemma_type_count"]
        totals["total"]["tokens"] += entry["token_count"]
        totals["total"]["types"] += entry["word_type_count"]
        totals["total"]["lemmas"] += entry["lemma_type_count"]

totals
