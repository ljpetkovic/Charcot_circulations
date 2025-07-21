import os
import re
import csv
import math

# === Paramètres de l'utilisateur ===
text_folder_path = "/Users/ljudmilapetkovic/Library/CloudStorage/Dropbox/SU/ObTIC/CHARCOT/corpus_echantillon/txt/txt_corpus_Autres"
# → Dossier contenant les fichiers texte (.txt) à analyser

regex_file_path = "/Users/ljudmilapetkovic/Library/CloudStorage/Dropbox/SU/ObTIC/CHARCOT/Charcot_circulations/concepts/liste_concepts_regex.txt"
# → Fichier contenant les regex (expressions régulières), une par ligne

output_file_path = "/Users/ljudmilapetkovic/Library/CloudStorage/Dropbox/SU/ObTIC/CHARCOT/Charcot_circulations/csv/output_autres_corpus_echantillon_210725.csv"
# → Fichier CSV de sortie pour enregistrer les résultats

# === Paramètres BM25 ===
k1 = 1.2  # Poids de la fréquence du terme
b = 0.75  # Contrôle de la normalisation par la longueur du document

# === Lecture des expressions régulières ===
with open(regex_file_path, "r", encoding="utf-8") as regex_file:
    regex_patterns = [re.compile(pattern.strip(), re.IGNORECASE) for pattern in regex_file if pattern.strip()]
    # → On compile chaque ligne non vide du fichier en regex insensible à la casse

    print(f" {len(regex_patterns)} regex chargées.")

# === Initialisation des structures de données ===
doc_stats = {}  # Dictionnaire des statistiques par document
doc_lengths = []  # Longueur (en mots) de chaque document
df_counts = {pattern.pattern: 0 for pattern in regex_patterns}  
# → df_counts[regex] = nombre de documents contenant cette regex

# === Traitement de chaque fichier texte du corpus ===
for filename in os.listdir(text_folder_path):
    if not filename.endswith(".txt"):
        continue  # On ignore les fichiers non .txt

    file_path = os.path.join(text_folder_path, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()  # Lecture du texte entier
        words = text.split()  # Découpe en mots (approximation grossière du token count)
        doc_len = len(words)  # Nombre de mots du document
        doc_lengths.append(doc_len)  # Ajout à la liste des longueurs

        regex_counts = {}  # Initialisation des comptes pour ce document

        for pattern in regex_patterns:
            count = len(pattern.findall(text))  # Nombre d'occurrences de la regex dans ce document
            if count > 0:
                name = pattern.pattern  # On utilise la chaîne de la regex comme identifiant
                regex_counts[name] = count  # On stocke la fréquence
                df_counts[name] += 1  # On incrémente le document frequency (df) pour cette regex

        doc_stats[filename] = {
            "length": doc_len,         # Longueur totale du document
            "regex_counts": regex_counts  # Fréquences des regex trouvées dans ce document
        }

# === Calcul de l'IDF (inverse document frequency) pour chaque regex ===
num_docs = len(doc_stats)  # Nombre total de documents
idf_scores = {
    regex: math.log(num_docs / df_counts[regex]) if df_counts[regex] > 0 else 0
    for regex in df_counts
}
# Formule : IDF = log(N / df) — plus une regex est fréquente, plus son IDF est faible

# === Calcul de la longueur moyenne des documents (utile pour BM25) ===
avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1

# === Calcul des scores TF-IDF et BM25 (par document, puis agrégés) ===
regex_scores = {}  # Dictionnaire pour stocker tous les résultats finaux

for regex in df_counts:
    tfidf_total = 0    # Score TF-IDF total (sommé sur tous les documents)
    bm25_total = 0     # Score BM25 total (sommé sur tous les documents)
    total_freq = 0     # Nombre total d’occurrences de la regex dans le corpus

    for doc, stats in doc_stats.items():
        f = stats["regex_counts"].get(regex, 0)  # f = nombre d'occurrences dans ce document
        if f == 0:
            continue  # Si le terme n'apparaît pas, on saute

        dl = stats["length"]  # Longueur du document
        idf = idf_scores[regex]  # IDF de la regex

        tf = f / dl  # Fréquence relative (terme/token)
        tfidf_total += tf * idf  # Ajout au score TF-IDF global de cette regex

        # Calcul du score BM25 pour ce document
        bm25 = idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avg_doc_len))))
        bm25_total += bm25  # Ajout au score BM25 global

        total_freq += f  # On ajoute les occurrences pour la fréquence totale

    regex_scores[regex] = {
        "Frequency": total_freq,
        "TF-IDF": tfidf_total,
        "BM25": bm25_total
    }

# === Normalisation des scores (entre 0 et 1) par min-max scaling ===
all_tfidf = [v["TF-IDF"] for v in regex_scores.values()]
all_bm25 = [v["BM25"] for v in regex_scores.values()]
min_tfidf, max_tfidf = min(all_tfidf), max(all_tfidf)
min_bm25, max_bm25 = min(all_bm25), max(all_bm25)

for scores in regex_scores.values():
    scores["Normalized TF-IDF"] = (
        (scores["TF-IDF"] - min_tfidf) / (max_tfidf - min_tfidf) if max_tfidf > min_tfidf else 0
    )
    scores["Normalized BM25"] = (
        (scores["BM25"] - min_bm25) / (max_bm25 - min_bm25) if max_bm25 > min_bm25 else 0
    )
# → On obtient ainsi des scores comparables visuellement (entre 0 et 1)

# === Export des résultats dans un fichier CSV ===
with open(output_file_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Regex", "Frequency", "TF-IDF", "BM25", "Normalized TF-IDF", "Normalized BM25"])

    for regex, scores in regex_scores.items():
        writer.writerow([
            regex,
            scores["Frequency"],
            round(scores["TF-IDF"], 4),
            round(scores["BM25"], 4),
            round(scores["Normalized TF-IDF"], 4),
            round(scores["Normalized BM25"], 4),
        ])

# Le fichier est maintenant écrit à l’emplacement donné
output_file_path
