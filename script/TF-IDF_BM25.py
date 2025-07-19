import os
import re
import csv
import math

# Define the path to the folder of UTF-8 text files and the path to the regex file
text_folder_path = "/Users/ljudmilapetkovic/Library/CloudStorage/Dropbox/SU/ObTIC/CHARCOT/corpus_echantillon/txt/txt_corpus_Autres"
regex_file_path = "/Users/ljudmilapetkovic/Library/CloudStorage/Dropbox/SU/ObTIC/CHARCOT/Charcot_circulations/concepts/liste_concepts_regex.txt"

# Define a dictionary to hold regex frequencies, TF, IDF, TF-IDF, and BM25 values
regex_frequencies = {}
regex_tf = {}
regex_idf = {}
regex_tfidf = {}
regex_bm25 = {}

# Read in the regex file and compile each regex pattern (case-insensitive)
with open(regex_file_path, "r", encoding="utf-8") as regex_file:
    regex_patterns = [re.compile(pattern.strip(), re.IGNORECASE) for pattern in regex_file]
    print(f"Read {len(regex_patterns)} regex patterns from {regex_file_path}")

# Loop through each file in the text folder and count the frequency of each regex pattern
num_files = 0
total_words = 0  # Track total words across documents

for filename in os.listdir(text_folder_path):
    file_path = os.path.join(text_folder_path, filename)

    try:
        with open(file_path, "r", encoding="utf-8") as text_file:
            text = text_file.read()
            num_files += 1
            total_words += len(text.split())  # Count words for avg doc length

            for pattern in regex_patterns:
                regex_count = len(re.findall(pattern, text))
                regex_name = pattern.pattern
                regex_frequencies[regex_name] = regex_frequencies.get(regex_name, 0) + regex_count
                
                if regex_count > 0:
                    regex_tf[regex_name] = regex_tf.get(regex_name, []) + [regex_count]

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

# Calculate the IDF values for each regex pattern
for regex_name, tf_list in regex_tf.items():
    doc_count = len(tf_list)
    if doc_count > 0:
        regex_idf[regex_name] = math.log(num_files / doc_count)
    else:
        regex_idf[regex_name] = 0  # Avoid division by zero

# Calculate the TF-IDF and BM25 weights for each regex pattern
# La formule utilise deux paramètres réglables, k1 et b, 
# pour contrôler l’impact de la fréquence du terme 
# et la normalisation de la longueur du document sur le score.
k1 = 1.2
b = 0.75
avg_doc_len = total_words / num_files if num_files > 0 else 1  # Avoid division by zero

for regex_name, tf_list in regex_tf.items():
    idf = regex_idf.get(regex_name, 0)
    doc_len = sum(tf_list)
    tf = sum(tf_list)

    tf_idf_weight = tf * idf
    bm25_weight = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len))))

    regex_tfidf[regex_name] = tf_idf_weight
    regex_bm25[regex_name] = bm25_weight

# Compute min and max values AFTER all regex calculations
max_tfidf = max(regex_tfidf.values()) if regex_tfidf else 1
min_tfidf = min(regex_tfidf.values()) if regex_tfidf else 0

max_bm25 = max(regex_bm25.values()) if regex_bm25 else 1
min_bm25 = min(regex_bm25.values()) if regex_bm25 else 0

# Normalize TF-IDF and BM25 scores using Min-Max scaling
regex_tfidf_norm = {
    regex: (value - min_tfidf) / (max_tfidf - min_tfidf) if max_tfidf > min_tfidf else 0
    for regex, value in regex_tfidf.items()
}

regex_bm25_norm = {
    regex: (value - min_bm25) / (max_bm25 - min_bm25) if max_bm25 > min_bm25 else 0
    for regex, value in regex_bm25.items()
}

# Write the regex frequencies, TF-IDF, Normalized TF-IDF, and BM25 weights to a CSV file
output_file_path = "/Users/ljudmilapetkovic/Library/CloudStorage/Dropbox/SU/ObTIC/CHARCOT/Charcot_circulations/csv/output_autres_corpus_echantillon_100725.csv"

with open(output_file_path, "w", encoding="utf-8", newline="") as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["Regex", "Frequency", "TF-IDF", "BM25", "Normalized TF-IDF", "Normalized BM25"])
    
    for regex_name, frequency in regex_frequencies.items():
        tf_idf = round(regex_tfidf.get(regex_name, 0), 2)
        bm25 = round(regex_bm25.get(regex_name, 0), 2)
        tf_idf_norm = round(regex_tfidf_norm.get(regex_name, 0), 4)  # Rounded to 4 decimal places
        bm25_norm = round(regex_bm25_norm.get(regex_name, 0), 4)

        csv_writer.writerow([regex_name, frequency, tf_idf, bm25, tf_idf_norm, bm25_norm])

print(f"Wrote {len(regex_frequencies)} regex frequencies, normalized TF-IDF, and normalized BM25 weights to {output_file_path}")

