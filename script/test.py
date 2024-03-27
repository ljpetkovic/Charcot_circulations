import os
import re
import csv
import math
from bert_score import score
import time

start_time = time.time()

# Your code block here

# Configuration
CONFIG = {
    "text_folder_path": "/home/ljudmila/Dropbox/Humanistica2023/Charcot_circulations/input/test/",
    "regex_file_path": "/home/ljudmila/Dropbox/Humanistica2023/Charcot_circulations/regex.txt",
    "output_file_path": "/home/ljudmila/Dropbox/Humanistica2023/Charcot_circulations/csv/test.csv",
    "k1": 1.2,
    "b": 0.75,
    "lang": "fr",  # Language for BERT score
}

def read_regex_patterns(regex_file_path):
    with open(regex_file_path, "r", encoding="utf-8") as regex_file:
        return [re.compile(pattern.strip()) for pattern in regex_file]

def process_files(text_folder_path, regex_patterns):
    regex_frequencies = {}
    regex_tf = {}
    num_files = 0
    for filename in os.listdir(text_folder_path):
        num_files += 1
        with open(os.path.join(text_folder_path, filename), "r", encoding="utf-8") as text_file:
            text = text_file.read()
            for pattern in regex_patterns:
                regex_name = pattern.pattern
                regex_count = len(pattern.findall(text))
                if regex_count > 0:
                    regex_frequencies[regex_name] = regex_frequencies.get(regex_name, 0) + regex_count
                    regex_tf[regex_name] = regex_tf.get(regex_name, []) + [regex_count]
    return regex_frequencies, regex_tf, num_files

def calculate_scores(regex_frequencies, regex_tf, num_files, config):
    regex_idf = {}
    regex_tfidf = {}
    regex_bm25 = {}
    avg_doc_len = sum(len(tf_list) for tf_list in regex_tf.values()) / len(regex_tf)
    for regex_name, tf_list in regex_tf.items():
        idf = math.log(num_files / len(tf_list))
        regex_idf[regex_name] = idf
        tf = sum(tf_list)
        tf_idf_weight = tf * idf
        bm25_weight = idf * ((tf * (config["k1"] + 1)) / (tf + config["k1"] * (1 - config["b"] + config["b"] * (len(tf_list) / avg_doc_len))))
        regex_tfidf[regex_name] = tf_idf_weight
        regex_bm25[regex_name] = bm25_weight
    return regex_idf, regex_tfidf, regex_bm25

def calculate_bert_scores(text_folder_path, regex_frequencies, config):
    regex_bert = {regex_name: 0 for regex_name in regex_frequencies}
    for filename in os.listdir(text_folder_path):
        with open(os.path.join(text_folder_path, filename), "r", encoding="utf-8") as text_file:
            text = text_file.read()
            for regex_name in regex_frequencies.keys():
                if re.search(regex_name, text):
                    _, _, f1 = score([regex_name], [text], lang=config["lang"])
                    regex_bert[regex_name] += f1.item()
    return regex_bert

def write_output(output_file_path, regex_frequencies, regex_tfidf, regex_bm25, regex_bert):
    with open(output_file_path, "w", encoding="utf-8", newline="") as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(["Regex", "Frequency", "TF-IDF", "BM25", "BERT"])
        for regex_name, frequency in regex_frequencies.items():
            csv_writer.writerow([
                regex_name, frequency,
                round(regex_tfidf.get(regex_name, 0), 2),
                round(regex_bm25.get(regex_name, 0), 2),
                round(regex_bert.get(regex_name, 0), 2)
            ])

def main(config):
    regex_patterns = read_regex_patterns(config["regex_file_path"])
    regex_frequencies, regex_tf, num_files = process_files(config["text_folder_path"], regex_patterns)
    _, regex_tfidf, regex_bm25 = calculate_scores(regex_frequencies, regex_tf, num_files, config)
    regex_bert = calculate_bert_scores(config["text_folder_path"], regex_frequencies, config)
    write_output(config["output_file_path"], regex_frequencies, regex_tfidf, regex_bm25, regex_bert)

# Don't forget to call main
if __name__ == "__main__":
    main(CONFIG)




end_time = time.time()
print("Execution time: {:.2f} seconds".format(end_time - start_time))

