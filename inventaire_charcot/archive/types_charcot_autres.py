import os
import json
import xml.etree.ElementTree as ET
import spacy

# Chargement du modèle spaCy
nlp = spacy.load("fr_core_news_lg")
nlp.max_length = 10_000_000  # Ajusté pour traiter de gros volumes

# Dossiers des deux corpus
corpora = {
    'corpus_Charcot': 'corpus_Charcot',
    'corpus_Autres': 'corpus_Autres'  
}

def extract_text_from_corpus(folder_path):
    full_text = ''
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                full_text += ' '.join(root.itertext())
            except Exception as e:
                print(f"Erreur dans le fichier {file_path}: {e}")
    return full_text

def compute_unique_types(text):
    types_set = set()
    chunk_size = 500_000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    for chunk in chunks:
        tokens = nlp(chunk)
        for token in tokens:
            if not token.is_space:
                types_set.add(token.lemma_.lower())
    return len(types_set)

# Traitement de chaque corpus
results = {}
for corpus_name, folder_path in corpora.items():
    text = extract_text_from_corpus(folder_path)
    type_count = compute_unique_types(text)
    results[corpus_name] = {'type_count': type_count}

# Sauvegarde dans un fichier JSON
with open('types_charcot_autres.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
