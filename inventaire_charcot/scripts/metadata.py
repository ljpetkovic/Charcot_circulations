import os
import json
import xml.etree.ElementTree as ET
import spacy
from collections import defaultdict

# Load spaCy model for tokenization
nlp = spacy.load("fr_core_news_lg")
nlp.max_length = 3_000_000

# Directories à traiter
folders = ['corpus_Charcot', 'corpus_Autres']

def extract_info_from_xml(file_path, corpus_name):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extraire le <teiHeader> sans tenir compte du namespace
    header = next((child for child in root if child.tag.endswith('teiHeader')), None)

    authors, title, date = [], None, None

    if header is not None:
        file_desc = header.find('fileDesc')
        if file_desc is not None:
            title_stmt = file_desc.find('titleStmt')
            if title_stmt is not None:
                authors = [el.text.strip() for el in title_stmt.findall('author') if el.text]
                title_el = title_stmt.find('title')
                if title_el is not None and title_el.text:
                    title = title_el.text.strip()

        profile_desc = header.find('profileDesc')
        if profile_desc is not None:
            creation = profile_desc.find('creation')
            if creation is not None:
                date_el = creation.find('date')
                if date_el is not None:
                    date = date_el.get('when') or (date_el.text.strip() if date_el.text else None)

    # Tokenisation et comptage
    text_content = ' '.join(root.itertext())
    chunk_size = 300_000
    chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]

    token_count = 0
    word_types = set()
    lemma_types = set()

    for chunk in chunks:
        doc = nlp(chunk)
        for token in doc:
            if not token.is_space:
                token_count += 1
                if token.is_alpha:
                    word_types.add(token.text.lower())
                    lemma_types.add(token.lemma_.lower())

    return {
        'corpus': corpus_name,
        'file': os.path.basename(file_path),
        'authors': authors,
        'title': title,
        'date': date,
        'token_count': token_count,
        'word_type_count': len(word_types),
        'lemma_type_count': len(lemma_types)
    }

# Extraction principale
all_data = []
for folder in folders:
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder, filename)
                info = extract_info_from_xml(file_path, folder)
                all_data.append(info)

# Écriture dans le fichier JSON
with open('tei_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)
