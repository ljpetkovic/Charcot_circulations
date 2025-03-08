import os
import re
import csv
import math
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd #j'ai rajouté ma librairie préferée, elle est plus rapide que 'csv' pour génerer les fichiers
# et surtout ici je préfère génerer les fichiers excel pour facilité la lecture

text_folder_path = "C:/Users/alrahabi/Documents/workspace/Python/TFIDF_BM25_BERT/input/"
regex_file_path = "C:/Users/alrahabi/Documents/workspace/Python/TFIDF_BM25_BERT/regex.txt"
corpus = 'Charcot' #changer sur 'autres', si l'autre corpus est analysé

regex_frequencies = {}
regex_tf = {}
regex_idf = {}
regex_tfidf = {}
regex_bm25 = {}
regex_bert = {}

'''
with open(regex_file_path, "r", encoding="utf-8") as regex_file:
    regex_patterns = [re.compile(pattern.strip()) for pattern in regex_file]
    print(f"Read {len(regex_patterns)} regex patterns from {regex_file_path}")'''
#j'ai gardé ton bloc de code en commentaire ci-dessus, mais il ne lit pas les données correctement.
#Si tu affiches le resultat de la lecture, tu reçois une liste de fonction re.compile comme celui-ci:
#print(regex_patterns) ==> [re.compile('Achromatopsie(s)? hystérique(s)?'), re.compile('Amblyopie(s)? hystérique(s)?'), re.compile('Amyotrophie(s)? protopathique(s)?')]
#le type des données, est 're.Pattern': print(type(regex_patterns[0])) ==> <class 're.Pattern'>
#et ici, pour toutes les trois mesures tfidf, BM25 et BERT (cosinus) nous avons besoin d'analyser les chaines de caractères (type == str)
#pour cette raison j'ai changé la fonction qui lit les données:
#j'ai enlevé les signes de ponctuation dans la liste de terms, j'ai extrait une liste de term en singulier et en pluriel
#et j'ai sommé les scores pour les occurances du même term en singulier et en pluriel.
#A la base de ces valeurs j'ai calculer les tfidf et BM25
#Voilà ma fonction de lecture et nettoyage de la liste de terms. Je n'ai pas eu beaucoup de temps, donc le code est loin d'être idéal, mail il marche.

def read_terms(regex_file_path):
    with open(regex_file_path, "r", encoding="utf-8") as regex_file:
        list_all = [it.lower().strip() for it in regex_file]
        regex_patterns = [pattern.strip().replace('(s)?', 's').strip() for pattern in list_all]
        print(f"Read {len(regex_patterns)} regex patterns from {regex_file_path}")
        regex_split = [' '.join([it.strip().replace('(s)?', '') for it in item.split(' ')]) if item.strip().endswith(
            's)?') else item for item in list_all]
        regex_patterns_double = [x for x in list(set(regex_patterns + regex_split)) if x != '']
        return regex_patterns_double, regex_patterns

regex_patterns_double, regex_patterns = read_terms(regex_file_path)

#j'ai écrit une fonction pour lire le text. Normalement, on aurait du lemmatiser le text, mais je n'ai pas eu temps de faire la lemmatisation,
#donc j'ai au moins enlevé la ponctuation et mis tout en minuscules
def read_text(text_folder_path):
    all_txt_files = [filename for filename in os.listdir(text_folder_path) if filename.endswith('txt')]
    all_docs = ''
    for txt_file in all_txt_files:
        with open(f'{text_folder_path}{txt_file}') as f:
            txt_file_as_string = f.read()
        all_docs += txt_file_as_string.lower()
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        text = all_docs.replace(i, ' ')
    return text

text = read_text(text_folder_path)
len_text = len([x for x in text.split()])

def create_regex_dict(regex_patterns_double):
    #je calucule le nombre d'occurances pour chaque term:
    regex_double = {}
    for pattern in regex_patterns_double:
        l = [x for x in text.split()]
        regex_double[pattern] = l.count(pattern)
    #je reunis les nombres d'occurances pour le singulier et pour le pluriel et je nettoye les signes inutiles:
    regex_count = {}
    for k, v in regex_double.items():
        if ' ' in regex_name:
            regex_name = ' '.join([x[:-1] if x.endswith('s') else x for x in k.split(' ')])
        elif k.endswith('s') and k[:-1] in list(regex_count.keys()):
            regex_name = k[:-1]
        else:
            regex_name = k
        regex_name = regex_name.replace('(', '').replace(')', '').replace('(x)?', 'x').replace('|des', '').replace(
            '(es)?', 'es').replace('|aux', '').replace('?', '').replace('|maux', '')
        if regex_name not in regex_tf:
            regex_count[regex_name] = v
        else:
            regex_count[regex_name] += v
    return regex_count

#j'ai créé le dictionnaire des occurances des terms
regex_count = create_regex_dict(regex_patterns_double)

'''
# Loop through each file in the text folder and count the frequency of each regex pattern
num_files = 0
for filename in os.listdir(text_folder_path):
    num_files += 1
    file_path = os.path.join(text_folder_path, filename)
    print(f"Processing file {file_path}")
    with open(file_path, "r", encoding="utf-8") as text_file:
        text = text_file.read()
        for pattern in regex_patterns:
            regex_count = len(pattern.findall(text)) ==> tu as besoin de trouver toutes les occurances du term, mais cette méthode
            # ne prend pas en compte les espace, donc pour le term 'os' elle te trouvera beaucoup plus d'occurences qu'il te faut,
            # parce qu'elle contera aussi le fragment 'os' dans 'fibrOSe', OStéopathe' etc.
            regex_name = pattern.pattern ==> je ne comprends pas cette ligne
            regex_frequencies[regex_name] = regex_frequencies.get(regex_name, 0) + regex_count
            if regex_count > 0:
                regex_tf[regex_name] = regex_tf.get(regex_name, []) + [regex_count]
                
==> ce bout de code te construit un dictionnaire avec les terms comme clés et comme valeurs des listes d'un seul nombre = nb d'occurrence du term:
==> si tu l'affiches, tu verras: ['os':[7], 'main': [5], 'psychose': [8] etc.]
==> pour cette raison dans la ligne 115 l'expression len(tf_list) sera toujours = 1

# Calculate the IDF values for each regex pattern:
for regex_name, tf_list in regex_tf.items():
    idf = math.log(num_files / len(tf_list)) ==> ici il faut enlever le 'len', sinon le denominateur est toujours égal à 1
    regex_idf[regex_name] = idf

# Calculate the TF-IDF and BM25 weights for each regex pattern:
k1 = 1.2
b = 0.75
avg_doc_len = sum(len(tf_list) for tf_list in regex_tf.values()) / len(regex_tf) ==> on a seulement un document dans le corpus
for regex_name, tf_list in regex_tf.items():
    idf = regex_idf[regex_name]
    doc_len = len(tf_list) ==> on a seulement un document dans le corpus, donc doc_len == avg_len
    avg_len = avg_doc_len
    tf = sum(tf_list) ==> ? ça fait: sum([3]), si le nb d'occurences == 3, par exemple
    tf_idf_weight = tf * idf
    bm25_weight = idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len))))
    regex_tfidf[regex_name] = tf_idf_weight
    regex_bm25[regex_name] = bm25_weight
'''

def calcul_tfidf(regex_count, len_text):
    regex_tfidf = {}
    for k, v in regex_count.items():
        if v != 0:
            regex_tfidf[k] = v / len_text * math.log(1/2)
        else:
            regex_tfidf[k] = 0
    return regex_tfidf
# alors ici, comme vous avez analysé apart le text de Charcot et autres documents, donc un seul document chaque fois,
# le nb de documents =1 et le nombre de documents où le term se trouve =1, si le term est présent dans le document.
#j'ai normalisé les valeurs de tfidf pour que ce soit possible de les comparer avec les résultats de la similarité cosinus (entre -1 et 1)

def normalize(regex_tfidf):
    mi = max(list(regex_tfidf.values()))
    ma = min(list(regex_tfidf.values()))
    regex_norm = {}
    for k, v in regex_tfidf.items():
        regex_norm[k] = 2*(v-mi)/(ma-mi)-1

regex_tdidf = normalize(calcul_tfidf(regex_count, len_text))
df = pd.DataFrame.from_dict({'term': list(regex_tfidf.keys()), 'score': list(regex_tfidf.values())})
df.to_excel(f'results_TF_IDF_{corpus}.xlsx', index = None, header= True)

def calcul_bm25(regex_count, len_text):
    k1 = 1.2
    b = 0.75
    for k, v in regex_count.items():
        if v != 0:
            idf = v / len_text * math.log(1 / 2)
        else:
            idf = 0
        regex_bm25[k] = idf * v / len_text * (k1 + 1) / (v /len_text + k1 * (1 - b + b)) #j'ai laissé les valeurs ici pour que ce soit plus explicite, mais normalement les coefficients 'b' s'enlèvent
        #dnas notre cas d = avdl, donc ces valeurs s'enlévent mathématiquement
    return regex_bm25

regex_bm25 = normalize(calcul_bm25(regex_count, len_text))
df = pd.DataFrame.from_dict({'term': list(regex_bm25.keys()), 'score': list(regex_bm25.values())})
df.to_excel(f'results_bm25_{corpus}.xlsx', index = None, header= True)

# Initialize the BERT tokenizer and model for French:
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")
#hier, j'ai utilisé comme toi le modèle 'cased', mais comme j'ai mis tout le texte en minuscules, il faudra utiliser ici le modèle 'uncased'

# Calculate the embeddings for each regex pattern:
def embeddings_pattern(regex_name):
    regex_tokens = tokenizer.tokenize(regex_name)
    regex_input = tokenizer.encode_plus(regex_tokens, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        return model(**regex_input).last_hidden_state[:, 0, :]
    #regex_embeddings[regex_name.pattern] = regex_output ==> je ne comprends pas cette ligne. Pourquoi .pattern?
    #j'ai transformé ton bout de code en fonction qui retourne l'embedding du term
    # le reste du code dans cette fonction etait très bon, surtout la decision de calculer la similarité cosinus à la base du premier vecteur [CLS] du term
    #j'ai également essayé de faire la sim cos sur un vecteur moyenneé de tous les sous-mots du term, mais cette méthode ne marche pas bien:
    #regex_embeddings[regex_name] = torch.mean(regex_output, dim=1)
    #le modèle BERT resume très bien les informations dans le premier vecteur [CLS] du mot ou de la phrase

# Calculate the BERT score for each regex pattern:
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
with torch.no_grad():
    '''
    for regex_name in regex_frequencies.keys():
        regex_bert[regex_name] = 0'''
    regex_bert = {} # je prefère d'initialiser un dictionnaire avec une seule ligne de code et sans boucle (time-consuming),
    # mais ta façon d'initialisation est aussi correcte
    '''for filename in os.listdir(text_folder_path):
        file_path = os.path.join(text_folder_path, filename)
        print(f"Processing file {file_path}")
        with open(file_path, "r", encoding="utf-8") as text_file:
            text = text_file.read()'''
    #comme j'avais déjà preparé mon texte (enlevé la ponctuation, lower, see: def read_text) et je l'ai stocké dans la variable 'text',
    # je l'utilise directement ici sans boucler sur les fichiers:
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_input["input_ids"].to(device)
    attention_mask = encoded_input["attention_mask"].to(device)
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token embedding
    # embeddings = torch.mean(outputs.last_hidden_state[:, :, :], dim=1) ==> j'ai essayé de nouveau le vecteur moyennée et
    # j'ai vu encore une fois que BERT est super puissant pour garder les informations dans le premier vecteur, même pour un long texte
    # c'est de la magie
    '''
            for regex_name in regex_frequencies.keys():
                if re.search(regex_name, text): ==> cette expression n'a pas de sens ici, parce que avec BERT on n'a pas besoin de trouver 
                ==> des terms exactes dans notre texte. On regarde juste la proximité sémantique du terme au contenu du texte.
                    similarity = torch.cosine_similarity(embeddings, regex_embeddings[regex_name])
                    regex_bert[regex_name] += similarity.item()''' #==> pourquoi sommer les valeurs de la similarité cosinus?
    #==> la valeur de sim/cos est entre -1 et 1, donc on ne peut pas les sommer.

    for regex_name in regex_tfidf.keys():
            similarity = torch.cosine_similarity(embeddings, embeddings_pattern(regex_name))
            regex_bert[regex_name] = similarity.item()

df = pd.DataFrame.from_dict({'term': list(regex_bert.keys()), 'score': list(regex_bert.values())})
df.to_excel('results_BERT_Charcot.xlsx', index = None, header= True)

# Write the regex frequencies, TF, IDF, BM25, and BERT scores to a CSV file:
output_file_path = "C:/Users/alrahabi/Documents/workspace/Python/TFIDF_BM25_BERT/output.csv"
with open(output_file_path, "w", encoding="utf-8", newline="") as output_file:
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(["Regex", "Frequency", "TF-IDF", "BM25", "BERT"])
    for regex_name, frequency in regex_frequencies.items():
        tf_idf = round(regex_tfidf.get(regex_name, 0), 2)
        bm25 = round(regex_bm25.get(regex_name, 0), 2)
        bert_score = round(regex_bert.get(regex_name, 0), 2)
        csv_writer.writerow([regex_name, frequency, tf_idf, bm25, bert_score])
    print(f"Wrote {len(regex_frequencies)} regex frequencies, TF-IDF, BM25, and BERT scores to {output_file_path}")
