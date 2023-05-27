if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--terms',
                      required=True,
                      help='patth to the txt-file with terms')
    parser.add_argument('--text',
                      required=True,
                      help='path to directory with texts-txt')
    parser.add_argument('--name',
                      required=True,
                      help='name of experiment')
    args = parser.parse_args()
    terms = args.terms
    txt = args.text
    corpus = args.name

    import os
    import torch
    from transformers import BertTokenizer, BertModel
    import pandas as pd

    regex_frequencies = {}
    regex_tf = {}
    regex_idf = {}
    regex_tfidf = {}
    regex_bm25 = {}
    regex_bert = {}

    def read_terms(my_terms):
        with open(my_terms, "r", encoding="utf-8") as regex_file:
            list_terms = [it.lower().strip() for it in regex_file]
            return list_terms

    list_terms = read_terms(terms)

    def read_text(my_text):
        all_txt_files = [filename for filename in os.listdir(my_text) if filename.endswith('txt')]
        all_docs = ''
        for txt_file in all_txt_files:
            with open(f'{my_text}{txt_file}') as f:
                txt_file_as_string = f.read()
            all_docs += txt_file_as_string
        return all_docs

    text = read_text(txt)

    # Calculate the embeddings for each term:
    def embedding_term(term, tokenizer, model):
        term_token = tokenizer.tokenize(term)
        term_input = tokenizer.encode_plus(term_token, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            return model(**term_input).last_hidden_state[:, 0, :]

    def calculate_similarity(list_terms, text):
        device = torch.device("cuda") if torch.cuda.is_available() else device = torch.device("cpu")
        # Initialize the BERT tokenizer and model for French:
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        model = BertModel.from_pretrained("bert-base-multilingual-uncased")
        cossim = {}
        # Calculate the BERT score for each term
        model.to(device)
        model.eval()
        with torch.no_grad():
            encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded_input["input_ids"].to(device)
            attention_mask = encoded_input["attention_mask"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state[:, 0, :]  # Use the [CLS] token embedding
            for term in list_terms:
                similarity = torch.cosine_similarity(embeddings, embedding_term(term, tokenizer, model))
                cossim[term] = similarity.item()
        return cossim

    cossim = calculate_similarity(list_terms, text)
    df = pd.DataFrame.from_dict({'term': list(cossim.keys()), 'score': list(cossim.values())})
    df.to_excel(f'results_BERT_{corpus}.xlsx', index = None, header= True)

