import json

# Fonctions utilitaires
def charger_et_compter_tokens(chemin_fichier):
    with open(chemin_fichier, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sum(item["token_count"] for item in data)

# Fichiers sources
fichier_charcot = "output/tei_metadata_charcot.json"
fichier_autres = "output/tei_metadata_autres.json"

# Calculs
tokens_charcot = charger_et_compter_tokens(fichier_charcot)
tokens_autres = charger_et_compter_tokens(fichier_autres)
tokens_total = tokens_charcot + tokens_autres

# Affichage des résultats
print(f"Nombre total de tokens dans le corpus Charcot : {tokens_charcot}")
print(f"Nombre total de tokens dans le corpus Autres : {tokens_autres}")
print(f"Nombre total de tokens dans le corpus entier : {tokens_total}")