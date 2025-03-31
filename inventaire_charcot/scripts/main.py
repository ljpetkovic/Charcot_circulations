import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt

def extraire_annees_depuis_tei(repertoire):
    annees = []
    for nom_fichier in os.listdir(repertoire):
        chemin = os.path.join(repertoire, nom_fichier)
        if os.path.isfile(chemin) and chemin.endswith(".xml"):
            try:
                tree = ET.parse(chemin)
                root = tree.getroot()
                for date in root.findall(".//date"):
                    when = date.get("when")
                    if when and re.match(r"^\d{4}$", when):
                        annees.append(int(when))
                        break  # une année par fichier
            except Exception as e:
                print(f"[!] Erreur avec {nom_fichier} : {e}")
    return annees

def compter_par_annee(liste_annees):
    compteur = {}
    for annee in liste_annees:
        compteur[annee] = compteur.get(annee, 0) + 1
    return compteur

def main(rep_charcot, rep_autres):
    # Extraction
    annees_charcot = extraire_annees_depuis_tei(rep_charcot)
    annees_autres = extraire_annees_depuis_tei(rep_autres)

    # Comptage
    compte_charcot = compter_par_annee(annees_charcot)
    compte_autres = compter_par_annee(annees_autres)

    # Fusion dans un DataFrame
    toutes_annees = sorted(set(compte_charcot) | set(compte_autres))
    df = pd.DataFrame({
        "Année de publication": toutes_annees,
        "Nombre d'ouvrages du corpus Charcot": [compte_charcot.get(a, 0) for a in toutes_annees],
        "Nombre d'ouvrages du corpus Autres": [compte_autres.get(a, 0) for a in toutes_annees],
    })

    # Export CSV
    df.to_csv("output/distribution_ouvrages.csv", index=False, encoding="utf-8")
    print("✅ CSV enregistré : distribution_ouvrages.csv")

    # Graphique
    plt.figure(figsize=(14, 6))
    largeur = 0.4
    x = range(len(df))

    plt.bar(x, df["Nombre d'ouvrages du corpus Charcot"], width=largeur, color='cornflowerblue', label="Charcot")
    plt.bar(x, df["Nombre d'ouvrages du corpus Autres"], width=largeur, bottom=df["Nombre d'ouvrages du corpus Charcot"], color='tomato', label="Autres")

    plt.xticks(ticks=x, labels=df["Année de publication"], rotation=90)
    plt.xlabel("Année de publication")
    plt.ylabel("Nombre d'ouvrages")
    # plt.title("Distribution des ouvrages par année")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/distribution_ouvrages.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # Remplacer par les chemins réels
    main("corpus_Charcot", "corpus_Autres")
