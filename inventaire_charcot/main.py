import os
import csv
import xml.etree.ElementTree as ET
import re

def extraire_dates_sans_namespace(repertoire):
    resultats = []

    for nom_fichier in os.listdir(repertoire):
        chemin = os.path.join(repertoire, nom_fichier)
        if os.path.isfile(chemin) and chemin.endswith('.xml'):
            annees = []
            try:
                tree = ET.parse(chemin)
                root = tree.getroot()
                # On cherche les balises <date> sans namespace
                for date in root.findall('.//date'):
                    when = date.get('when')
                    if when and re.match(r'^\d{4}$', when):
                        annees.append(when)
                if annees:
                    resultats.append((nom_fichier, ', '.join(annees)))
            except ET.ParseError:
                print(f"[!] Erreur de parsing : {chemin}")
            except Exception as e:
                print(f"[!] Problème avec {chemin} : {e}")
    return resultats

def enregistrer_csv(donnees, nom_fichier_csv):
    with open(nom_fichier_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Nom du fichier", "Année(s) extraite(s)"])
        writer.writerows(donnees)
    print(f"✅ Résultats enregistrés dans : {nom_fichier_csv}")

def main(rep1, rep2):
    dates1 = extraire_dates_sans_namespace(rep1)
    enregistrer_csv(dates1, "output/dates_repertoire1.csv")

    dates2 = extraire_dates_sans_namespace(rep2)
    enregistrer_csv(dates2, "output/dates_repertoire2.csv")

if __name__ == "__main__":
    # Remplacer ces chemins par les bons
    main("corpus_Charcot", "corpus_Autres")
