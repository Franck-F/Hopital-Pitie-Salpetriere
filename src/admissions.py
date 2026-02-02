import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# --- CONFIGURATION (Basée sur Chiffres Clés 2015 Pitié-Salpêtrière) ---

# 1. ORIGINE GEOGRAPHIQUE (Page 14)
ORIGINES = {
    "Paris (75)": 0.35, "Val-de-Marne (94)": 0.13, "Hauts-de-Seine (92)": 0.06,
    "Seine-Saint-Denis (93)": 0.09, "Seine-et-Marne (77)": 0.06,
    "Autres IDF": 0.15, "Hors IDF / Étranger": 0.16
}

# 2. STATISTIQUES PAR PÔLE (Pages 4, 6, 8 + Flux Urgences)
POLES_V4 = {
    "Urgences (Passage court)": {
        "weight": 114133, "dms": 0, "ratio_ambu": 1.0, 
        "mode_entree": "Urgences",
        "motifs": ["Traumatologie légère", "Douleur abdominale non spécifiée", "Malaise vagal", 
                   "Plaie superficielle", "Infection ORL", "Gastro-entérite légère", "Crise d'angoisse"]
    },
    "Chirurgie": {
        "weight": 34356, "dms": 4.39, "ratio_ambu": 0.355, 
        "mode_entree": "Programmé",
        "motifs": ["Fracture fémur", "Appendicite", "Prothèse hanche", "Hernie inguinale", "Chirurgie viscérale"]
    },
    "PRAGUES (Réa/Pneumo/Admis Urgences)": {
        "weight": 17265, "dms": 3.25, "ratio_ambu": 0.151,
        "mode_entree": "Urgences", 
        "motifs": ["Insuffisance respiratoire", "Pneumonie", "Grippe sévère", "Bronchiolite", "Embolie pulmonaire"]
    },
    "MSN (Neuro/Psy)": {
        "weight": 25014, "dms": 6.20, "ratio_ambu": 0.731,
        "mode_entree": "Mixte",
        "motifs": ["AVC Ischémique", "Sclérose en plaques", "Épilepsie", "Parkinson", "Dépression sévère"]
    },
    "Chirurgie Neuro-sensorielle": {
        "weight": 10633, "dms": 4.07, "ratio_ambu": 0.318,
        "mode_entree": "Programmé",
        "motifs": ["Cataracte", "Otite chronique", "Tumeur hypophysaire", "Glaucome", "Chirurgie ORL"]
    },
    "Coeur-Metabolisme": {
        "weight": 25355, "dms": 4.21, "ratio_ambu": 0.427,
        "mode_entree": "Mixte",
        "motifs": ["Insuffisance cardiaque", "Infarctus", "Diabète déséquilibré", "Troubles du rythme", "Obésité"]
    },
    "Pôle 31 (Infectieux/Rhumato)": {
        "weight": 21231, "dms": 8.19, "ratio_ambu": 0.587,
        "mode_entree": "Mixte",
        "motifs": ["VIH", "Polyarthrite rhumatoïde", "Lupus", "Spondylarthrite", "Maladie de Lyme"]
    },
    "ORPHé (Onco/Hémato)": {
        "weight": 36959, "dms": 7.57, "ratio_ambu": 0.914,
        "mode_entree": "Programmé",
        "motifs": ["Chimiothérapie", "Tumeur sein", "Leucémie", "Lymphome", "Myélome"]
    },
    "Gériatrie (Charles Foix)": {
        "weight": 6303, "dms": 8.76, "ratio_ambu": 0.256,
        "mode_entree": "Mixte",
        "motifs": ["Chute", "Dénutrition", "Alzheimer", "Syndrome confusionnel", "Perte d'autonomie"]
    }
}

# --- FONCTION DE GENERATION ---
def generate_complete_dataset_2024():
    # Normalisation des poids
    total_vol = sum(p["weight"] for p in POLES_V4.values())
    poles_list = list(POLES_V4.keys())
    poles_weights = [POLES_V4[p]["weight"] / total_vol for p in poles_list]
    
    origins_list = list(ORIGINES.keys())
    origins_weights = list(ORIGINES.values())

    data = []
    start_date = datetime(2024, 1, 1)
    global_id = 0
    
    print("Génération en cours (365 jours)...")
    
    for day_offset in range(365):
        current_date = start_date + timedelta(days=day_offset)
        
        # Facteur Saisonnier (Hiver +25%, Été -20%)
        month = current_date.month
        if month in [12, 1, 2]: season_factor = 1.25
        elif month in [7, 8]: season_factor = 0.8
        else: season_factor = 1.0
        
        # Volume quotidien (~773/jour en moyenne)
        daily_n = int(np.random.normal(773, 60) * season_factor)
        
        for _ in range(daily_n):
            pole_name = random.choices(poles_list, weights=poles_weights, k=1)[0]
            pole_conf = POLES_V4[pole_name]
            
            # Type & DMS
            is_ambu = random.random() < pole_conf["ratio_ambu"]
            if pole_name == "Urgences (Passage court)":
                adm_type, dms, mode_entree = "Passage Urgences", 0, "Urgences"
            else:
                adm_type = "Ambulatoire" if is_ambu else "Hospitalisation Complète"
                dms = 0 if is_ambu else max(1, int(np.random.gamma(2.0, pole_conf["dms"]/2.0)))
                mode_entree = random.choice(["Urgences", "Programmé", "Mutation"]) if pole_conf["mode_entree"] == "Mixte" else pole_conf["mode_entree"]

            # Motif & Greffes
            motif = random.choice(pole_conf["motifs"])
            is_transplant = False
            if pole_name == "Coeur-Metabolisme" and random.random() < 0.005:
                motif, is_transplant, dms = "Greffe Cardiaque", True, max(15, int(np.random.normal(30, 10)))
            if pole_name == "Chirurgie" and random.random() < 0.005:
                motif, is_transplant, dms = random.choice(["Greffe Hépatique", "Greffe Rénale"]), True, max(10, int(np.random.normal(20, 5)))

            # Démographie
            sex = "F" if random.random() < 0.49 else "M"
            if pole_name == "Gériatrie (Charles Foix)": age = int(np.random.normal(85, 7))
            elif motif == "Bronchiolite": age = random.randint(0, 2)
            else: age = int(np.random.normal(56 if sex=="M" else 55, 18))
            age = max(0, min(105, age))

            # Gravité & Origine
            gravite = random.randint(3, 5) if ("Réa" in pole_name or is_transplant) else random.randint(1, 4)
            origine = random.choices(origins_list, weights=origins_weights, k=1)[0]

            data.append({
                "id_admission": f"ADM-{202400000+global_id}",
                "date_entree": current_date.strftime("%Y-%m-%d"),
                "service": pole_name,
                "type_admission": adm_type,
                "mode_entree": mode_entree,
                "motif_principal": motif,
                "transplantation": is_transplant,
                "duree_sejour_jours": dms,
                "age_patient": age,
                "sexe_patient": sex,
                "departement_patient": origine,
                "gravite_score": gravite
            })
            global_id += 1

    return pd.DataFrame(data)

# Exécution
if __name__ == "__main__":
    from pathlib import Path
    
    # Créer le répertoire de sortie s'il n'existe pas
    output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "admissions_hopital_pitie_2024.csv"
    
    df = generate_complete_dataset_2024()
    df.to_csv(output_file, index=False)
    print(f"✅ Fichier généré : {output_file}")
    print(f"📊 Nombre de lignes : {len(df):,}")