import numpy as np
import pandas as pd

# --------------------------------------------------------------------
# 0. Paramètres
# --------------------------------------------------------------------
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# --------------------------------------------------------------------
# 1. Chargement des données 2019–2025 (tous sites)
#    -> adapter les chemins si besoin
# --------------------------------------------------------------------
sejours = pd.read_csv(
    "sejours_2019_2025.csv",
    parse_dates=["date_admission", "date_sortie"]
)
diagnostics = pd.read_csv("diagnostics_2019_2025.csv")
patients = pd.read_csv("patients_2019_2025.csv")

# --------------------------------------------------------------------
# 2. Récupérer pathologie_groupe au niveau séjour
#    (on suppose qu'elle est stockée dans diagnostics)
# --------------------------------------------------------------------
patho_per_sejour = (
    diagnostics[["id_sejour", "pathologie_groupe"]]
    .drop_duplicates(subset=["id_sejour"])
)

sejours = sejours.merge(patho_per_sejour, on="id_sejour", how="left")

# --------------------------------------------------------------------
# 3. Filtrer Pitié uniquement
# --------------------------------------------------------------------
sejours_pitie = sejours[sejours["site"] == "Pitie"].copy()
diagnostics_pitie = diagnostics[
    diagnostics["id_sejour"].isin(sejours_pitie["id_sejour"])
].copy()
patients_pitie = patients[
    patients["id_patient"].isin(sejours_pitie["id_patient"])
].copy()

# --------------------------------------------------------------------
# 4. Filtrer sur l'année 2024 uniquement
# --------------------------------------------------------------------
sejours_pitie_2024 = sejours_pitie[sejours_pitie["annee"] == 2024].copy()
diagnostics_pitie_2024 = diagnostics_pitie[
    diagnostics_pitie["id_sejour"].isin(sejours_pitie_2024["id_sejour"])
].copy()
patients_pitie_2024 = patients_pitie[
    patients_pitie["id_patient"].isin(sejours_pitie_2024["id_patient"])
].copy()

# --------------------------------------------------------------------
# 5. Rééchantillonnage de l'âge par pathologie_groupe pour Pitié 2024
# --------------------------------------------------------------------
age_bins_patho = {
    "cancer":        [(0, 17, 0.02), (18, 44, 0.10), (45, 64, 0.40), (65, 79, 0.35), (80, 100, 0.13)],
    "neuro":         [(0, 17, 0.05), (18, 44, 0.15), (45, 64, 0.30), (65, 79, 0.30), (80, 100, 0.20)],
    "cardio":        [(0, 17, 0.01), (18, 44, 0.05), (45, 64, 0.30), (65, 79, 0.35), (80, 100, 0.29)],
    "endocrine_metabo": [(0, 17, 0.03), (18, 44, 0.20), (45, 64, 0.40), (65, 79, 0.25), (80, 100, 0.12)],
    "geno_urinaire": [(0, 17, 0.02), (18, 44, 0.10), (45, 64, 0.35), (65, 79, 0.30), (80, 100, 0.23)],
    "ortho_traumato":[(0, 17, 0.05), (18, 44, 0.20), (45, 64, 0.30), (65, 79, 0.25), (80, 100, 0.20)],
    "digestif":      [(0, 17, 0.03), (18, 44, 0.20), (45, 64, 0.35), (65, 79, 0.25), (80, 100, 0.17)],
    "obstetrique":   [(0, 17, 0.00), (18, 44, 0.95), (45, 64, 0.05), (65, 79, 0.00), (80, 100, 0.00)],
    "infectieux":    [(0, 17, 0.05), (18, 44, 0.20), (45, 64, 0.30), (65, 79, 0.25), (80, 100, 0.20)],
}

def sample_age_for_patho(patho, n):
    bins = age_bins_patho[patho]
    probs = np.array([w for (_, _, w) in bins])
    probs = probs / probs.sum()
    idx = rng.choice(len(bins), size=n, p=probs)
    ages = []
    for i in idx:
        low, high, _ = bins[i]
        ages.append(rng.integers(low, high + 1))
    return np.array(ages)

# Si tu as des pathologies non prévues, on les ignore ou tu peux leur attribuer un profil par défaut
ages = np.zeros(len(sejours_pitie_2024), dtype=int)
for patho, bins in age_bins_patho.items():
    idx = sejours_pitie_2024["pathologie_groupe"] == patho
    n = idx.sum()
    if n == 0:
        continue
    ages[idx] = sample_age_for_patho(patho, n)

sejours_pitie_2024["age"] = ages

# --------------------------------------------------------------------
# 6. Mettre à jour l'année de naissance approximative pour les patients 2024
# --------------------------------------------------------------------
# On calcule une année de naissance approx à partir du premier séjour 2024
tmp = sejours_pitie_2024[["id_patient", "annee", "age"]].copy()
tmp["annee_naissance_approx"] = tmp["annee"] - tmp["age"]
tmp = (
    tmp.sort_values(["id_patient", "annee"])
    .drop_duplicates(subset=["id_patient"], keep="first")[["id_patient", "annee_naissance_approx"]]
)

patients_pitie_2024 = (
    patients_pitie_2024.drop(columns=[c for c in patients_pitie_2024.columns if c.startswith("annee_naissance")], errors="ignore")
    .merge(tmp, on="id_patient", how="left")
)

# --------------------------------------------------------------------
# 7. Sauvegarde des 3 datasets finaux Pitié 2024
# --------------------------------------------------------------------
patients_pitie_2024.to_csv("patients_pitie_2024.csv", index=False)
sejours_pitie_2024.to_csv("sejours_pitie_2024.csv", index=False)
diagnostics_pitie_2024.to_csv("diagnostics_pitie_2024.csv", index=False)

print("Fichiers générés :")
print(" - patients_pitie_2024.csv")
print(" - sejours_pitie_2024.csv")
print(" - diagnostics_pitie_2024.csv")
