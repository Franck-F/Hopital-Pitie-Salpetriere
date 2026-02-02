import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

MEDICAMENTS = [
    # medicament, stock_initial, seuil_reappro (en % du stock initial), jours_livraison(0=lun..6=dim)
    {"name": "Paracetamol",      "init_stock": 12000, "threshold_pct": 0.22, "delivery_days": [1, 4]},  # mar/ven
    {"name": "Amoxicilline",     "init_stock":  4200, "threshold_pct": 0.20, "delivery_days": [2, 5]},  # mer/sam
    {"name": "Salbutamol",       "init_stock":  2600, "threshold_pct": 0.25, "delivery_days": [1, 4]},  # mar/ven
    {"name": "SerumPhysio",      "init_stock":  9000, "threshold_pct": 0.18, "delivery_days": [0, 3]},  # lun/jeu
    {"name": "Corticosteroides", "init_stock":  1800, "threshold_pct": 0.25, "delivery_days": [2, 5]},  # mer/sam
]

# Consommation : base + coeff * admissions + facteur épidémie (hiver)
# (valeurs fictives, mais cohérentes et stables)
CONSO_PARAMS = {
    "Paracetamol":      {"base": 90, "k_adm": 0.18, "k_winter": 0.35},
    "Amoxicilline":     {"base": 28, "k_adm": 0.06, "k_winter": 0.25},
    "Salbutamol":       {"base": 14, "k_adm": 0.03, "k_winter": 0.45},
    "SerumPhysio":      {"base": 70, "k_adm": 0.12, "k_winter": 0.15},
    "Corticosteroides": {"base":  8, "k_adm": 0.02, "k_winter": 0.20},
}

# Livraisons : livraison planifiée ~ pourcentage stock init, + urgence si stock < seuil
DELIVERY_PARAMS = {
    "planned_pct": 0.10,   # 10% du stock initial les jours planifiés
    "urgent_pct":  0.18,   # 18% du stock initial si urgence
}

# Pour reproductibilité
RNG_SEED = 2024


# ============================================================
# UTILITAIRES
# ============================================================

def season_factor(date: datetime) -> float:
    """
    Même logique que ton code admissions :
    - Hiver (déc/jan/fév) +25%
    - Été (juil/août) -20%
    - sinon 1.0
    """
    month = date.month
    if month in [12, 1, 2]:
        return 1.25
    elif month in [7, 8]:
        return 0.8
    return 1.0


def winter_epidemic_factor(date: datetime) -> float:
    """
    Facteur "épidémie" simple :
    - jan/fév : fort
    - nov/déc : moyen
    - sinon : faible
    """
    m = date.month
    if m in (1, 2):
        return 1.0
    if m in (11, 12):
        return 0.6
    return 0.2


def load_daily_admissions(admissions_csv_path: Path) -> pd.DataFrame:
    """
    Lit le CSV admissions déjà généré et calcule le volume par jour.
    On se base sur 'date_entree'.
    """
    df = pd.read_csv(admissions_csv_path)
    if "date_entree" not in df.columns:
        raise ValueError("Le fichier admissions doit contenir une colonne 'date_entree' (YYYY-MM-DD).")

    daily = (
        df.groupby("date_entree")
          .size()
          .reset_index(name="admissions_jour")
          .rename(columns={"date_entree": "date"})
    )

    # sécurise les dates manquantes (si jamais)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


# ============================================================
# GENERATION STOCKS
# ============================================================

def generate_stocks_2024(admissions_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Génère stocks_medicaments_2024.csv :
    date, medicament, stock, consommation, livraison
    """
    rng = np.random.default_rng(RNG_SEED)
    random.seed(RNG_SEED)

    # Crée l'index de dates complet 2024
    start = datetime(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(366)]  # 2024 est bissextile

    # Map admissions par date
    adm_map = {d.date(): int(v) for d, v in zip(admissions_daily["date"], admissions_daily["admissions_jour"])}

    # Stocks courants (par médicament)
    stock = {m["name"]: int(m["init_stock"]) for m in MEDICAMENTS}

    rows = []

    for d in dates:
        # admissions du jour (si absent -> 0)
        adm = adm_map.get(d.date(), 0)

        sf = season_factor(d)
        wf = winter_epidemic_factor(d)

        for med_conf in MEDICAMENTS:
            med = med_conf["name"]
            init_stock = med_conf["init_stock"]
            threshold = int(init_stock * med_conf["threshold_pct"])

            # ----------------------------
            # CONSOMMATION (Poisson)
            # ----------------------------
            p = CONSO_PARAMS[med]
            mean = (p["base"] + p["k_adm"] * adm) * sf * (1.0 + p["k_winter"] * wf)

            # sécurité : moyenne minimale
            mean = max(1.0, mean)

            consommation = int(rng.poisson(mean))

            # plafonne la consommation si stock insuffisant
            consommation = min(consommation, stock[med])

            # ----------------------------
            # LIVRAISON (planifiée + urgence)
            # ----------------------------
            livraison = 0
            dow = d.weekday()  # 0=lun ... 6=dim

            # livraison planifiée certains jours
            if dow in med_conf["delivery_days"]:
                planned_mean = init_stock * DELIVERY_PARAMS["planned_pct"]
                livraison += int(rng.poisson(planned_mean))

            # livraison d'urgence si stock sous seuil (avant conso du jour)
            # (tu peux aussi la déclencher après conso, mais ici c'est plus simple)
            if stock[med] < threshold:
                urgent_mean = init_stock * DELIVERY_PARAMS["urgent_pct"]
                livraison += int(rng.poisson(urgent_mean))

            # ----------------------------
            # MISE A JOUR STOCK
            # stock fin de journée
            # ----------------------------
            stock[med] = max(0, stock[med] - consommation + livraison)

            rows.append({
                "date": d.strftime("%Y-%m-%d"),
                "medicament": med,
                "stock": int(stock[med]),
                "consommation": int(consommation),
                "livraison": int(livraison)
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # chemins (comme ton script admissions)
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    admissions_file = raw_dir / "admissions_hopital_pitie_2024.csv"
    output_file = raw_dir / "stocks_medicaments_2024.csv"

    if not admissions_file.exists():
        raise FileNotFoundError(
            f"Fichier admissions introuvable : {admissions_file}\n"
            "Génère d'abord admissions_hopital_pitie_2024.csv."
        )

    admissions_daily = load_daily_admissions(admissions_file)
    df_stocks = generate_stocks_2024(admissions_daily)

    df_stocks.to_csv(output_file, index=False)
    print(f"Fichier généré : {output_file}")
    print(f"Nombre de lignes : {len(df_stocks):,}")
    print(df_stocks.head(10).to_string(index=False))

