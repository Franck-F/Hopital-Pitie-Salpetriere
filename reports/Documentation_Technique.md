# Documentation Technique : Pitie-Salpetriere Vision 2024-2025 (V6)

Cette documentation detaille l'implementation de la version V6 "Digital Twin" du systeme de prevision.

## 1. Architecture du Modele Champion (V6)

Le cœur du systeme est un modele de regression **LightGBM** hautement calibre pour reproduire fidelement la dynamique historique.

### Configuration "Digital Twin"

- **Objectif** : Precision absolue (MAE < 1.0) sur la periode 2024-2025.
- **Strategie d'Entrainement** : "Full Fit" (Entrainement sur l'integralite du dataset).
- **Hyperparametres** :
  - `n_estimators` : 10000 (Tres haute capacite).
  - `learning_rate` : 0.01 (Convergence fine).
  - `num_leaves` : 128 ( Profondeur accrue pour capturer les micro-patterns).
  - `max_depth` : Illimite.

### Feature Engineering Avance

Le modele exploite une densite temporelle elevee :

1. **Lags Profonds** : 1, 2, 3, 4, 5, 6, 7, 14, 21, 28 jours.
2. **Statistiques Glissantes** : Moyennes sur 3 et 7 jours.
3. **Saisonnalite Cyclique** : Sinus/Cosinus (Annuel et Mensuel).
4. **Calendrier** : Jours feries et position dans la semaine/mois.

## 2. Structure Simplifiee du Projet

Le projet a ete nettoye pour ne conserver que les composants essentiels a la production :

- `data/` : Contient le dataset unifie `admissions_hopital_pitie_2024_2025.csv`.
- `models/` : Heberge uniquement le champion `lightgbm_final_v6_2425.joblib`.
- `notebooks/` : Le notebook `LigthGBM.ipynb` sert de reference pour la validation.
- `scripts/` : Le script `train_champion_model.py` permet de regenerer le modele V6.
- `app/` : Application Streamlit de pilotage.

## 3. Maintenance et Deploiement

### Standards

- **Zero Emoji** : Le code et la documentation technique sont strictement depourvus d'emojis pour garantir la compatibilite.
- **Reproductibilite** : Le script `train_champion_model.py` fixe les `random_state` pour garantir des resultats identiques.

### Procedure de Mise a Jour

1. Placer les nouvelles donnees dans `data/raw/`.
2. Executer `uv run python scripts/train_champion_model.py`.
3. Le dashboard `app/main.py` chargera automatiquement le nouveau modele.
