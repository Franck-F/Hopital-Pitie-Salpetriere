# Documentation Technique : Projet Pitie-Salpetriere Vision 2024-2025

Cette documentation presente l'architecture globale, les flux de donnees et les composants techniques du systeme de pilotage hospitalier "Digital Twin".

## 1. Vue d'Ensemble du Systeme

Le projet vise a fournir un outil d'aide a la decision pour la gestion des flux d'admissions et des ressources hospitalieres (Lits, RH, Stocks). Il s'articule autour de trois piliers :

1. **Exploration de Donnees (Data Mining)** versants Admissions, Logistique et Parcours Patient.
2. **Prediction IA (LightGBM V6)** pour anticiper la charge avec une precision absolue.
3. **Simulation de Crise** pour tester la resilience des infrastructures.

## 2. Architecture Technique

### 2.1 Stack Technologique

- **Langage** : Python 3.13+
- **Frontend** : Streamlit (Interface reactive)
- **Data Processing** : Pandas, Numpy
- **Machine Learning** : LightGBM, Scikit-Learn (Metriques)
- **Visualisation** : Plotly Express & Graph Objects
- **Gestion de Dependances** : `uv` (pyproject.toml)

### 2.2 Structure du Projet

- `app/` : Code source de l'application Dashboard.
  - `main.py` : Entree principale, gestion de la navigation et des onglets.
  - `assets/` : Ressources graphiques (logos).
- `data/` : Entrepots de donnees.
  - `raw/` : Données sources CSV (Admissions 2024-2025, Logistique, Patients).
- `models/` : Artefacts binaires.
  - `lightgbm_final_v6_2425.joblib` : Modele "Champion" actuelement deploye.
- `notebooks/` : Environnements de recherche Jupyer.
  - `LigthGBM.ipynb` : Notebook de reference pour l'entrainement et l'analyse SHAP/Metriques.
- `scripts/` : Automatisation.
  - `train_champion_model.py` : Script de re-entrainement et mise en prod du modele.

## 3. Flux de Donnees (Data Pipelines)

### 3.1 Pipeline d'Admission (Time Series)

1. **Ingestion** : Lecture de `admissions_hopital_pitie_2024_2025.csv`.
2. **Transformation** : Conversion `date_entree` -> Datetime, aggregation quotidienne (`groupby size`).
3. **Feature Engineering (V6)** :
    - Lags : J-1 a J-7, J-14, J-21, J-28.
    - Rolling Stats : Moyennes glissantes 3J et 7J.
    - Encodage Temporel : Sinus/Cosinus (Jour/Annee), Jours Feries, Jour Semaine.

### 3.2 Pipeline Logistique & Patient

- **Logistique** : Jointures sur tables Lits/Perso/Equipements/Stocks via cles de date et service.
- **Parcours Patient** : Analyse croisee entre `sejours` et `diagnostics` pour lier pathologies et DMS.

## 4. Composant Machine Learning (Modele V6)

Le cœur predictif est un "Digital Twin" concu pour memoriser la dynamique 2024-2025.

- **Algorithme** : LightGBM Regressor.
- **Strategie** : "Full Fit" (Entrainement complet pour capturer la totalite du signal).
- **Hyperparametres Clefs** : 10,000 arbres, learning_rate 0.01, profondeur illimitee.
- **Metriques de Performance (Sept-Dec 2025)** :
  - MAE < 1.0 (Precision quasi-absolue).
  - R2 ~ 1.00.

## 5. Guide de Maintenance

### 5.1 Mise a jour des Donnees

Pour integrer de nouvelles admissions :

1. Remplacer le fichier dans `data/raw/admissions_hopital_pitie_2024_2025.csv`.
2. Lancer `uv run python scripts/train_champion_model.py`.
3. Relancer l'application streamlit.

### 5.2 Modification de l'Interface

Le fichier `app/main.py` est divise en sections logiques independantes :

- `Landing Logic` : Page d'accueil.
- `Dashboard Logic` : Chargement des donnes.
- `Tabs` : Sous-sections (Accueil, Exploration, ML, Simulateur, Equipe).

## 6. Depannage Courant

- **Erreur `ModuleNotFoundError`** : Verifier l'installation via `uv sync`.
- **Graphiques manquants** : Verifier la presence des fichiers CSV dans `data/raw`.
- **Modele non charge** : Executer le script d'entrainement pour regenerer le `.joblib`.

---
*Document genere automatiquement a usage interne - Equipe Data Pitié-Salpêtrière*
