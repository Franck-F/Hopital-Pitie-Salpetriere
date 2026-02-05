# Documentation Technique : Projet Pitié-Salpêtrière Vision 2024-2025

Cette documentation présente l'architecture globale, les flux de données et les composants techniques du système de pilotage hospitalier .

## 1. Vue d'Ensemble du Système

Le projet vise à fournir un outil d'aide à la décision pour la gestion des flux d'admissions et des ressources hospitalières (Lits, RH, Stocks). Il s'articule autour de trois piliers :

1. **Exploration de Données (Data Mining)** : Admissions, Logistique et Parcours Patient.
2. **Prédiction IA (LightGBM)** : Pour anticiper la charge avec une précision absolue.
3. **Simulation de Crise** : Pour tester la résilience des infrastructures.

## 2. Architecture Technique

### 2.1 Stack Technologique

- **Langage** : Python 3.13+
- **Frontend** : Streamlit (Interface réactive)
- **Data Processing** : Pandas, Numpy
- **Machine Learning** : LightGBM, Scikit-Learn (Métriques)
- **Visualisation** : Plotly Express & Graph Objects
- **Gestion de Dépendances** : `uv` (pyproject.toml)

### 2.2 Structure du Projet

- `app/` : Code source de l'application Dashboard.
  - `main.py` : Entrée principale, gestion de la navigation et des onglets.
  - `assets/` : Ressources graphiques (logos).
- `data/` : Entrepôts de données.
  - `raw/` : Données sources CSV (Admissions 2024-2025, Logistique, Patients).
- `models/` : Artefacts binaires.
  - `lightgbm.joblib` : Modèle "Champion" actuellement déployé.
- `notebooks/` : Environnements de recherche Jupyter.
  - `LigthGBM.ipynb` : Notebook de référence pour l'entraînement et l'analyse SHAP/Métriques.
- `scripts/` : Automatisation.
  - `train_champion_model.py` : Script de ré-entraînement et mise en prod du modèle.

## 3. Flux de Données (Data Pipelines)

### 3.1 Pipeline d'Admission (Time Series)

1. **Ingestion** : Lecture de `admissions_hopital_pitie_2024_2025.csv`.
2. **Transformation** : Conversion `date_entree` -> Datetime, agrégation quotidienne (`groupby size`).
3. **Feature Engineering (V6)** :
    - Lags : J-1 à J-7, J-14, J-21, J-28.
    - Rolling Stats : Moyennes glissantes 3J et 7J.
    - Encodage Temporel : Sinus/Cosinus (Jour/Année), Jours Fériés, Jour Semaine.

### 3.2 Pipeline Logistique & Patient

- **Logistique** : Jointures sur tables Lits/Perso/Equipements/Stocks via clés de date et service.
- **Parcours Patient** : Analyse croisée entre `sejours` et `diagnostics` pour lier pathologies et DMS.

## 4. Composant Machine Learning

- **Algorithme** : LightGBM Regressor.
- **Stratégie** : Entraînement complet pour capturer la totalité du signal sauf les 4 derniers mois (septembre-décembre 2025).
- **Hyperparamètres Clés** : 4000 arbres, learning_rate 0.01, profondeur illimitée(-1).
- **Métriques de Performance (Sept-Dec 2025)** :
  - MAE = 1.31

## 5. Guide de Maintenance

### 5.1 Mise à jour des Données

Pour intégrer de nouvelles admissions :

1. Remplacer le fichier dans `data/raw/admissions_hopital_pitie_2024_2025.csv`.
2. Exécuter `uv run python scripts/train_model.py`.
3. Relancer l'application streamlit.

### 5.2 Modification de l'Interface

Le fichier `app/main.py` est divisé en sections logiques indépendantes :

- `Landing Logic` : Page d'accueil.
- `Dashboard Logic` : Chargement des données.
- `Tabs` : Sous-sections (Accueil, Exploration, ML, Simulateur, Équipe).

## 6. Dépannage Courant

- **Erreur `ModuleNotFoundError`** : Vérifier l'installation via `uv sync`.
- **Graphiques manquants** : Vérifier la présence des fichiers CSV dans `data/raw`.
- **Modèle non chargé** : Exécuter le script d'entraînement pour régénérer le `.joblib`.
