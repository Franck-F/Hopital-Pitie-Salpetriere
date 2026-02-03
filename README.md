# Hôpital Pitié-Salpêtrière : Système de Prévision et de Gestion des Ressources

## Overview

Ce projet déploie un système de prévision avancé pour l'Hôpital Pitié-Salpêtrière (Vision 2024-2025). Il est centré sur un modèle de "Jumeau Numérique" (Digital Twin) capable de reproduire la dynamique des admissions avec une précision quasi-absolue.

## Objectifs Atteints

- **Vision 2024-2025** : Couverture complète de la période historique récente.
- **Digital Twin V6** : Modèle LightGBM "Full Fit" atteignant une MAE < 1.0.
- **Robustesse** : Suppression des artefacts non pertinents et optimisation pour la prise de décision réelle.
- **Dashboard Opérationnel** : Interface Streamlit épurée pour le pilotage stratégique.

## Stack Technique

- **Gestion** : `uv`
- **Langage** : Python 3.13+
- **Core** : `pandas`, `numpy`, `joblib`
- **Modélisation** : `lightgbm` (Champion V6), `scikit-learn`
- **Visualisation** : `plotly`
- **Application** : `streamlit`

## Structure du Projet

- `data/` : Dataset unifié `admissions_hopital_pitie_2024_2025.csv`.
- `notebooks/` : Notebook unique de référence `LigthGBM.ipynb` (V6).
- `models/` : Modèle Champion `lightgbm_final_v6_2425.joblib` uniquement.
- `scripts/` : Script de ré-entraînement `train_champion_model.py`.
- `app/` : Application de pilotage.
- `reports/` : Documentation technique et conception.

## Installation

Assurez-vous d'avoir `uv` installé.

```bash
uv sync
```

## Utilisation

Pour lancer l'application Streamlit :

```bash
uv run streamlit run app/main.py
```

## Auteurs

- [@FranckF](https://github.com/FranckF)
- [@koffigaetan-adj](https://github.com/koffigaetan-adj)
- [@Djouhratabet](https://github.com/Djouhratabet)
- [@cmartineau15](https://github.com/cmartineau15)
