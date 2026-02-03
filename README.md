# Hopital Pitie-Salpetriere : Systeme de Prevision et de Gestion des Ressources

## Overview

Ce projet deploie un systeme de prevision avance pour l'Hopital Pitie-Salpetriere (Vision 2024-2025). Il est centre sur un modele de "Jumeau Numerique" (Digital Twin) capable de reproduire la dynamique des admissions avec une precision quasi-absolue.

## Objectifs Atteints

- **Vision 2024-2025** : Couverture complete de la periode historique recente.
- **Digital Twin V6** : Modele LightGBM "Full Fit" atteignant une MAE < 1.0.
- **Robustesse** : Suppression des artefacts non pertinents et optimisation pour la prise de decision reelle.
- **Dashboard Operationnel** : Interface Streamlit epuree pour le pilotage strategique.

## Stack Technique

- **Gestion** : `uv`
- **Langage** : Python 3.13+
- **Core** : `pandas`, `numpy`, `joblib`
- **Modelisation** : `lightgbm` (Champion V6), `scikit-learn`
- **Visualisation** : `plotly`
- **Application** : `streamlit`

## Structure du Projet

- `data/` : Dataset unifie `admissions_hopital_pitie_2024_2025.csv`.
- `notebooks/` : Notebook unique de reference `LigthGBM.ipynb` (V6).
- `models/` : Modele Champion `lightgbm_final_v6_2425.joblib` uniquement.
- `scripts/` : Script de re-entrainement `train_champion_model.py`.
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
