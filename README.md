# Hôpital Pitié-Salpêtrière : Système de Prévision et de Gestion des Ressources

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53.1-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LightGBM](https://img.shields.io/badge/lightgbm-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://lightgbm.com.)
---

[![Aperçu Application](app/assets/demo_screenshot.png)](https://hopital-pitie-salpetrieregit-jsfpemvrjtde9tma3f7yq6.streamlit.app/)

## Overview

Ce projet vise à développer un système de prévision et de gestion des ressources pour l'Hôpital Pitié-Salpêtrière. Il permet d'anticiper les photos d'activité, d'optimiser l'allocation des lits et du personnel, et de fournir un tableau de bord interactif pour les décideurs hospitaliers.

## Stack Technique

- **Gestion** : `uv`
- **Langage** : Python 3.13+
- **Core** : `pandas`, `numpy`, `joblib`
- **Modélisation** : `lightgbm` (Champion V6), `scikit-learn`
- **Visualisation** : `plotly`
- **Application** : `streamlit`

## Structure du Projet

- `data/` : Dataset unifié `admissions_hopital_pitie_2024_2025.csv`.
- `notebooks/` : `LigthGBM.ipynb` `EDA_Admission.ipynb` `EDA_logistique.ipynb` `EDA_patient_séjour.ipynb`.
- `models/` : Modèle  `lightgbm_final.joblib`.
- `scripts/` : Script de ré-entraînement `train_model.py`.
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
- [@farah2791](https://github.com/farah2791)
