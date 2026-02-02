# Hôpital Pitié-Salpêtrière : Système de Prévision et de Gestion des Ressources

![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)
![Status](https://img.shields.io/badge/status-in--development-yellow.svg)

## Overview

Ce projet vise à développer un système de prévision et de gestion des ressources pour l'Hôpital Pitié-Salpêtrière. Il permet d'anticiper les pics d'activité, d'optimiser l'allocation des lits et du personnel, et de fournir un tableau de bord interactif pour les décideurs hospitaliers.

## Objectifs

- Générer un jeu de données réaliste basé sur l'activité hospitalière.
- Analyser les tendances d'admissions et identifier les périodes critiques.
- Développer un modèle prédictif (Machine Learning) pour anticiper les flux.
- Créer un prototype MVP avec un tableau de bord interactif (Streamlit/Plotly).

## Stack Technique

- **Gestion des dépendances** : `uv`
- **Langage** : Python 3.13+
- **Analyse de données** : `pandas`, `numpy`, `scipy`
- **Visualisation** : `plotly`, `matplotlib`, `seaborn`
- **Machine Learning** : `scikit-learn`, `xgboost`, `prophet`, `statsmodels`
- **Application Web** : `streamlit`
- **Tests** : `pytest`

## Structure du Projet

- `data/` : Jeux de données bruts et transformés.
- `notebooks/` : Analyses exploratoires et développement des modèles.
- `src/` : Code source pour la génération de données et le processing.
- `app/` : Application Streamlit (MVP).
- `reports/` : Rapports de conception, d'analyse et stratégiques.
- `tests/` : Tests unitaires.

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
