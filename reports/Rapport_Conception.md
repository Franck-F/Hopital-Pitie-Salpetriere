# Rapport de Conception : Pitié-Salpêtrière Vision 2026

Ce document decrit l'architecture technique, les choix methodologiques et l'implementation du dashboard decisionnel pour l'Hôpital Pitié-Salpêtrière.

## 1. Vision du Projet

L'objectif est de fournir une interface unique permettant de piloter les flux de patients, d'optimiser les ressources logistiques et d'anticiper les prises en charge via l'intelligence artificielle.

## 2. Architecture Technique

Le projet repose sur une pile technologique moderne et performante :

- **Interface** : Streamlit avec une charte graphique premium (Outfit Typography, Glassmorphism).
- **Visualisation** : Plotly pour des graphiques interactifs et dynamiques (Sunburst, Heatmaps, Boxplots).
- **Gestion des Donnees** : Pandas et NumPy pour le traitement des flux admission/logistique/sejour.
- **Moteur Predictif** : LightGBM  pour la prevision de charge hospitaliere.

## 3. Strategie de Modelisation (ML)

Le modele retenu est un **LightGBM Regressor** optimise.

- **Performance** : MAE de 67.92 (Erreur Absolue Moyenne).
- **Variables Clefs (Features)** :
  - Lag 1 : Admissions de la veille.
  - Lag 2 : Tendance à 48h.
  - Lag 7 : Recurrence hebdomadaire (Pattern lundi-dimanche).
- **Robustesse** : Le modele a ete calibre pour resister au bruit statistique des periodes de transition (ex: mois de decembre).

## 4. Modules Exploration (EDA)

Le dashboard integre trois axes d'analyse exhaustive :

- **Admissions** : Analyse des motifs, des modes d'entree et des pics temporels.
- **Logistique** : Suivi des effectifs (ratio Infirmier/Lit), des stocks de securite et de l'occupation des zones critiques (Urgences/Reanimation).
- **Patients** : Profiling demographique et analyse des parcours via la nomenclature CIM-10.

## 5. Simulateur de Tension

Un moteur de stress-test permet de simuler des scenarios de crise (Epidemies, Plans Blancs). Il utilise les previsions du modele LightGBM comme baseline et applique un surcroit de charge parametrable pour evaluer la resilience des infrastructures actuelles.

## 6. Maintenance et Evolutivite

- **Emoji-Free** : Le code source et la documentation respectent une stricte absence d'emojis pour garantir une compatibilite maximale avec les environnements de production.
- **Gestion UV** : Utilisation d'Astral UV pour une gestion reproductible des dependances et de l'environnement virtuel.
