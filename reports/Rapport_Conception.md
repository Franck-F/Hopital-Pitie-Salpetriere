# Rapport de Conception : Pitié-Salpêtrière Vision 2026

Ce document décrit l'architecture technique, les choix méthodologiques et l'implémentation du dashboard décisionnel pour l'Hôpital Pitié-Salpêtrière.

## 1. Vision du Projet

L'objectif est de fournir une interface unique permettant de piloter les flux de patients, d'optimiser les ressources logistiques et d'anticiper les prises en charge via l'intelligence artificielle.

## 2. Architecture Technique

Le projet repose sur une pile technologique moderne et performante :

- **Interface** : Streamlit avec une charte graphique premium (Typographie Outfit, Glassmorphisme).
- **Visualisation** : Plotly pour des graphiques interactifs et dynamiques (Sunburst, Heatmaps, Boxplots).
- **Gestion des Données** : Pandas et NumPy pour le traitement des flux admission/logistique/séjour.
- **Moteur Prédictif** : LightGBM pour la prévision de charge hospitalière.

## 3. Stratégie de Modélisation (ML)

Le modèle retenu est un **LightGBM Regressor**.

- **Performance** : MAE < 5.0 (Erreur Absolue Moyenne).
- **Variables Clés (Features)** :
  - Lag 1 : Admissions de la veille.
  - Lag 2 : Tendance à 48h.
  - Lag 7 : Récurrence hebdomadaire (Pattern lundi-dimanche).
- **Robustesse** : Le modèle a été calibré pour résister au bruit statistique des périodes de transition (ex: mois de décembre, janvier et février).

## 4. Modules Exploration (EDA)

Le dashboard intègre trois axes d'analyse exhaustive :

- **Admissions** : Analyse des motifs, des modes d'entrée et des pics temporels.
- **Logistique** : Suivi des effectifs (ratio Infirmier/Lit), des stocks de sécurité et de l'occupation des zones critiques (Urgences/Réanimation).
- **Patients** : Profilage démographique et analyse des parcours via la nomenclature CIM-10.

## 5. Simulateur de Tension

Un moteur de stress-test permet de simuler des scénarios de crise (Épidémies, Plans Blancs). Il utilise les prévisions du modèle LightGBM comme baseline et applique un surcroît de charge paramétrable pour évaluer la résilience des infrastructures actuelles.

## 6. Maintenance et Évolutivité

- **Emoji-Free** : Le code source et la documentation respectent une stricte absence d'emojis pour garantir une compatibilité maximale avec les environnements de production.
- **Gestion UV** : Utilisation d'Astral UV pour une gestion reproductible des dépendances et de l'environnement virtuel.
