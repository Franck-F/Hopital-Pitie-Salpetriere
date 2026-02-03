# Documentation Technique : Pitié-Salpêtrière Vision 2026

Cette documentation detaille l'implementation logicielle, la configuration du modele de machine learning et les procedures de maintenance.

## 1. Structure du Code Source

Le projet suit une organisation modulaire pour faciliter l'evolutivite :

- `app/main.py` : Point d'entree principal du dashboard Streamlit. Contient la logique d'interface (UI) et le routage.
- `data/raw/` : Stockage des jeux de donnees source au format CSV (Admissions, Logistique, Sejours).
- `models/` : Contient les modeles serialises (Format `.joblib`).
- `scripts/` : Utilitaires d'entrainement, d'optimisation et de verification de precision.
- `reports/` : Livrables de conception et documentation.

## 2. Pipeline de Prevision (Machine Learning)

### Modele Champion

- **Type** : LightGBM Regressor.
- **Parametres Clefs** : `num_leaves=31`, `learning_rate=0.05`, `n_estimators=1000`.
- **Pre-traitement** : Generation recursive de previsions pour un horizon J+14.

### Logique des Variables (Features)

Le modele s'appuie sur l'auto-regression :

- `lag1` : Volume d'admissions du jour precedent.
- `lag2` : Volume d'admissions d'il y a 48 heures.
- `lag7` : Volume d'admissions d'il y a une semaine (capture la saisonnalite hebdomadaire).

## 3. Interface Utilisateur (UI/UX)

La charte graphique est injectee via `st.markdown` avec du CSS personnalise :

- **Typographie** : 'Outfit' (Google Fonts) pour un aspect institutionnel et moderne.
- **Couleurs Alpha** : Utilisation intensive de fonds semi-transparents (`rgba`) pour l'effet Glassmorphism.
- **Composants Plotly** : Les graphiques sont configures avec `template="plotly_dark"` et des fonds transparents pour s'integrer fluidement au dashboard.

## 4. Maintenance Technique

### Regles de Codage

- **Formatage** : Strict respect de l'absence d'emojis dans le code source (`utf-8` sans symboles speciaux).
- **Cachage** : Utilisation de `@st.cache_data` pour les donnees volumineuses et `@st.cache_resource` pour les modeles ML afin de garantir la fluidite de l'interface.

### Deploiement Local

Le projet utilise `uv` pour gerer l'environnement :

1. Installer les dependances : `uv sync`
2. Lancer le dashboard : `uv run streamlit run app/main.py`

## 5. Gestion des Erreurs

- Les graphiques Plotly mixtes (ex: Pie dans Barres) necessitent de definir explicitement les `specs` dans `make_subplots`.
- Les previsions sont garanties positives via la fonction `max(0, pred)` dans la boucle recursive pour eviter des anomalies physiques.
