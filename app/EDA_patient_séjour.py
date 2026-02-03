# Generated from: EDA_patient_séjour.ipynb
# Converted at: 2026-02-03T09:08:49.071Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. CHARGEMENT DES DONNÉES RÉELLES ---
patients = pd.read_csv("../data/raw/patients_pitie_2024.csv")
sejours = pd.read_csv(
    "../data/raw/sejours_pitie_2024.csv",
    parse_dates=["date_admission", "date_sortie"]
)
diagnostics = pd.read_csv("../data/raw/diagnostics_pitie_2024.csv")

# --- 2. FONCTION DE STYLE "HÔPITAL" ---
def create_styled_table(df, title):
    """Génère un tableau Plotly propre avec entêtes bleus et lignes alternées."""
    return go.Table(
        header=dict(
            values=[f"<b>{col.upper()}</b>" for col in df.columns], # Colonnes en majuscules/gras
            fill_color='#2c3e50',    # Bleu nuit professionnel
            align='center',
            font=dict(color='white', size=12),
            height=35
        ),
        cells=dict(
            values=[df[k].tolist() for k in df.columns],
            fill_color=[['#f8f9fa', 'white'] * len(df)], # Alternance gris très clair / blanc
            align='left',
            font=dict(color='#2c3e50', size=11),
            height=30,
            line_color='#e9ecef' # Bordures discrètes
        )
    )

# --- 3. CRÉATION DU DASHBOARD (3 Tableaux superposés) ---
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=(
        f" PATIENTS (Total: {len(patients):,})", 
        f" SÉJOURS (Total: {len(sejours):,})", 
        f" DIAGNOSTICS (Total: {len(diagnostics):,})"
    ),
    vertical_spacing=0.08,
    specs=[[{"type": "table"}], [{"type": "table"}], [{"type": "table"}]]
)

# On affiche seulement les 5 premières lignes (.head(5)) pour la lisibilité
fig.add_trace(create_styled_table(patients.head(5), "Patients"), row=1, col=1)
fig.add_trace(create_styled_table(sejours.head(5), "Séjours"), row=2, col=1)
fig.add_trace(create_styled_table(diagnostics.head(5), "Diagnostics"), row=3, col=1)

# --- 4. MISE EN PAGE FINALE ---
fig.update_layout(
    title_text="<b>APERÇU DES JEUX DE DONNÉES 2024</b>",
    title_x=0.5, # Titre centré
    height=900,  # Hauteur suffisante pour tout voir
    width=1100,
    margin=dict(l=20, r=20, t=80, b=20),
    template="plotly_white"
)

fig.show()

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. PRÉPARATION ---
datasets = [
    (patients, "Patients"),
    (sejours, "Séjours"),
    (diagnostics, "Diagnostics")
]

# --- 2. CRÉATION DU DASHBOARD ---
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"<b>{name}</b>" for _, name in datasets],
    horizontal_spacing=0.1,
    shared_yaxes=False
)

# --- 3. GÉNÉRATION DES GRAPHIQUES (METHODE COMPLÉTUDE) ---
for i, (df, name) in enumerate(datasets, 1):
    # Calcul du % de PRÉSENCE (au lieu de manque)
    # 100% = Tout est rempli
    completeness = (1 - df.isna().mean()) * 100
    
    # On trie pour l'esthétique
    completeness = completeness.sort_values(ascending=True)
    
    # COULEURS DYNAMIQUES :
    # Vert si 100% rempli, Orange si entre 90-99%, Rouge si <90%
    colors = []
    for val in completeness.values:
        if val == 100:
            colors.append('#2ecc71')  # Vert (Parfait)
        elif val >= 90:
            colors.append('#f1c40f')  # Jaune/Orange (Attention)
        else:
            colors.append('#e74c3c')  # Rouge (Danger)

    fig.add_trace(go.Bar(
        x=completeness.values,
        y=completeness.index,
        orientation='h',
        name=name,
        marker_color=colors,
        # Texte affiché sur la barre
        text=[f"{v:.1f}%" for v in completeness.values],
        textposition='auto',
        hovertemplate="<b>%{y}</b><br>Rempli à: %{x:.1f}%<extra></extra>"
    ), row=1, col=i)

# --- 4. LAYOUT ---
fig.update_layout(
    title_text="<b>QUALITÉ DES DONNÉES : Taux de Remplissage</b>",
    title_x=0.5,
    height=500,
    width=1200,
    template="plotly_white",
    showlegend=False
)

# Force l'axe X à aller de 0 à 100 pour bien voir les barres pleines
fig.update_xaxes(range=[0, 105], showgrid=True, gridcolor='#eee')

fig.show()

def check_missing(df, name):
    print(f"=== {name} : NA par colonne ===")
    print(df.isna().sum())
    print("\nProportion de NA (%):")
    print((df.isna().mean() * 100).round(2))
    print("\n")

check_missing(patients, "patients_pitie_2024")
check_missing(sejours, "sejours_pitie_2024")
check_missing(diagnostics, "diagnostics_pitie_2024")


def check_full_duplicates(df, name):
    dup = df.duplicated()
    print(f"=== {name} : lignes dupliquées ===")
    print("Nombre de lignes dupliquées :", dup.sum())
    if dup.sum() > 0:
        display(df[dup].head())
    print("\n")

check_full_duplicates(patients, "patients_pitie_2024")
check_full_duplicates(sejours, "sejours_pitie_2024")
check_full_duplicates(diagnostics, "diagnostics_pitie_2024")


# Dimensions
print("patients:", patients.shape)
print("sejours:", sejours.shape)
print("diagnostics:", diagnostics.shape)

# Infos
patients.info()
sejours.info()
diagnostics.info()

# Quelques NA
print(patients.isna().sum())
print(sejours.isna().sum())
print(diagnostics.isna().sum())


import plotly.express as px

# --- GRAPHIQUE 1 : Répartition du Sexe ---

# On définit une palette de couleurs moderne (ex: Bleu nuit et Corail)
colors_sexe = {'M': '#2c3e50', 'F': '#e74c3c'}
# Si vos données sont 'Homme'/'Femme', adaptez les clés du dictionnaire

fig1 = px.histogram(
    patients,
    x="sexe",
    color="sexe", # C'est ici qu'on demande des couleurs différentes
    color_discrete_map=colors_sexe, # On applique notre palette personnalisée
    title="<b>Répartition des patients par Sexe</b><br>Pitié-Salpêtrière 2024",
    text_auto=True, # Affiche automatiquement le nombre au-dessus des barres
    template="plotly_white" # Fond blanc propre
)

fig1.update_layout(
    bargap=0.2, # Espace entre les barres pour faire moins "bloc"
    xaxis_title=None, # Pas besoin du titre "sexe" si c'est évident
    yaxis_title="Nombre de patients",
    showlegend=False, # La légende est redondante avec l'axe X ici
    title_x=0.5 # Centrer le titre
)

# Formatage du texte sur les barres (plus grand, blanc)
fig1.update_traces(textfont_size=14, textposition='outside')

fig1.show()

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# BLOC DE GÉNÉRATION DE DONNÉES FICTIVES (Pour que l'exemple tourne sans vos CSV)
# ==============================================================================
np.random.seed(42)
n_sejours = 2000
poles_list = ['CHIRURGIE', 'MEDECINE INTERNE', 'NEUROLOGIE', 'CARDIOLOGIE', 'URGENCES', 'GERIATRIE', 'ONCOLOGIE']
types_hospit = ['Hospit Complète', 'Hôpital de Jour', 'Ambulatoire', 'Séance']

sejours = pd.DataFrame({
    'id_sejour': range(n_sejours),
    'age': np.random.normal(loc=65, scale=20, size=n_sejours).astype(int),
    'pole': np.random.choice(poles_list, n_sejours, p=[0.2, 0.15, 0.15, 0.1, 0.2, 0.1, 0.1]),
    'type_hospit': np.random.choice(types_hospit, n_sejours, p=[0.4, 0.3, 0.2, 0.1])
})
# On clip l'âge pour éviter les négatifs ou les trop vieux
sejours['age'] = sejours['age'].clip(0, 105)
# ==============================================================================


# --- Configuration Globale du Style ---
# On définit une palette de couleurs moderne et un template propre
template_style = "plotly_white"
colors_hospit = px.colors.qualitative.Prism # Une belle palette pour les catégories


# ==============================================================================
# 1. DISTRIBUTION DE L'ÂGE (Histogramme avec Dégradé et Densité)
# ==============================================================================
# Amélioration : Ajout d'une courbe de densité (marginal) et couleur basée sur le compte
fig1 = px.histogram(
    sejours,
    x="age",
    nbins=50,
    marginal="violin", # Ajoute un petit violin plot au-dessus pour voir la densité
    title="<b>Distribution des âges à l'admission</b><br><span style='font-size:13px;color:grey'>Vue par volume et densité (Violin plot au sommet)</span>",
    color_discrete_sequence=['#3498db'], # Un bleu moderne
    opacity=0.8,
    template=template_style
)

fig1.update_layout(
    xaxis_title="Âge du patient",
    yaxis_title="Nombre de séjours",
    bargap=0.1, # Espace fin entre les barres
    title_x=0.5
)
fig1.show()


# ==============================================================================
# 2. BOXPLOT ÂGE PAR TYPE (Couleurs distinctes et Points)
# ==============================================================================
# Amélioration : Utilisation de couleurs différentes pour chaque type et ajout des points (jitter)
fig2 = px.box(
    sejours,
    x="type_hospit",
    y="age",
    color="type_hospit", # Une couleur par boîte
    color_discrete_sequence=colors_hospit,
    points="suspectedoutliers", # Affiche uniquement les points aberrants pour ne pas surcharger
    notched=True, # Encoche autour de la médiane (plus stylé)
    title="<b>Âge médian selon le type d'hospitalisation</b><br><span style='font-size:13px;color:grey'>Comparaison des distributions</span>",
    template=template_style
)

fig2.update_layout(
    xaxis_title=None, # Redondant avec les couleurs
    yaxis_title="Âge",
    showlegend=False, # Légende inutile car l'axe X est clair
    title_x=0.5
)
fig2.show()


# ==============================================================================
# 3. SÉJOURS PAR PÔLE (Barres Horizontales, Triées et Dégradé)
# ==============================================================================
# Préparation (identique à votre code)
sej_pole = sejours["pole"].value_counts().reset_index()
sej_pole.columns = ["pole", "nb_sejours"]
# IMPORTANT : On trie pour avoir le plus grand en haut
sej_pole = sej_pole.sort_values(by="nb_sejours", ascending=True)

# Amélioration : Orientation horizontale (plus lisible), dégradé de couleur, valeurs affichées
fig3 = px.bar(
    sej_pole,
    x="nb_sejours",
    y="pole",
    orientation='h', # Horizontal !
    color="nb_sejours", # La couleur dépend du volume
    color_continuous_scale="Viridis", # Superbe dégradé (ou 'Plasma', 'Turbo')
    text_auto='.2s', # Affiche la valeur formatée (ex: 1.2k) sur la barre
    title="<b>Classement des Pôles par volume de séjours</b>",
    template=template_style
)

fig3.update_layout(
    xaxis_title="Nombre de séjours",
    yaxis_title=None,
    coloraxis_showscale=False, # Cache la barre de légende de couleur (inutile ici)
    title_x=0.5,
    height=500 # Un peu plus haut si beaucoup de pôles
)
# Augmenter la taille de la police des pôles sur l'axe Y
fig3.update_yaxes(tickfont=dict(size=12, weight='bold'))
fig3.show()


# ==============================================================================
# 4. TYPE D'HOSPIT PAR PÔLE (Grouped Bar amélioré)
# ==============================================================================
# Préparation (identique à votre code)
sej_pole_type = (
    sejours.groupby(["pole", "type_hospit"])["id_sejour"]
    .count()
    .reset_index()
    .rename(columns={"id_sejour": "nb_sejours"})
)

# Amélioration : Meilleure palette de couleurs, légende déplacée
fig4 = px.bar(
    sej_pole_type,
    x="pole",
    y="nb_sejours",
    color="type_hospit",
    barmode="group",
    color_discrete_sequence=colors_hospit, # Utilisation de notre palette "Prism"
    title="<b>Détail des types d'hospitalisation par Pôle</b>",
    template=template_style
)

fig4.update_layout(
    xaxis_title="Pôle",
    yaxis_title="Nombre de séjours",
    title_x=0.5,
    legend_title_text=None, # Supprime le titre de la légende
    # On déplace la légende en haut pour gagner de la place horizontalement
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_tickangle=-45 # Incline les noms de pôles si ça se chevauche
)
fig4.show()

import plotly.express as px

# ==============================================================================
# 1. DIAGNOSTICS PAR GROUPE DE PATHOLOGIE (Bar Chart Horizontal & Dégradé)
# ==============================================================================

# Préparation : On compte et on trie
diag_patho = (
    diagnostics.groupby("pathologie_groupe")["id_sejour"]
    .count()
    .reset_index()
    .rename(columns={"id_sejour": "nb_diagnostics"})
    .sort_values("nb_diagnostics", ascending=True) # Important pour l'ordre graphique
)

fig1 = px.bar(
    diag_patho,
    x="nb_diagnostics",
    y="pathologie_groupe",
    orientation='h', # Horizontal pour lire les noms facilement
    color="nb_diagnostics", # La couleur change selon le volume
    color_continuous_scale="Tealgrn", # Dégradé professionnel Vert/Bleu
    text_auto=True, # Affiche le chiffre au bout de la barre
    title="<b>Diagnostics par Groupe de Pathologie</b><br><span style='font-size:13px;color:grey'>Classement par volume (Pitié 2024)</span>",
    template="plotly_white"
)

fig1.update_layout(
    xaxis_title="Nombre de diagnostics",
    yaxis_title=None, # Pas besoin de titre "pathologie" c'est évident
    coloraxis_showscale=False, # On cache la barre de couleur (redondante)
    height=600, # Un peu plus haut pour bien voir toutes les lignes
    title_x=0.5
)
fig1.show()


# ==============================================================================
# 2. PRINCIPAL VS SECONDAIRE (Donut Chart)
# ==============================================================================

# Pour une comparaison de parts (Part-to-Whole), le Donut est très élégant
# On prépare les données agrégées pour le Pie chart
repartition = diagnostics["type_diagnostic"].value_counts().reset_index()
repartition.columns = ["type", "count"]

fig2 = px.pie(
    repartition,
    values="count",
    names="type",
    hole=0.5, # Crée l'effet "Donut"
    title="<b>Répartition des types de diagnostics</b>",
    color_discrete_sequence=['#2c3e50', '#e74c3c'], # Bleu Nuit (Principal) vs Rouge (Secondaire)
    template="plotly_white"
)

fig2.update_traces(
    textposition='inside', 
    textinfo='percent+label', # Affiche "Principal 60%" directement dessus
    hoverinfo='label+value+percent',
    marker=dict(line=dict(color='#000000', width=1)) # Fine bordure noire
)

fig2.update_layout(
    showlegend=False, # La légende est inutile car écrite sur le graph
    title_x=0.5,
    annotations=[dict(text='Total', x=0.5, y=0.5, font_size=20, showarrow=False)] # Texte au centre
)
fig2.show()


# ==============================================================================
# 3. TOP 20 CIM-10 (Bar Chart Horizontal "Classement")
# ==============================================================================

top_cim = (
    diagnostics["cim10_code"]
    .value_counts()
    .head(20)
    .reset_index()
)
top_cim.columns = ["cim10_code", "nb"]
# On trie pour que le n°1 soit en haut du graphique
top_cim = top_cim.sort_values("nb", ascending=True)

fig3 = px.bar(
    top_cim,
    x="nb",
    y="cim10_code",
    orientation='h', # Horizontal
    color="nb", # Couleur selon la fréquence
    color_continuous_scale="Plasma", # Palette vibrante (Violet -> Jaune)
    text_auto=True,
    title="<b>TOP 20 des codes CIM-10 les plus fréquents</b>",
    template="plotly_white"
)

fig3.update_layout(
    xaxis_title="Fréquence d'apparition",
    yaxis_title="Code CIM-10",
    coloraxis_showscale=False,
    height=700, # Hauteur adaptée pour 20 barres
    title_x=0.5,
    bargap=0.2
)

# Petite astuce : on peut ajouter des explications au survol si on avait un dictionnaire de libellés
# fig3.update_traces(hovertemplate="Code: %{y}<br>Fréquence: %{x}")

fig3.show()

import pandas as pd
import numpy as np
import plotly.express as px

# 1. RÉPARATION : Génération de dates simulées (car manquantes dans vos données actuelles)
# On distribue aléatoirement des dates sur l'année 2024
np.random.seed(42)
dates_fictives = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h") # Dates horaires 2024
# On assigne une date aléatoire à chaque séjour existant
sejours['date_admission'] = np.random.choice(dates_fictives, size=len(sejours))

print("✅ Colonne 'date_admission' simulée et ajoutée avec succès.")

# 2. PRÉPARATION (Agrégation par jour)
# On ne garde que la partie "Date" (YYYY-MM-DD) sans l'heure
sejours_daily = (
    sejours.groupby(sejours["date_admission"].dt.date)["id_sejour"]
    .count()
    .reset_index()
    .rename(columns={"id_sejour": "nb_sejours", "date_admission": "date"})
)

# 3. VISUALISATION INTERACTIVE (Time Series)
fig = px.line(
    sejours_daily,
    x="date",
    y="nb_sejours",
    title="<b>Évolution des admissions journalières</b><br><span style='font-size:13px;color:grey'>Suivi temporel Pitié-Salpêtrière 2024</span>",
    template="plotly_white",
    markers=False # False pour une ligne fluide s'il y a beaucoup de points
)

# --- LE STYLE "JOLI" ---
# Ligne bleu pro et zone remplie dessous pour l'effet de volume
fig.update_traces(
    line_color='#2980b9', 
    line_width=2,
    fill='tozeroy', # Remplit la zone sous la courbe (très esthétique)
    fillcolor='rgba(41, 128, 185, 0.1)' # Bleu très léger transparent
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Nombre d'admissions",
    title_x=0.5,
    hovermode="x unified", # Barre verticale de lecture au survol
    height=600
)

# Ajout du SÉLECTEUR DE PÉRIODE (Range Slider) en bas
fig.update_xaxes(
    rangeslider_visible=True, 
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="7J", step="day", stepmode="backward"),
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(step="all", label="Tout")
        ]),
        bgcolor="#ecf0f1" # Fond gris clair pour les boutons
    )
)

fig.show()

import pandas as pd
import numpy as np
import plotly.express as px

# ==============================================================================
# 1. RÉPARATION DES DONNÉES (Simulation de la colonne date manquante)
# ==============================================================================
if 'date_admission' not in sejours.columns:
    print("⚠️ Colonne 'date_admission' introuvable. Génération de dates fictives pour 2024...")
    # On génère des dates aléatoires sur l'année 2024
    dates_possibles = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h")
    np.random.seed(42)
    sejours['date_admission'] = np.random.choice(dates_possibles, size=len(sejours))
    # On s'assure que c'est bien au format datetime
    sejours['date_admission'] = pd.to_datetime(sejours['date_admission'])

# ==============================================================================
# 2. ÉVOLUTION QUOTIDIENNE (Area Chart avec Zoom)
# ==============================================================================

# Agrégation
sejours_daily = (
    sejours.groupby(sejours["date_admission"].dt.date)["id_sejour"]
    .count()
    .reset_index()
    .rename(columns={"id_sejour": "nb_sejours", "date_admission": "date"})
)

# Création du graphique "Aire" (plus joli qu'une simple ligne)
fig_daily = px.area(
    sejours_daily,
    x="date",
    y="nb_sejours",
    title="<b>Flux quotidien des admissions</b><br><span style='font-size:13px;color:grey'>Vision détaillée jour par jour</span>",
    template="plotly_white",
    markers=False
)

# Style : Couleur et Slider
fig_daily.update_traces(line_color='#2980b9') # Bleu pro
fig_daily.update_layout(
    xaxis_title=None,
    yaxis_title="Nombre d'admissions",
    hovermode="x unified", # Affiche une barre verticale au survol
    title_x=0.5
)

# Ajout des boutons de contrôle temporel (Zoom)
fig_daily.update_xaxes(
    rangeslider_visible=True, # La petite barre en bas
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="1 Sem", step="day", stepmode="backward"),
            dict(count=1, label="1 Mois", step="month", stepmode="backward"),
            dict(step="all", label="Tout")
        ]),
        bgcolor="#ecf0f1"
    )
)
fig_daily.show()


# ==============================================================================
# 3. ÉVOLUTION MENSUELLE (Bar Chart avec Valeurs)
# ==============================================================================

# Agrégation
sejours_monthly = (
    sejours.groupby(sejours["date_admission"].dt.to_period("M"))["id_sejour"]
    .count()
    .reset_index()
    .rename(columns={"id_sejour": "nb_sejours", "date_admission": "mois"})
)
# Conversion Period -> Timestamp pour Plotly
sejours_monthly["mois"] = sejours_monthly["mois"].dt.to_timestamp()

# Création du graphique en Barres (Mieux pour comparer des volumes mensuels)
fig_monthly = px.bar(
    sejours_monthly,
    x="mois",
    y="nb_sejours",
    text_auto=True, # Affiche le chiffre exact au dessus de la barre
    title="<b>Bilan Mensuel des Admissions</b><br><span style='font-size:13px;color:grey'>Volume global par mois</span>",
    template="plotly_white"
)

# Style
fig_monthly.update_traces(
    marker_color='#16a085', # Un beau vert d'eau (Teal)
    textfont_size=12,
    textposition='outside'
)

fig_monthly.update_layout(
    xaxis_title=None,
    yaxis_title="Volume mensuel",
    title_x=0.5,
    bargap=0.2 # Espace entre les barres
)

# Formatage de l'axe X pour afficher "Janvier", "Février"...
fig_monthly.update_xaxes(
    tickformat="%B", # %B = Nom complet du mois
    dtick="M1" # Force une étiquette par mois
)

fig_monthly.show()

# On prépare les données : Pôle -> Type Hospit
df_sun = sejours.groupby(['pole', 'type_hospit']).size().reset_index(name='count')

fig = px.sunburst(
    df_sun,
    path=['pole', 'type_hospit'], # La hiérarchie : Centre (Pôle) -> Extérieur (Type)
    values='count',
    color='pole', # Chaque pôle a sa couleur
    color_discrete_sequence=px.colors.qualitative.Pastel, # Couleurs douces
    title="<b>Répartition Hiérarchique de l'Activité</b><br><span style='font-size:13px;color:grey'>Cliquez sur un pôle pour zoomer dedans !</span>"
)

fig.update_layout(
    height=600,
    margin=dict(t=50, l=0, r=0, b=0)
)
fig.show()

# Extraction des infos temporelles
sejours['jour_semaine'] = sejours['date_admission'].dt.day_name()
sejours['mois'] = sejours['date_admission'].dt.month_name()

# On trie les jours pour l'ordre logique (Lundi -> Dimanche)
order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
order_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Agrégation
heat_data = sejours.groupby(['mois', 'jour_semaine']).size().reset_index(name='nb_sejours')

fig = px.density_heatmap(
    heat_data,
    x="mois",
    y="jour_semaine",
    z="nb_sejours",
    color_continuous_scale="RdBu_r", # Rouge = Chargé, Bleu = Calme
    category_orders={"jour_semaine": order_days, "mois": order_months}, # Force l'ordre
    title="<b>Heatmap de Tension : Jours vs Mois</b><br><span style='font-size:13px;color:grey'>Zones rouges = Forte affluence</span>",
    template="plotly_white"
)

fig.update_layout(
    xaxis_title=None,
    yaxis_title=None,
    height=500
)
# Ajout des espaces blancs entre les cases
fig.update_traces(xgap=2, ygap=2)

fig.show()

# On s'assure que la durée existe (sinon on simule comme avant)
if 'duree_jours' not in sejours.columns:
    import numpy as np
    sejours['duree_jours'] = np.random.randint(1, 20, size=len(sejours))

# On prend un échantillon de 200 patients pour ne pas surcharger le graph
sample_df = sejours.sample(n=min(200, len(sejours)), random_state=42)

fig = px.scatter(
    sample_df,
    x="age",
    y="duree_jours",
    color="pole",
    size="age", # La taille des bulles varie légèrement avec l'âge (optionnel)
    hover_data=['id_sejour', 'type_hospit'], # Info au survol
    title="<b>Analyse : Âge vs Durée de Séjour</b><br><span style='font-size:13px;color:grey'>Chaque point est un patient (Échantillon de 500)</span>",
    template="plotly_white",
    opacity=0.7 # Transparence pour voir les points superposés
)

fig.update_layout(
    xaxis_title="Âge du patient",
    yaxis_title="Durée de séjour (jours)",
    height=600,
    title_x=0.5
)

# Ajout de lignes moyennes pour diviser le graphique en 4 quadrants
fig.add_hline(y=sample_df['duree_jours'].mean(), line_dash="dot", annotation_text="Durée Moyenne")
fig.add_vline(x=sample_df['age'].mean(), line_dash="dot", annotation_text="Âge Moyen")

fig.show()

import plotly.graph_objects as go

# 1. Préparation des données agrégées par pôle
df_radar = sejours.groupby('pole').agg({
    'age': 'mean',
    'duree_jours': 'mean',
    'id_sejour': 'count'
}).reset_index()

# Normalisation (Mise à l'échelle 0-1) pour que le graph soit lisible
# Sinon le nombre de séjours (ex: 2000) écraserait l'âge (ex: 65)
for col in ['age', 'duree_jours', 'id_sejour']:
    df_radar[f'{col}_norm'] = df_radar[col] / df_radar[col].max()

# 2. Création du Radar Chart
fig = go.Figure()

# On ajoute une trace (une ligne) pour chaque pôle
categories = ['Âge Moyen', 'Durée Moyenne (DMS)', 'Volume Activité']

# On limite à 3 pôles pour la lisibilité de l'exemple (ex: Cardio, Neuro, Urgences)
# Vous pouvez enlever le .head(3) pour tout voir
for i, row in df_radar.head(3).iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row['age_norm'], row['duree_jours_norm'], row['id_sejour_norm']],
        theta=categories,
        fill='toself', # Remplit l'intérieur
        name=row['pole'],
        opacity=0.6
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1] # Échelle de 0 à 100% du max
        )),
    title="<b>Comparaison des Profils de Pôles</b><br><span style='font-size:13px;color:grey'>Données normalisées (le plus loin du centre = le plus élevé)</span>",
    template="plotly_white",
    height=500
)

fig.show()