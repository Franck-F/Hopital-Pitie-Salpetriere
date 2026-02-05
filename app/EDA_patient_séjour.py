import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CHARGEMENT DES DONNÉES RÉELLES 
patients = pd.read_csv("../data/raw/patients_pitie_2024.csv")
sejours = pd.read_csv(
    "../data/raw/sejours_pitie_2024.csv",
    parse_dates=["date_admission", "date_sortie"]
)
diagnostics = pd.read_csv("../data/raw/diagnostics_pitie_2024.csv")

# FONCTION DE STYLE "HÔPITAL"
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

# CRÉATION DU DASHBOARD 
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

# MISE EN PAGE FINALE 
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

# PRÉPARATION 
datasets = [
    (patients, "Patients"),
    (sejours, "Séjours"),
    (diagnostics, "Diagnostics")
]

# CRÉATION DU DASHBOARD
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[f"<b>{name}</b>" for _, name in datasets],
    horizontal_spacing=0.1,
    shared_yaxes=False
)

# GÉNÉRATION DES GRAPHIQUES 
for i, (df, name) in enumerate(datasets, 1):
    # Calcul du % de PRÉSENCE 
    # 100% = Tout est rempli
    completeness = (1 - df.isna().mean()) * 100
    
    # On trie pour l'esthétique
    completeness = completeness.sort_values(ascending=True)
    
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

# GRAPHIQUE 1 : Répartition du Sexe 

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

# BLOC DE GÉNÉRATION DE DONNÉES FICTIVES pour tester 

np.random.seed(42)
n_sejours = 10000
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

# Configuration Globale du Style 
# On définit une palette de couleurs moderne et un template propre
template_style = "plotly_white"
colors_hospit = px.colors.qualitative.Prism 


# DISTRIBUTION DE L'ÂGE (Histogramme avec Dégradé et Densité)

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


#  BOXPLOT ÂGE PAR TYPE 
fig2 = px.box(
    sejours,
    x="type_hospit",
    y="age",
    color="type_hospit", 
    color_discrete_sequence=colors_hospit,
    points="suspectedoutliers", 
    notched=True, 
    title="<b>Âge médian selon le type d'hospitalisation</b><br><span style='font-size:13px;color:grey'>Comparaison des distributions</span>",
    template=template_style
)

fig2.update_layout(
    xaxis_title=None, 
    yaxis_title="Âge",
    showlegend=False,
    title_x=0.5
)
fig2.show()

# SÉJOURS PAR PÔLE (Barres Horizontales, Triées et Dégradé)

# Préparation 
sej_pole = sejours["pole"].value_counts().reset_index()
sej_pole.columns = ["pole", "nb_sejours"]
sej_pole = sej_pole.sort_values(by="nb_sejours", ascending=True)

# Amélioration : Orientation horizontale
fig3 = px.bar(
    sej_pole,
    x="nb_sejours",
    y="pole",
    orientation='h', 
    color="nb_sejours",
    color_continuous_scale="Viridis", 
    text_auto='.2s', 
    title="<b>Classement des Pôles par volume de séjours</b>",
    template=template_style
)

fig3.update_layout(
    xaxis_title="Nombre de séjours",
    yaxis_title=None,
    coloraxis_showscale=False, 
    title_x=0.5,
    height=500 
)
# Augmenter la taille de la police des pôles sur l'axe Y
fig3.update_yaxes(tickfont=dict(size=12, weight='bold'))
fig3.show()


# TYPE D'HOSPIT PAR PÔLE 

sej_pole_type = (
    sejours.groupby(["pole", "type_hospit"])["id_sejour"]
    .count()
    .reset_index()
    .rename(columns={"id_sejour": "nb_sejours"})
)


fig4 = px.bar(
    sej_pole_type,
    x="pole",
    y="nb_sejours",
    color="type_hospit",
    barmode="group",
    color_discrete_sequence=colors_hospit, 
    title="<b>Détail des types d'hospitalisation par Pôle</b>",
    template=template_style
)

fig4.update_layout(
    xaxis_title="Pôle",
    yaxis_title="Nombre de séjours",
    title_x=0.5,
    legend_title_text=None, 
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis_tickangle=-45 
)
fig4.show()



import plotly.express as px

# DIAGNOSTICS PAR GROUPE DE PATHOLOGIE 

# Préparation : On compte et on trie
diag_patho = (
    diagnostics.groupby("pathologie_groupe")["id_sejour"]
    .count()
    .reset_index()
    .rename(columns={"id_sejour": "nb_diagnostics"})
    .sort_values("nb_diagnostics", ascending=True) 
)

fig1 = px.bar(
    diag_patho,
    x="nb_diagnostics",
    y="pathologie_groupe",
    orientation='h', 
    color="nb_diagnostics",
    color_continuous_scale="Tealgrn", 
    text_auto=True, 
    title="<b>Diagnostics par Groupe de Pathologie</b><br><span style='font-size:13px;color:grey'>Classement par volume (Pitié 2024)</span>",
    template="plotly_white"
)

fig1.update_layout(
    xaxis_title="Nombre de diagnostics",
    yaxis_title=None, 
    coloraxis_showscale=False,
    height=600, 
    title_x=0.5
)
fig1.show()

# PRINCIPAL VS SECONDAIRE
repartition = diagnostics["type_diagnostic"].value_counts().reset_index()
repartition.columns = ["type", "count"]

fig2 = px.pie(
    repartition,
    values="count",
    names="type",
    hole=0.5, 
    title="<b>Répartition des types de diagnostics</b>",
    color_discrete_sequence=['#2c3e50', '#e74c3c'], 
    template="plotly_white"
)

fig2.update_traces(
    textposition='inside', 
    textinfo='percent+label', 
    hoverinfo='label+value+percent',
    marker=dict(line=dict(color='#000000', width=1)) 
)

fig2.update_layout(
    showlegend=False, 
    title_x=0.5,
    annotations=[dict(text='Total', x=0.5, y=0.5, font_size=20, showarrow=False)] 
)
fig2.show()


# TOP 20 CIM-10 

top_cim = (
    diagnostics["cim10_code"]
    .value_counts()
    .head(20)
    .reset_index()
)
top_cim.columns = ["cim10_code", "nb"]
top_cim = top_cim.sort_values("nb", ascending=True)

fig3 = px.bar(
    top_cim,
    x="nb",
    y="cim10_code",
    orientation='h', 
    color="nb", 
    color_continuous_scale="Plasma",
    text_auto=True,
    title="<b>TOP 20 des codes CIM-10 les plus fréquents</b>",
    template="plotly_white"
)

fig3.update_layout(
    xaxis_title="Fréquence d'apparition",
    yaxis_title="Code CIM-10",
    coloraxis_showscale=False,
    height=700, 
    title_x=0.5,
    bargap=0.2
)

fig3.show()

import pandas as pd
import numpy as np
import plotly.express as px

np.random.seed(42)
dates_fictives = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h") 
sejours['date_admission'] = np.random.choice(dates_fictives, size=len(sejours))

print(" Colonne 'date_admission' simulée et ajoutée avec succès.")

# PRÉPARATION 
sejours_daily = (
    sejours.groupby(sejours["date_admission"].dt.date)["id_sejour"]
    .count()
    .reset_index()
    .rename(columns={"id_sejour": "nb_sejours", "date_admission": "date"})
)

# VISUALISATION INTERACTIVE
fig = px.line(
    sejours_daily,
    x="date",
    y="nb_sejours",
    title="<b>Évolution des admissions journalières</b><br><span style='font-size:13px;color:grey'>Suivi temporel Pitié-Salpêtrière 2024</span>",
    template="plotly_white",
    markers=False 
)

# LE STYLE "JOLI" 
fig.update_traces(
    line_color='#2980b9', 
    line_width=2,
    fill='tozeroy',
    fillcolor='rgba(41, 128, 185, 0.1)' 
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Nombre d'admissions",
    title_x=0.5,
    hovermode="x unified", 
    height=600
)

# Ajout du SÉLECTEUR DE PÉRIODE
fig.update_xaxes(
    rangeslider_visible=True, 
    rangeselector=dict(
        buttons=list([
            dict(count=7, label="7J", step="day", stepmode="backward"),
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(step="all", label="Tout")
        ]),
        bgcolor="#ecf0f1" 
    )
)

fig.show()

import pandas as pd
import numpy as np
import plotly.express as px

# RÉPARATION DES DONNÉES 
if 'date_admission' not in sejours.columns:
    print("Colonne 'date_admission' introuvable. Génération de dates fictives pour 2024...")
    # On génère des dates aléatoires sur l'année 2024
    dates_possibles = pd.date_range(start="2024-01-01", end="2024-12-31", freq="h")
    np.random.seed(42)
    sejours['date_admission'] = np.random.choice(dates_possibles, size=len(sejours))
    # On s'assure que c'est bien au format datetime
    sejours['date_admission'] = pd.to_datetime(sejours['date_admission'])

# ÉVOLUTION QUOTIDIENNE 

# Agrégation
sejours_daily = (
    sejours.groupby(sejours["date_admission"].dt.date)["id_sejour"]
    .count()
    .reset_index()
    .rename(columns={"id_sejour": "nb_sejours", "date_admission": "date"})
)

# Création du graphique 
fig_daily = px.area(
    sejours_daily,
    x="date",
    y="nb_sejours",
    title="<b>Flux quotidien des admissions</b><br><span style='font-size:13px;color:grey'>Vision détaillée jour par jour</span>",
    template="plotly_white",
    markers=False
)

# Style : Couleur et Slider
fig_daily.update_traces(line_color='#2980b9') 
fig_daily.update_layout(
    xaxis_title=None,
    yaxis_title="Nombre d'admissions",
    hovermode="x unified",
    title_x=0.5
)

# Ajout des boutons de contrôle temporel 
fig_daily.update_xaxes(
    rangeslider_visible=True, 
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


#  ÉVOLUTION MENSUELLE 

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
    text_auto=True,
    title="<b>Bilan Mensuel des Admissions</b><br><span style='font-size:13px;color:grey'>Volume global par mois</span>",
    template="plotly_white"
)

# Style
fig_monthly.update_traces(
    marker_color='#16a085', 
    textfont_size=12,
    textposition='outside'
)

fig_monthly.update_layout(
    xaxis_title=None,
    yaxis_title="Volume mensuel",
    title_x=0.5,
    bargap=0.2 
)

# Formatage de l'axe X pour afficher "Janvier", "Février"...
fig_monthly.update_xaxes(
    tickformat="%B",
    dtick="M1" 
)

fig_monthly.show()

# On prépare les données : Pôle -> Type Hospit
df_sun = sejours.groupby(['pole', 'type_hospit']).size().reset_index(name='count')

fig = px.sunburst(
    df_sun,
    path=['pole', 'type_hospit'], 
    values='count',
    color='pole', 
    color_discrete_sequence=px.colors.qualitative.Pastel, 
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

# On trie les jours pour l'ordre correct
order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
order_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Agrégation
heat_data = sejours.groupby(['mois', 'jour_semaine']).size().reset_index(name='nb_sejours')

fig = px.density_heatmap(
    heat_data,
    x="mois",
    y="jour_semaine",
    z="nb_sejours",
    color_continuous_scale="RdBu_r", 
    category_orders={"jour_semaine": order_days, "mois": order_months}, 
    title="<b>Heatmap de Tension : Jours vs Mois</b><br><span style='font-size:13px;color:grey'>Zones rouges = Forte affluence</span>",
    template="plotly_white"
)

fig.update_layout(
    xaxis_title=None,
    yaxis_title=None,
    height=500
)

fig.update_traces(xgap=2, ygap=2)

fig.show()

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
    size="age", 
    hover_data=['id_sejour', 'type_hospit'], 
    title="<b>Analyse : Âge vs Durée de Séjour</b><br><span style='font-size:13px;color:grey'>Chaque point est un patient (Échantillon de 500)</span>",
    template="plotly_white",
    opacity=0.7 
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

# Préparation des données agrégées par pôle
df_radar = sejours.groupby('pole').agg({
    'age': 'mean',
    'duree_jours': 'mean',
    'id_sejour': 'count'
}).reset_index()


for col in ['age', 'duree_jours', 'id_sejour']:
    df_radar[f'{col}_norm'] = df_radar[col] / df_radar[col].max()

# Création du Radar Chart
fig = go.Figure()

# On ajoute une trace (une ligne) pour chaque pôle
categories = ['Âge Moyen', 'Durée Moyenne (DMS)', 'Volume Activité']


for i, row in df_radar.head(3).iterrows():
    fig.add_trace(go.Scatterpolar(
        r=[row['age_norm'], row['duree_jours_norm'], row['id_sejour_norm']],
        theta=categories,
        fill='toself', 
        name=row['pole'],
        opacity=0.6
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1] 
        )),
    title="<b>Comparaison des Profils de Pôles</b><br><span style='font-size:13px;color:grey'>Données normalisées (le plus loin du centre = le plus élevé)</span>",
    template="plotly_white",
    height=500
)

fig.show()

import plotly.express as px


age_pole_sorted = age_pole.sort_values(by="nb_sejours", ascending=True)

fig = px.bar(
    age_pole_sorted,
    x="nb_sejours",
    y="pole",
    color="age_bin", 
    orientation='h', 
    title="<b>Répartition des Âges par Pôle</b><br><span style='font-size:13px;color:grey'>Qui fréquente quel service ?</span>",
    text_auto=True, 
    template="plotly_white",
    color_discrete_sequence=px.colors.sequential.RdBu_r 
)

fig.update_layout(
    xaxis_title="Nombre de séjours",
    yaxis_title=None,
    legend_title="Classe d'âge",
    height=600,
    title_x=0.5
)

fig.show()


import plotly.express as px

# Extraction des features temporelles
sejours['heure'] = sejours['date_admission'].dt.hour
sejours['jour'] = sejours['date_admission'].dt.day_name()

# Ordre correct des jours
jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Agrégation
tension = sejours.groupby(['jour', 'heure']).size().reset_index(name='nb_admissions')

# Heatmap
fig = px.density_heatmap(
    tension,
    x="heure",
    y="jour",
    z="nb_admissions",
    nbinsx=24, 
    category_orders={"jour": jours_ordre},
    color_continuous_scale="YlOrRd", 
    title="<b>Heatmap de Tension : Quand arrivent les patients ?</b><br><span style='font-size:13px;color:grey'>Zones rouges = Pic d'activité (Staff nécessaire)</span>",
    template="plotly_white"
)

fig.update_layout(
    xaxis_title="Heure d'admission",
    yaxis_title=None,
    height=500
)

fig.update_traces(xgap=2, ygap=2)
fig.show()