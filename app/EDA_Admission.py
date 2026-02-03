# Generated from: EDA_Admission.ipynb
# Converted at: 2026-02-02T16:09:29.482Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # ANALYSE EXPLORATOIRE - ADMISSIONS HOSPITALIÈRES


# ### Bibliothèques 


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Configuration pour Plotly dans Jupyter
import plotly.io as pio
pio.renderers.default = "notebook"
print("Bibliothèques importées avec succès")

# * **Aperçu**


df_admission = pd.read_csv('../data/raw/admissions_hopital_pitie_2024.csv')
df_admission.head()

df_admission.info()

# 
# ## ANALYSE DESCRIPTIVE COMPLÈTE
# 


# ### Vue d'ensemble des données


# Statistiques descriptives complètes
print("="*80)
print("STATISTIQUES GÉNÉRALES DES ADMISSIONS 2024".center(80))
print("="*80)
print(f"\n Période d'analyse : {df_admission['date_entree'].min()} → {df_admission['date_entree'].max()}")
print(f" Nombre total d'admissions/passages : {len(df_admission):,}")
print(f" Nombre de pôles/services : {df_admission['service'].nunique()}")
print(f"Départements d'origine : {df_admission['departement_patient'].nunique()}")
print(f"Modes d'entrée : {df_admission['mode_entree'].nunique()}")

# Répartition par catégorie - TABLEAU
print("\n" + "="*80)
print("RÉPARTITION PAR TYPE D'ADMISSION")
print("="*80 + "\n")

type_counts = df_admission['service'].value_counts().reset_index()
type_counts.columns = ['Service', 'Nombre d\'admissions']
type_counts['Pourcentage (%)'] = (type_counts['Nombre d\'admissions'] / len(df_admission) * 100).round(2)
type_counts['Pourcentage'] = type_counts['Pourcentage (%)'].apply(lambda x: f"{x:.2f}%")

# Afficher le tableau
display(type_counts[['Service', 'Nombre d\'admissions', 'Pourcentage']].style
        .format({'Nombre d\'admissions': '{:,}'})
        .background_gradient(subset=['Nombre d\'admissions'], cmap='Blues')
        .set_properties(**{'text-align': 'left'})
        .set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), 
                                          ('font-weight', 'bold'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('border', '1px solid #ddd'), ('padding', '8px')]}
        ])
)


# ### Distribution des variables catégorielles


# Graphiques de distribution pour toutes les variables catégorielles
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Distribution par Pôle/Service', 'Distribution par Mode d\'Entrée', 
                    'Distribution Géographique', 'Top 20 Motifs d\'Admission'),
    specs=[[{'type': 'bar'}, {'type': 'pie'}],
           [{'type': 'bar'}, {'type': 'bar'}]]
)

# Distribution par pôle (barres horizontales)
pole_counts = df_admission['service'].value_counts().head(10)
fig.add_trace(
    go.Bar(y=pole_counts.index, x=pole_counts.values, orientation='h',
           marker_color='lightblue', name='Pôles'),
    row=1, col=1
)

# Mode d'entrée (pie chart)
mode_counts = df_admission['mode_entree'].value_counts()
fig.add_trace(
    go.Pie(labels=mode_counts.index, values=mode_counts.values, name='Mode'),
    row=1, col=2
)

# Origine géographique
geo_counts = df_admission['departement_patient'].value_counts().head(10)
fig.add_trace(
    go.Bar(x=geo_counts.index, y=geo_counts.values,
           marker_color='coral', name='Origine'),
    row=2, col=1
)

# 4. Top motifs d'admission
motif_counts = df_admission['motif_principal'].value_counts().head(20)
fig.add_trace(
    go.Bar(x=motif_counts.index, y=motif_counts.values,
           marker_color='lightgreen', name='Motifs'),
    row=2, col=2
)

fig.update_layout(height=900, showlegend=False, title_text="<b>Distributions des Variables Catégorielles</b>")
fig.update_xaxes(tickangle=45, row=2, col=1)
fig.update_xaxes(tickangle=45, row=2, col=2)
fig.show()

# ## ANALYSE TEMPORELLE APPROFONDIE
# 
# ### Tendances et Saisonnalité


# Préparation des données temporelles
df_admission['date_entree'] = pd.to_datetime(df_admission['date_entree'])
df_admission['annee'] = df_admission['date_entree'].dt.year
df_admission['mois'] = df_admission['date_entree'].dt.month
df_admission['mois_nom'] = df_admission['date_entree'].dt.month_name()
df_admission['jour_semaine'] = df_admission['date_entree'].dt.dayofweek
df_admission['jour_semaine_nom'] = df_admission['date_entree'].dt.day_name()
df_admission['semaine'] = df_admission['date_entree'].dt.isocalendar().week
df_admission['jour_annee'] = df_admission['date_entree'].dt.dayofyear
df_admission['trimestre'] = df_admission['date_entree'].dt.quarter
df_admission['est_weekend'] = df_admission['jour_semaine'].isin([5, 6])
df_admission['heure'] = pd.to_datetime(df_admission['date_entree']).dt.hour if 'heure' not in df_admission.columns else df_admission['heure']

# Agrégations temporelles
daily_admissions = df_admission.groupby('date_entree').size()
weekly_admissions = df_admission.groupby('semaine').size()
monthly_admissions = df_admission.groupby('mois').size()

print("Variables temporelles créées")
print(f"   • Période : {df_admission['date_entree'].min()} → {df_admission['date_entree'].max()}")
print(f"   • Moyenne quotidienne : {daily_admissions.mean():.1f} admissions")
print(f"   • Écart-type quotidien : {daily_admissions.std():.1f}")
print(f"   • Jour max : {daily_admissions.max()} admissions")

# Décomposition de la série temporelle (Tendance + Saisonnalité + Résidus)
daily_series = daily_admissions.asfreq('D', fill_value=0)

# Décomposition additive
decomposition = seasonal_decompose(daily_series, model='additive', period=7)

# Visualisation interactive
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=('Série Temporelle Originale', 'Tendance (Trend)', 
                    'Composante Saisonnière', 'Résidus'),
    vertical_spacing=0.08,
    row_heights=[0.25, 0.25, 0.25, 0.25]
)

# Série originale
fig.add_trace(
    go.Scatter(
        x=daily_series.index, 
        y=daily_series.values,
        mode='lines',
        name='Série Originale',
        line=dict(color='blue', width=1.5),
        hovertemplate='Date: %{x}<br>Admissions: %{y}<extra></extra>'
    ),
    row=1, col=1
)

# Tendance
fig.add_trace(
    go.Scatter(
        x=decomposition.trend.index, 
        y=decomposition.trend.values,
        mode='lines',
        name='Tendance',
        line=dict(color='red', width=2),
        hovertemplate='Date: %{x}<br>Tendance: %{y:.2f}<extra></extra>'
    ),
    row=2, col=1
)

# Saisonnalité
fig.add_trace(
    go.Scatter(
        x=decomposition.seasonal.index, 
        y=decomposition.seasonal.values,
        mode='lines',
        name='Saisonnalité',
        line=dict(color='green', width=1.5),
        hovertemplate='Date: %{x}<br>Saisonnalité: %{y:.2f}<extra></extra>'
    ),
    row=3, col=1
)

# Résidus
fig.add_trace(
    go.Scatter(
        x=decomposition.resid.index, 
        y=decomposition.resid.values,
        mode='lines',
        name='Résidus',
        line=dict(color='orange', width=1),
        hovertemplate='Date: %{x}<br>Résidus: %{y:.2f}<extra></extra>'
    ),
    row=4, col=1
)

# Ajouter une ligne horizontale à zéro pour les résidus
fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=4, col=1)

# Mise à jour du layout
fig.update_xaxes(title_text="Date", row=4, col=1)
fig.update_yaxes(title_text="Admissions", row=1, col=1)
fig.update_yaxes(title_text="Tendance", row=2, col=1)
fig.update_yaxes(title_text="Saisonnalité", row=3, col=1)
fig.update_yaxes(title_text="Résidus", row=4, col=1)

fig.update_layout(
    height=1200,
    showlegend=False,
    title_text="<b>Décomposition de la Série Temporelle (Additive - Période: 7 jours)</b>",
    template='plotly_white',
    hovermode='x unified'
)

fig.show()

# ### Patterns Hebdomadaires et Mensuels


# Analyse par jour de la semaine et par mois
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Admissions par Jour de la Semaine', 'Admissions par Mois',
                    'Boxplot par Jour de Semaine', 'Heatmap Semaine x Mois'),
    specs=[[{'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'box'}, {'type': 'heatmap'}]]
)

# Moyenne par jour de la semaine
jour_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
jour_stats = df_admission.groupby('jour_semaine_nom').size().reindex(jour_ordre)

fig.add_trace(
    go.Bar(x=jour_stats.index, y=jour_stats.values, marker_color='lightblue', name='Jours'),
    row=1, col=1
)

# Par mois
mois_ordre = ['January', 'February', 'March', 'April', 'May', 'June', 
              'July', 'August', 'September', 'October', 'November', 'December']
mois_stats = df_admission.groupby('mois_nom').size().reindex(mois_ordre)

fig.add_trace(
    go.Bar(x=mois_stats.index, y=mois_stats.values, marker_color='coral', name='Mois'),
    row=1, col=2
)

# Boxplot par jour (variabilité)
for jour in jour_ordre:
    data_jour = df_admission[df_admission['jour_semaine_nom'] == jour].groupby('date_entree').size()
    fig.add_trace(
        go.Box(y=data_jour.values, name=jour[:3], showlegend=False),
        row=2, col=1
    )

#Heatmap Jour x Mois
pivot_data = df_admission.groupby(['jour_semaine', 'mois']).size().unstack(fill_value=0)
fig.add_trace(
    go.Heatmap(z=pivot_data.values, x=pivot_data.columns, y=jour_ordre,
               colorscale='YlOrRd', showscale=True),
    row=2, col=2
)

fig.update_layout(height=1000, showlegend=False, title_text="<b>Patterns Temporels des Admissions</b>")
fig.update_xaxes(tickangle=45, row=1, col=2)
fig.show()


# Statistiques
print("\n STATISTIQUES PAR JOUR DE LA SEMAINE:")
print("="*60)
for jour in jour_ordre:
    count = jour_stats[jour]
    pct = (count / len(df_admission)) * 100
    print(f"  {jour:<12} : {count:>7,} ({pct:>5.2f}%)")

print("\nSTATISTIQUES PAR MOIS:")
print("="*60)
for mois in mois_ordre:
    count = mois_stats[mois]
    pct = (count / len(df_admission)) * 100
    print(f"  {mois:<12} : {count:>7,} ({pct:>5.2f}%)")

# ## ANALYSE PAR PÔLES/SERVICES
# 
# ### Comparaison des Pôles


# Analyse détaillée par pôle
poles_analysis = df_admission.groupby('service').agg({
    'date_entree': 'count',
    'departement_patient': lambda x: x.mode()[0] if len(x) > 0 else 'N/A',
    'mode_entree': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
}).rename(columns={'date_entree': 'total_admissions'})

poles_analysis['pct_total'] = (poles_analysis['total_admissions'] / len(df_admission)) * 100
poles_analysis = poles_analysis.sort_values('total_admissions', ascending=False)

print("="*100)
print("ANALYSE DÉTAILLÉE PAR PÔLE".center(100))
print("="*100)
print(f"\n{'Pôle':<45} {'Admissions':<15} {'% Total':<12} {'Origine Dom.':<20} {'Mode':<15}")
print("-"*100)

for pole, row in poles_analysis.iterrows():
    print(f"{pole:<45} {row['total_admissions']:>12,}   {row['pct_total']:>7.2f}%   {row['departement_patient']:<20} {row['mode_entree']:<15}")

# Évolution temporelle par pôle (top 5)
top_poles = poles_analysis.head(5).index

fig = go.Figure()
for pole in top_poles:
    pole_daily = df_admission[df_admission['service'] == pole].groupby('date_entree').size()
    pole_daily_ma = pole_daily.rolling(window=7).mean()
    
    fig.add_trace(go.Scatter(
        x=pole_daily_ma.index,
        y=pole_daily_ma.values,
        mode='lines',
        name=pole,
        line=dict(width=2)
    ))

fig.update_layout(
    title='<b>Évolution des Top 5 Pôles (Moyenne Mobile 7 jours)</b>',
    xaxis_title='Date',
    yaxis_title='Admissions',
    template='plotly_white',
    hovermode='x unified',
    height=600
)
fig.show()

# ## ANALYSE GÉOGRAPHIQUE
# 
# ### Distribution géographique des admissions


# Analyse géographique détaillée
geo_stats = df_admission.groupby('departement_patient').agg({
    'date_entree': 'count',
    'service': lambda x: x.mode()[0] if len(x) > 0 else 'N/A',
    'mode_entree': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
}).rename(columns={'date_entree': 'total'})

geo_stats['pct'] = (geo_stats['total'] / len(df_admission)) * 100
geo_stats = geo_stats.sort_values('total', ascending=False)

print("="*100)
print("DISTRIBUTION GÉOGRAPHIQUE DES ADMISSIONS".center(100))
print("="*100)
print(f"\n{'Origine':<35} {'Admissions':<15} {'% Total':<12} {'Pôle Principal':<30} {'Mode Principal':<15}")
print("-"*100)

for origine, row in geo_stats.iterrows():
    print(f"{origine:<35} {row['total']:>12,}   {row['pct']:>7.2f}%   {row['service']:<30} {row['mode_entree']:<15}")
# Graphiques
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Répartition Géographique', 'Top 10 Origines'),
    specs=[[{'type': 'pie'}, {'type': 'bar'}]]
)

# Pie chart
fig.add_trace(
    go.Pie(labels=geo_stats.index, values=geo_stats['total'], 
           textposition='inside', textinfo='percent+label'),
    row=1, col=1
)

# Bar chart top 10
top_geo = geo_stats.head(10)
fig.add_trace(
    go.Bar(x=top_geo.index, y=top_geo['total'], marker_color='teal'),
    row=1, col=2
)

fig.update_layout(height=500, showlegend=False, title_text="<b>Analyse Géographique</b>")
fig.update_xaxes(tickangle=45, row=1, col=2)
fig.show()



# Croisement Géographie x Pôle
print("\n" + "="*100)
print("CROISEMENT : ORIGINE GÉOGRAPHIQUE × PÔLE (Top 5 combinaisons)".center(100))
print("="*100)
geo_pole_cross = df_admission.groupby(['departement_patient', 'service']).size().sort_values(ascending=False).head(10)
for (geo, service), count in geo_pole_cross.items():
    pct = (count / len(df_admission)) * 100
    print(f"  {geo:<25} × {service:<40} : {count:>6,} ({pct:>5.2f}%)")

# ## DÉTECTION D'ANOMALIES ET VALEURS EXTRÊMES
# 
# ### Identification des pics et creux anormaux


# Détection d'outliers (méthode IQR et Z-score)
from scipy import stats as scipy_stats

# Statistiques sur les admissions quotidiennes
Q1 = daily_admissions.quantile(0.25)
Q3 = daily_admissions.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Z-score
z_scores = np.abs(scipy_stats.zscore(daily_admissions))
outliers_zscore = daily_admissions[z_scores > 3]

# IQR outliers
outliers_iqr_high = daily_admissions[daily_admissions > upper_bound]
outliers_iqr_low = daily_admissions[daily_admissions < lower_bound]

print("="*80)
print("DÉTECTION D'ANOMALIES - ADMISSIONS QUOTIDIENNES".center(80))
print("="*80)
print(f"\nStatistiques:")
print(f"   • Moyenne : {daily_admissions.mean():.1f}")
print(f"   • Médiane : {daily_admissions.median():.1f}")
print(f"   • Q1 : {Q1:.1f}")
print(f"   • Q3 : {Q3:.1f}")
print(f"   • IQR : {IQR:.1f}")
print(f"   • Bornes IQR : [{lower_bound:.1f}, {upper_bound:.1f}]")

print(f"\n Outliers détectés:")
print(f"   • Méthode IQR (> Q3 + 1.5×IQR) : {len(outliers_iqr_high)} jours")
print(f"   • Méthode IQR (< Q1 - 1.5×IQR) : {len(outliers_iqr_low)} jours")
print(f"   • Méthode Z-score (|z| > 3) : {len(outliers_zscore)} jours")

# Top 10 jours avec le plus d'admissions
print(f"\nTOP 10 JOURS AVEC LE PLUS D'ADMISSIONS:")
print("-"*80)
top_days = daily_admissions.sort_values(ascending=False).head(10)
for date, count in top_days.items():
    day_name = pd.to_datetime(date).day_name()
    print(f"   {date} ({day_name:<10}) : {count:>5} admissions")

# Bottom 10
print(f"\nTOP 10 JOURS AVEC LE MOINS D'ADMISSIONS:")
print("-"*80)
bottom_days = daily_admissions.sort_values(ascending=True).head(10)
for date, count in bottom_days.items():
    day_name = pd.to_datetime(date).day_name()
    print(f"   {date} ({day_name:<10}) : {count:>5} admissions")

# Visualisation avec outliers marqués
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=daily_admissions.index,
    y=daily_admissions.values,
    mode='markers',
    name='Admissions quotidiennes',
    marker=dict(color='lightblue', size=4)
))

# Marquer les outliers
fig.add_trace(go.Scatter(
    x=outliers_iqr_high.index,
    y=outliers_iqr_high.values,
    mode='markers',
    name='Outliers (haute)',
    marker=dict(color='red', size=8, symbol='x')
))

if len(outliers_iqr_low) > 0:
    fig.add_trace(go.Scatter(
        x=outliers_iqr_low.index,
        y=outliers_iqr_low.values,
        mode='markers',
        name='Outliers (basse)',
        marker=dict(color='orange', size=8, symbol='x')
    ))

# Ajouter les limites IQR
fig.add_hline(y=upper_bound, line_dash="dash", line_color="red", 
              annotation_text=f"Limite supérieure ({upper_bound:.0f})")
fig.add_hline(y=lower_bound, line_dash="dash", line_color="orange",
              annotation_text=f"Limite inférieure ({lower_bound:.0f})")
fig.add_hline(y=daily_admissions.mean(), line_dash="dot", line_color="green",
              annotation_text=f"Moyenne ({daily_admissions.mean():.0f})")

fig.update_layout(
    title='<b>Détection d\'Anomalies - Admissions Quotidiennes</b>',
    xaxis_title='Date',
    yaxis_title='Nombre d\'admissions',
    template='plotly_white',
    height=600,
    hovermode='x unified'
)
fig.show()

# ## ANALYSE MULTIVARIÉE AVANCÉE
# 
# ### Analyse des motifs d'admission


# Analyse des motifs d'admission
motif_stats = df_admission.groupby('motif_principal').agg({
    'date_entree': 'count',
    'service': lambda x: x.mode()[0] if len(x) > 0 else 'N/A',
    'departement_patient': lambda x: x.mode()[0] if len(x) > 0 else 'N/A',
    'mode_entree': lambda x: x.mode()[0] if len(x) > 0 else 'N/A'
}).rename(columns={'date_entree': 'total'})

motif_stats['pct'] = (motif_stats['total'] / len(df_admission)) * 100
motif_stats = motif_stats.sort_values('total', ascending=False)

print("="*120)
print("TOP 30 MOTIFS D'ADMISSION".center(120))
print("="*120)
print(f"\n{'Motif':<45} {'Admissions':<15} {'%':<8} {'Service':<35} {'Mode':<15}")
print("-"*120)

for motif, row in motif_stats.head(30).iterrows():
    print(f"{motif:<45} {row['total']:>12,}  {row['pct']:>6.2f}%  {row['service']:<35} {row['mode_entree']:<15}")
# Visualisation Top 25
top_motifs = motif_stats.head(25)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Top 25 Motifs', 'Distribution par Mode d\'Entrée'),
    specs=[[{'type': 'bar'}, {'type': 'sunburst'}]],
    column_widths=[0.6, 0.4]
)

# Barplot des top motifs
fig.add_trace(
    go.Bar(
        y=top_motifs.index,
        x=top_motifs['total'],
        orientation='h',
        marker_color='steelblue',
        text=top_motifs['total'],
        textposition='outside'
    ),
    row=1, col=1
)

# Sunburst : Mode -> Pôle -> Top motifs
sunburst_data = df_admission[df_admission['motif_principal'].isin(top_motifs.head(15).index)]
sunburst_df = sunburst_data.groupby(['mode_entree', 'service', 'motif_principal']).size().reset_index(name='count')

fig.add_trace(
    go.Sunburst(
        labels=sunburst_df['motif_principal'].tolist() + sunburst_df['service'].unique().tolist() + sunburst_df['mode_entree'].unique().tolist(),
        parents=sunburst_df['service'].tolist() + [sunburst_df['mode_entree'].iloc[0]]*len(sunburst_df['service'].unique()) + ['']*len(sunburst_df['mode_entree'].unique()),
        values=sunburst_df['count'].tolist() + [0]*len(sunburst_df['service'].unique()) + [0]*len(sunburst_df['mode_entree'].unique()),
    ),
    row=1, col=2
)

fig.update_layout(height=1000, showlegend=False, title_text="<b>Analyse des Motifs d'Admission</b>")
fig.show()

# ### Analyse comparative : Jours de semaine vs Weekend


# Comparaison Semaine vs Weekend
weekday_data = df_admission[~df_admission['est_weekend']]
weekend_data = df_admission[df_admission['est_weekend']]

print("="*100)
print("COMPARAISON : JOURS DE SEMAINE vs WEEKEND".center(100))
print("="*100)

print(f"\nVolume:")
print(f"   • Jours de semaine : {len(weekday_data):,} admissions ({len(weekday_data)/len(df_admission)*100:.1f}%)")
print(f"   • Weekend : {len(weekend_data):,} admissions ({len(weekend_data)/len(df_admission)*100:.1f}%)")

print(f"\nMoyennes quotidiennes:")
weekday_daily = weekday_data.groupby('date_entree').size()
weekend_daily = weekend_data.groupby('date_entree').size()
print(f"   • Jours de semaine : {weekday_daily.mean():.1f} ± {weekday_daily.std():.1f}")
print(f"   • Weekend : {weekend_daily.mean():.1f} ± {weekend_daily.std():.1f}")

# Test statistique (t-test)
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(weekday_daily, weekend_daily)
print(f"\nTest t de Student:")
print(f"   • t-statistique : {t_stat:.4f}")
print(f"   • p-value : {p_value:.6f}")
if p_value < 0.05:
    print(f"    Différence SIGNIFICATIVE entre semaine et weekend")
else:
    print(f"   PAS de différence significative")

# Répartition par pôle
print(f"\nTop 5 Pôles - Semaine:")
weekday_poles = weekday_data['service'].value_counts().head(5)
for pole, count in weekday_poles.items():
    pct = (count / len(weekday_data)) * 100
    print(f"   {pole:<50} {count:>7,} ({pct:>5.2f}%)")

print(f"\nTop 5 Pôles - Weekend:")
weekend_poles = weekend_data['service'].value_counts().head(5)
for pole, count in weekend_poles.items():
    pct = (count / len(weekend_data)) * 100
    print(f"   {pole:<50} {count:>7,} ({pct:>5.2f}%)")

# Visualisations comparatives
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Distribution Volumes', 'Distribution Quotidienne',
                    'Top Pôles - Semaine', 'Top Pôles - Weekend'),
    specs=[[{'type': 'bar'}, {'type': 'box'}],
           [{'type': 'bar'}, {'type': 'bar'}]]
)

# Volumes totaux
fig.add_trace(
    go.Bar(x=['Semaine', 'Weekend'], 
           y=[len(weekday_data), len(weekend_data)],
           marker_color=['steelblue', 'coral'],
           text=[len(weekday_data), len(weekend_data)],
           textposition='outside'),
    row=1, col=1
)

# Box plots quotidiens
fig.add_trace(go.Box(y=weekday_daily.values, name='Semaine', marker_color='steelblue'), row=1, col=2)
fig.add_trace(go.Box(y=weekend_daily.values, name='Weekend', marker_color='coral'), row=1, col=2)

# 3. Top pôles semaine
fig.add_trace(
    go.Bar(y=weekday_poles.index, x=weekday_poles.values, 
           orientation='h', marker_color='steelblue', showlegend=False),
    row=2, col=1
)

# Top pôles weekend
fig.add_trace(
    go.Bar(y=weekend_poles.index, x=weekend_poles.values,
           orientation='h', marker_color='coral', showlegend=False),
    row=2, col=2
)

fig.update_layout(height=900, showlegend=False, 
                  title_text="<b>Analyse Comparative : Semaine vs Weekend</b>")
fig.show()

# ## INSIGHTS
# 
# ### Synthèse des découvertes clés


# Tableau de synthèse final
print("="*100)
print("SYNTHÈSE FINALE - INSIGHTS PRINCIPAUX".center(100))
print("="*100)

insights = [
    {
        'Catégorie': 'Volume',
        'Insight': f"Total de {len(df_admission):,} admissions/passages en 2024",
        'Impact': 'Élevé',
        'Action': 'Planification des ressources'
    },
    {
        'Catégorie': 'Temporel',
        'Insight': f"Moyenne de {daily_admissions.mean():.0f} admissions/jour (±{daily_admissions.std():.0f})",
        'Impact': 'Élevé',
        'Action': 'Staffing dynamique'
    },
    {
        'Catégorie': 'Saisonnalité',
        'Insight': f"Pic en {monthly_admissions.idxmax()} ({monthly_admissions.max():,} admissions)",
        'Impact': 'Moyen',
        'Action': 'Anticipation saisonnière'
    },
    {
        'Catégorie': 'Géographie',
        'Insight': f"Top origine: {geo_stats.index[0]} ({geo_stats.iloc[0]['pct']:.1f}%)",
        'Impact': 'Moyen',
        'Action': 'Partenariats locaux'
    },
    {
        'Catégorie': 'Pôle',
        'Insight': f"Pôle dominant: {poles_analysis.index[0]} ({poles_analysis.iloc[0]['pct_total']:.1f}%)",
        'Impact': 'Élevé',
        'Action': 'Allocation ressources ciblée'
    },
    {
        'Catégorie': 'Anomalies',
        'Insight': f"{len(outliers_iqr_high)} jours avec pics anormaux détectés",
        'Impact': 'Élevé',
        'Action': 'Plan de crise'
    },
    {
        'Catégorie': 'Variabilité',
        'Insight': f"CV = {(daily_admissions.std()/daily_admissions.mean()*100):.1f}%",
        'Impact': 'Moyen',
        'Action': 'Flexibilité opérationnelle'
    }
]

insights_df = pd.DataFrame(insights)
print(f"\n{insights_df.to_string(index=False)}")