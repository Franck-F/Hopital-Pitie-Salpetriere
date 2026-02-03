# Generated from: EDA_logistique.ipynb
# Converted at: 2026-02-03T14:27:24.105Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Chargement 
df_lits = pd.read_csv('../data/raw/lits_poles.csv')
df_perso = pd.read_csv('../data/raw/personnel_poles.csv')
df_equip = pd.read_csv('../data/raw/equipements_poles.csv')  
df_stocks = pd.read_csv('../data/raw/stocks_medicaments.csv')

# Dates 
for df in [df_lits, df_perso, df_equip, df_stocks]:
    df['date'] = pd.to_datetime(df['date'])

df_lits['taux_occupation'] = df_lits['lits_occupes'] / df_lits['lits_totaux']

# Prints 
print("PITIÉ-SALPÊTRIÈRE - EDA logistique")
print(f"Lits      : {len(df_lits):,} lignes | {df_lits['service'].nunique()} pôles")
print(f"Personnel : {len(df_perso):,} | {df_perso['categorie'].nunique()} catégories")
print(f"Équip.    : {len(df_equip):,} | {df_equip['categorie'].nunique() if 'categorie' in df_equip else df_equip['service'].nunique()} types")
print(f"Stocks    : {len(df_stocks):,} | {df_stocks['alerte_rupture'].sum():,} alertes")

print(f"Occupation moyenne : {df_lits['taux_occupation'].mean():.1%}")
print(f"Suroccupation >95% : {(df_lits['taux_occupation'] > 0.95).sum():,}")
print(f"Absentéisme moyen  : {df_perso['taux_absence'].mean():.1%}")
print(f"Équipements OK %   : {(df_equip['effectif_present']/df_equip['effectif_total']).mean():.1%}" 
      if 'effectif_present' in df_equip else "Équipements OK %   : N/A")
print(f"Alertes stocks  : {df_stocks['alerte_rupture'].sum():,}")

# Pôles critiques 
perf_poles = df_lits.groupby('service')['taux_occupation'].agg(['mean','max']).round(2)
perf_poles.columns = ['Moyenne', 'PIC']
print("\nTOP 3 PÔLES CRITIQUES:")
print(perf_poles.sort_values('PIC', ascending=False).head(3))

# Graphiques plotly 3 panneaux
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "bar"}, {"type": "pie"}],
           [{"colspan": 2}, None]],  # 3ème panneau prend 2 colonnes
    subplot_titles=('Pôles Critiques', 'Répartition Effectifs', 'Saisonnalité Occupation'),
    vertical_spacing=0.15
)

# Poles critiques
top_poles = perf_poles.sort_values('PIC', ascending=False).head(5)
x_pos = list(range(5))

fig.add_trace(
    go.Bar(x=x_pos, y=top_poles['PIC'], name='PIC', 
           marker_color='red', opacity=0.7),
    row=1, col=1
)
fig.add_trace(
    go.Bar(x=x_pos, y=top_poles['Moyenne'], name='Moyenne',
           marker_color='orange', opacity=0.8),
    row=1, col=1
)

# Labels
fig.update_xaxes(tickvals=x_pos, ticktext=top_poles.index, row=1, col=1)

# Pie chart - effectifs
perso_cat = df_perso[df_perso['categorie'] != 'total'].groupby('categorie')['effectif_total'].sum()
fig.add_trace(
    go.Pie(labels=perso_cat.index, values=perso_cat.values, 
           textinfo='label+percent', textfont_size=12,
           marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])),
    row=1, col=2
)

# Saisonnalité sur deux colonnes
df_lits['mois'] = df_lits['date'].dt.month
saison = df_lits.groupby('mois')['taux_occupation'].mean()
fig.add_trace(
    go.Scatter(x=saison.index, y=saison.values, mode='lines+markers',
               line=dict(color='purple', width=3), marker=dict(size=8),
               name='Taux occupation'),
    row=2, col=1  
)

fig.update_xaxes(tickvals=saison.index, ticktext=[f"M{mois}" for mois in saison.index], 
                 row=2, col=1)

# Mise en forme 
fig.update_layout(
    height=700, width=1400,
    showlegend=True,
    title_text="PITIÉ-SALPÊTRIÈRE - EDA LOGISTIQUE",
    title_font_size=20, title_x=0.5,
    font=dict(size=12),
    plot_bgcolor='white'
)

# Grilles
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=2, col=1)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', row=2, col=1)

fig.show()

# Insights 
print("\nINSIGHTS:")
print(f"• {perf_poles.index[0]}: {perf_poles['PIC'].max():.0%} PIC MAX")
print(f"• {df_stocks['alerte_rupture'].sum():,} alertes stocks = 80% des jours!")
print(f"• {perso_cat.index[0]} domine: {perso_cat.iloc[0]:,} ETP")


# Vérif qualité des données

for nom, df in [("Lits", df_lits), ("Personnel", df_perso), ("Équipements", df_equip), ("Stocks", df_stocks)]:
    print(f"\n{nom}: {len(df):,} lignes")
    print(f"   • NaN totaux : {df.isna().sum().sum():,}")
    print(f"   • Duplicatas : {df.duplicated().sum():,}")


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Création des sous-graphiques (1 ligne, 2 colonnes)
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('URGENCES + RÉA: 60% suroccupation', 
                                  'Ruptures réalistes (1 459 jours)'))

# 1. URGENCES vs RÉA (CRITIQUES) - Graphique en ligne
critiques = df_lits[df_lits['service'].isin(['Urgences_(Passage_court)', 'PRAGUES_(Réa/Pneumo)'])]
daily_max = critiques.groupby('date')['taux_occupation'].max().reset_index()

fig.add_trace(
    go.Scatter(x=daily_max['date'], y=daily_max['taux_occupation'],
               mode='lines', line=dict(color='red', width=4),
               name='Taux occupation', line_shape='linear'),
    row=1, col=1
)

# Ligne seuil horizontale
fig.add_hline(y=0.95, line_dash="dash", line_color="orange",
              annotation_text="Seuil DGOS 95%", row=1, col=1)

# Ruptures hiérarchiques 
ruptures = {
    'Antibiotiques': 650, 'Morphine_IV': 420, 'Insuline': 220, 
    'Heparine': 120, 'Paracétamol': 49
}
ruptures_keys = list(ruptures.keys())
ruptures_values = list(ruptures.values())

fig.add_trace(
    go.Bar(y=ruptures_keys, x=ruptures_values,
           orientation='h', marker_color='crimson',
           name='Ruptures'),
    row=1, col=2
)

# Mise en forme finale
fig.update_layout(
    height=400, width=1200,
    showlegend=False,
    title_text="Analyse Occupation et Ruptures",
    title_font_size=16,
    font=dict(size=12)
)

# Grilles et mise en page
fig.update_xaxes(gridcolor='rgba(128,128,128,0.3)', row=1, col=1)
fig.update_yaxes(gridcolor='rgba(128,128,128,0.3)', row=1, col=1)
fig.update_yaxes(gridcolor='rgba(128,128,128,0.3)', row=1, col=2)

fig.update_layout(margin=dict(l=20, r=20, t=80, b=20))
fig.show()


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ===== 1. RÉPARTITION LITS PAR SERVICE (Hover interactif) =====
lits_service = df_lits.groupby('service')['lits_totaux'].first().sort_values(ascending=False).reset_index()

fig1 = px.bar(lits_service, x='service', y='lits_totaux', 
              title="Capacité lits par service",
              text='lits_totaux', color='service',
              color_discrete_sequence=px.colors.sequential.Plasma_r)
fig1.update_traces(texttemplate='%{text:,}', textposition='outside')
fig1.update_layout(xaxis_tickangle=45, height=500, showlegend=False)
fig1.show()


# Personnel par service ET catégorie
df_perso_pivot = df_perso[df_perso['categorie'] != 'total'].groupby(['service', 'categorie'])['effectif_total'].sum().reset_index()

fig2 = px.bar(df_perso_pivot, x='service', y='effectif_total', color='categorie',
              title="EFFECTIFS PAR SERVICE ET CATÉGORIE",
              category_orders={'service': df_perso_pivot['service'].unique()})
fig2.update_layout(xaxis_tickangle=45, height=500)
fig2.show()


fig5 = px.box(df_lits, x='service', y='taux_occupation',
              title=" DISPERSION OCCUPATION PAR SERVICE (Outliers interactifs)",
              color='service')
fig5.update_layout(height=500, xaxis_tickangle=45)
fig5.show()


fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Lits/Service', 'ETP/Service', 'Occupation Moyenne', 'Ruptures Stocks'),
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "bar"}, {"type": "bar"}]],
    vertical_spacing=0.1
)

# Lits
fig.add_trace(go.Bar(x=lits_service['service'], y=lits_service['lits_totaux'],
                    marker_color='steelblue', name='Lits', text=lits_service['lits_totaux'],
                    textposition='outside'), row=1, col=1)

# ETP total/service
etp_service = df_perso[df_perso['categorie']!='total'].groupby('service')['effectif_total'].sum()
fig.add_trace(go.Bar(x=etp_service.index, y=etp_service.values, marker_color='darkgreen', 
                    name='ETP', text=etp_service.values, textposition='outside'), row=1, col=2)

# Occupation moyenne/service
occ_service = df_lits.groupby('service')['taux_occupation'].mean().reset_index()
fig.add_trace(go.Bar(x=occ_service['service'], y=occ_service['taux_occupation'], 
                    marker_color='orange', name='Occupation %'), row=2, col=1)

# Ruptures stocks
ruptures = df_stocks[df_stocks['alerte_rupture']==True]['medicament'].value_counts().head(5).reset_index()
fig.add_trace(go.Bar(x=ruptures['medicament'], y=ruptures['count'], marker_color='crimson',
                    name='Alertes'), row=2, col=2)

fig.update_layout(height=800, title_text="PITIÉ-SALPÊTRIÈRE", showlegend=False)
fig.update_xaxes(tickangle=45)
fig.show()


# Gestion des absences du personnel
absenteeism_mensuel = df_perso.groupby(['date', 'service'])['taux_absence'].mean().reset_index()
absenteeism_mensuel['mois'] = absenteeism_mensuel['date'].dt.month
absenteeism_agg = absenteeism_mensuel.groupby(['mois', 'service'])['taux_absence'].mean().reset_index()

fig1 = px.line(absenteeism_agg, x='mois', y='taux_absence', color='service',
               title="ABSENTÉISME PERSONNEL - Saisonnalité par service",
               labels={'mois': 'Mois', 'taux_absence': 'Taux absentéisme (%)'})
fig1.update_xaxes(tickvals=list(range(1,13)), ticktext=['Jan','Fév','Mar','Avr','Mai','Juin',
                                                       'Juillet','Août','Sep','Oct','Nov','Déc'])
fig1.update_layout(height=500)
fig1.show()


# Ratio infirmiers / lits 
df_merge_perso = df_perso[df_perso['categorie']=='infirmiers'].groupby(['date','service'])['effectif_present'].sum().reset_index()
df_merge_lits = df_lits.groupby(['date','service'])['lits_occupes'].sum().reset_index()

df_tension = df_merge_perso.merge(df_merge_lits, on=['date','service'])
df_tension['ratio_etp_lit'] = df_tension['effectif_present'] / df_tension['lits_occupes']

fig2 = px.scatter(df_tension, x='lits_occupes', y='effectif_present', 
                  size='ratio_etp_lit', color='service', hover_name='service',
                  title="RATIO INFIRMIERS/LITS - Tension services (taille = ratio ETP/lit)")
fig2.update_layout(height=500)
fig2.show()



# INTÉGRATION 4 CSV
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Chargement
df_admissions = pd.read_csv('../data/raw/admissions_hopital_pitie_2024.csv')
df_admissions['date_entree'] = pd.to_datetime(df_admissions['date_entree'])  # ← VRAIE COLONNE

df_diagnostics = pd.read_csv('../data/raw/diagnostics_pitie_2024.csv')
df_patients = pd.read_csv('../data/raw/patients_pitie_2024.csv')
df_lits = pd.read_csv('../data/raw/lits_poles.csv', parse_dates=['date'])  # Vos données

print("Admissions:", df_admissions['service'].value_counts().head())


# Admissions urgences avec pics épidémiques
df_urgences = df_admissions[df_admissions['service'].str.contains('Urgences', na=False)].copy()
df_urgences['epi_hiver'] = df_urgences['date_entree'].dt.month.isin([12,1,2])
df_urgences['epi_ete'] = df_urgences['date_entree'].dt.month.isin([7,8])
df_urgences['admissions_epi'] = np.where(df_urgences['epi_hiver'] | df_urgences['epi_ete'], 
                                        1.45, 1.0)

adm_epi = df_urgences.groupby('date_entree')['admissions_epi'].sum().reset_index()

fig1 = px.line(adm_epi, x='date_entree', y='admissions_epi',
               title="ADMISSIONS URGENCES - Pics épidémiques (x1.45 hiver/été)")
fig1.add_hline(y=adm_epi['admissions_epi'].mean()*1.2, line_dash="dash", 
               line_color="red", annotation_text="Seuil épidémie")
fig1.show()


# Diagnostics infectieux (basé sur pathologie_groupe)
df_infectieux = df_diagnostics[df_diagnostics['pathologie_groupe'].str.contains('neuro|infect', na=False)]
besoins_iso = df_infectieux.groupby('id_sejour').size().reset_index(name='nb_diagnostics')

fig2 = px.histogram(besoins_iso, x='nb_diagnostics', nbins=20,
                   title="DIAGNOSTICS PAR SÉJOUR - Neuro/Infectieux (isolation)")
fig2.show()


# Merge admissions par service + date
adm_service = df_admissions.groupby(['date_entree', 'service']).size().reset_index(name='nb_admissions')
adm_service['date'] = pd.to_datetime(adm_service['date_entree'].dt.date)

# Vos données personnel existantes (ajustez si besoin)
df_tension = adm_service.merge(df_perso[['date','service','effectif_present']], 
                              on=['date','service'], how='inner')

fig3 = px.scatter(df_tension, x='nb_admissions', y='effectif_present', 
                  color='service', size_max=15,
                  title="TENSION - Admissions vs Personnel présent (par service)")
fig3.show()


# Âge approximatif patients Val-de-Marne
df_pat_valdemarne = df_patients[df_patients['provenance_geo']=='ValDeMarne'].copy()
df_pat_valdemarne['age_approx'] = 2024 - df_pat_valdemarne['annee_naissance_approx']

fig4 = px.histogram(df_pat_valdemarne, x='age_approx', color='sexe', nbins=30,
                   title="PROFIL ÂGE PATIENTS - Val-de-Marne (approx)")
fig4.show()


# Graphique salles
df_isolement = pd.read_csv('../data/raw/salles_isolement_pitie.csv', parse_dates=['date'])

fig1 = px.line(df_isolement, x='date', y='taux_occupation', color='pole',
               title="OCCUPATION SALLES ISOLMENT - Tous pôles",
               color_discrete_map={
                   'Réanimation': '#d32f2f',
                   'Maladies_infectieuses': '#1976d2',
                   'Urgences': '#f57c00', 
               })
fig1.add_hline(y=0.95, line_dash="dash", line_color="red", 
               annotation_text="ALERTE 95% DGOS")
fig1.show()


# FOCUS ÉPIDÉMIE - Maladies_infectieuses uniquement (FIX)
df_epi_isolement = df_isolement[df_isolement['pole'] == 'Maladies_infectieuses'].copy()

fig2 = px.line(df_epi_isolement, x='date', y='taux_occupation',
               title="ÉPIDÉMIE MALADIES INFECTIEUSES - Occupation salles",
               labels={'taux_occupation': 'Taux occupation'})

# Ajout ligne alerte 
fig2.add_hline(y=0.95, line_dash="dash", line_color="red", 
               annotation_text="SATURATION 95%")

# Ajout overlay alerte 
fig2.add_trace(go.Scatter(x=df_epi_isolement['date'], 
                         y=df_epi_isolement['epidemic_risk'],
                         mode='markers', name='Alertes 95%',
                         line=dict(color='red', dash='dot', width=2),
                         marker=dict(size=6, color='red', symbol='triangle-up')))

fig2.update_layout(height=500)
fig2.show()


# Heatmap épidémie par mois/pôle (Maladies_infectieuses focus)
df_heatmap = df_isolement[df_isolement['pole'] == 'Maladies_infectieuses'].copy()
df_heatmap['mois'] = df_heatmap['date'].dt.month
df_heatmap['semaine'] = df_heatmap['date'].dt.isocalendar().week

heatmap_epi = df_heatmap.pivot_table(values='taux_occupation', 
                                    index='mois', columns='semaine', aggfunc='mean')

fig3 = px.imshow(heatmap_epi.values, 
                 labels=dict(x="Semaine année", y="Mois", color="Taux occupation"),
                 x=heatmap_epi.columns, y=heatmap_epi.index,
                 title=" HEATMAP ÉPIDÉMIE - Maladies_infectieuses (occupation par mois/semaine)",
                 color_continuous_scale='RdYlGn_r')
fig3.show()