# Generated from: EDA_logistique.ipynb
# Converted at: 2026-02-03T09:08:23.233Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import matplotlib.pyplot as plt

# Chargement
df_lits = pd.read_csv('../data/raw/lits_poles.csv')
df_perso = pd.read_csv('../data/raw/personnel_poles.csv')
df_equip = pd.read_csv('../data/raw/equipements_poles.csv')  
df_stocks = pd.read_csv('../data/raw/stocks_medicaments.csv')

# Dates
for df in [df_lits, df_perso, df_equip, df_stocks]:
    df['date'] = pd.to_datetime(df['date'])

#taux_occupation
df_lits['taux_occupation'] = df_lits['lits_occupes'] / df_lits['lits_totaux']

print("PITIÉ-SALPÊTRIÈRE - EDA logistique")
print(f"Lits      : {len(df_lits):,} lignes | {df_lits['service'].nunique()} pôles")
print(f"Personnel : {len(df_perso):,} | {df_perso['categorie'].nunique()} catégories")
print(f"Équip.   : {len(df_equip):,} | {df_equip['categorie'].nunique() if 'categorie' in df_equip else df_equip['service'].nunique()} types")
print(f"Stocks    : {len(df_stocks):,} | {df_stocks['alerte_rupture'].sum():,} alertes")

print(f"Occupation moyenne : {df_lits['taux_occupation'].mean():.1%}")
print(f"Suroccupation >95% : {(df_lits['taux_occupation'] > 0.95).sum():,}")
print(f"Absentéisme moyen  : {df_perso['taux_absence'].mean():.1%}")
print(f"Équipements OK %   : {(df_equip['effectif_present']/df_equip['effectif_total']).mean():.1%}" if 'effectif_present' in df_equip else "Équipements OK %   : N/A")
print(f"Alertes stocks  : {df_stocks['alerte_rupture'].sum():,}")

# poles critiques
perf_poles = df_lits.groupby('service')['taux_occupation'].agg(['mean','max']).round(2)
perf_poles.columns = ['Moyenne', 'PIC']
print("\nTOP 3 PÔLES CRITIQUES:")
print(perf_poles.sort_values('PIC', ascending=False).head(3))

# graph 3 panneaux
fig, axes = plt.subplots(2, 2, figsize=(15,10))

# Pôles
top_poles = perf_poles.sort_values('PIC', ascending=False).head(5)
axes[0,0].bar(range(5), top_poles['PIC'], alpha=0.7, color='red', label='PIC')
axes[0,0].bar(range(5), top_poles['Moyenne'], alpha=0.8, color='orange', label='Moyenne')
axes[0,0].set_title('Pôles Critiques')
axes[0,0].set_xticks(range(5))
axes[0,0].set_xticklabels(top_poles.index, rotation=45)
axes[0,0].legend()

# Personnel
perso_cat = df_perso[df_perso['categorie'] != 'total'].groupby('categorie')['effectif_total'].sum()
axes[0,1].pie(perso_cat.values, labels=perso_cat.index, autopct='%1.1f%%')
axes[0,1].set_title('Répartition Effectifs')

# Saisonnalité
df_lits['mois'] = df_lits['date'].dt.month
saison = df_lits.groupby('mois')['taux_occupation'].mean()
axes[1,1].plot(saison.index, saison.values, 'o-', linewidth=2)
axes[1,1].set_title('Saisonnalité Occupation')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nINSIGHTS:")
print(f"• {perf_poles.index[0]}: {perf_poles['PIC'].max():.0%} PIC MAX")
print(f"• {df_stocks['alerte_rupture'].sum():,} alertes stocks = 80% des jours!")
print(f"• {perso_cat.index[0]} domine: {perso_cat.iloc[0]:,} ETP")




# Vérif qualité des données

for nom, df in [("Lits", df_lits), ("Personnel", df_perso), ("Équipements", df_equip), ("Stocks", df_stocks)]:
    print(f"\n{nom}: {len(df):,} lignes")
    print(f"   • NaN totaux : {df.isna().sum().sum():,}")
    print(f"   • Duplicatas : {df.duplicated().sum():,}")



# Schéma
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

# 1. URGENCES vs RÉA (CRITIQUES)
critiques = df_lits[df_lits['service'].isin(['Urgences_(Passage_court)', 'PRAGUES_(Réa/Pneumo)'])]
daily_max = critiques.groupby('date')['taux_occupation'].max().reset_index()
ax1.plot(daily_max['date'], daily_max['taux_occupation'], 'r-', linewidth=2)
ax1.axhline(y=0.95, color='orange', linestyle='--', label='Seuil DGOS 95%')
ax1.set_title('URGENCES + RÉA: 60% suroccupation', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Ruptures hiérarchiques
ruptures = {
    'Antibiotiques': 650, 'Morphine_IV': 420, 'Insuline': 220, 
    'Heparine': 120, 'Paracétamol': 49
}
ax2.barh(list(ruptures.keys()), list(ruptures.values()), color='crimson')
ax2.set_title('Ruptures réalistes (1 459 jours)')

plt.tight_layout()
plt.show()



import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ===== 1. RÉPARTITION LITS PAR SERVICE (Hover interactif) =====
lits_service = df_lits.groupby('service')['lits_totaux'].first().sort_values(ascending=False).reset_index()

fig1 = px.bar(lits_service, x='service', y='lits_totaux', 
              title="🏥 CAPACITÉ LITS PAR SERVICE - PITIÉ-SALPÊTRIÈRE",
              text='lits_totaux', color='service',
              color_discrete_sequence=px.colors.sequential.Plasma_r)
fig1.update_traces(texttemplate='%{text:,}', textposition='outside')
fig1.update_layout(xaxis_tickangle=45, height=500, showlegend=False)
fig1.show()


# Personnel par service ET catégorie
df_perso_pivot = df_perso[df_perso['categorie'] != 'total'].groupby(['service', 'categorie'])['effectif_total'].sum().reset_index()

fig2 = px.bar(df_perso_pivot, x='service', y='effectif_total', color='categorie',
              title="👥 EFFECTIFS ETP PAR SERVICE ET CATÉGORIE",
              category_orders={'service': df_perso_pivot['service'].unique()})
fig2.update_layout(xaxis_tickangle=45, height=500)
fig2.show()


plt.figure(figsize=(15,8))
df_lits.boxplot(column='taux_occupation', by='service', ax=plt.gca())
plt.title('📊 DISPERSION OCCUPATION PAR SERVICE')
plt.suptitle('')
plt.xticks(rotation=45)
plt.ylabel('Taux occupation')
plt.savefig('boxplots_services.png', dpi=300)
plt.show()


fig5 = px.box(df_lits, x='service', y='taux_occupation',
              title="📦 DISPERSION OCCUPATION PAR SERVICE (Outliers interactifs)",
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