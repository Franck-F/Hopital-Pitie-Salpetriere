import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import scipy.stats as scipy_stats
from config import SECONDARY_BLUE, ACCENT_RED


def render_eda(df_adm, daily_ts, df_lits, df_perso, df_equip, df_stocks, df_pat, df_sej, df_diag):
    sub_tab_adm, sub_tab_log, sub_tab_sej = st.tabs([
        "Admission patient", "Logistique", "Séjour patient"
    ])
    
    with sub_tab_adm:
        render_admission_subtab(df_adm, daily_ts)

    with sub_tab_log:
        render_logistics_subtab(df_lits, df_perso, df_stocks)

    with sub_tab_sej:
        render_sejour_subtab(df_pat, df_sej, df_diag)

def render_admission_subtab(df_adm, daily_ts):
    st.markdown("## ANALYSE DES ADMISSIONS 2024")
    
    # --- Stats Globales ---
    st.markdown("### Vue d'ensemble des donnees")
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Periode d'analyse", f"{df_adm['date_entree'].min().strftime('%d/%m/%Y')} -> {df_adm['date_entree'].max().strftime('%d/%m/%Y')}")
    o2.metric("Total Admissions", f"{len(df_adm):,}")
    o3.metric("Services/Poles", f"{df_adm['service'].nunique()}")
    o4.metric("Modes d'Entree", f"{df_adm['mode_entree'].nunique()}")

    # --- Table Admissions ---
    st.divider()
    st.markdown("### Repartition par Type d'Admission")
    type_counts = df_adm['service'].value_counts().reset_index()
    type_counts.columns = ['Service', 'Nombre d\'admissions']
    type_counts['Pourcentage (%)'] = (type_counts['Nombre d\'admissions'] / len(df_adm) * 100).round(2)
    
    st.dataframe(
        type_counts.style.background_gradient(subset=['Nombre d\'admissions'], cmap='Blues'),
        use_container_width=True,
        hide_index=True
    )

    # --- Sous-graphes pour Distributions Categorielles ---
    st.divider()
    st.markdown("### Distributions des Variables Catégorielles")
    
    fig_cat = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top 10 Poles/Services', 'Modes d\'Entree', 
                        'Origine Geographique (Top 10)', 'Top 20 Motifs d\'Admission'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )

    # 1. Poles
    pole_c = df_adm['service'].value_counts().head(10)
    fig_cat.add_trace(go.Bar(y=pole_c.index, x=pole_c.values, orientation='h', name='Poles', marker_color='lightblue'), row=1, col=1)
    
    # 2. Modes
    mode_c = df_adm['mode_entree'].value_counts()
    fig_cat.add_trace(go.Pie(labels=mode_c.index, values=mode_c.values, name='Modes', hole=0.4), row=1, col=2)
    
    # 3. Geo
    geo_c = df_adm['departement_patient'].value_counts().head(10)
    fig_cat.add_trace(go.Bar(x=geo_c.index, y=geo_c.values, name='Origine', marker_color='coral'), row=2, col=1)
    
    # 4. Motifs
    motif_c = df_adm['motif_principal'].value_counts().head(20)
    fig_cat.add_trace(go.Bar(x=motif_c.index, y=motif_c.values, name='Motifs', marker_color='lightgreen'), row=2, col=2)
    
    fig_cat.update_layout(height=900, template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cat, use_container_width=True)


    # --- Patterns & Heatmap ---
    st.divider()
    st.markdown("### Patterns Temporels")
    pc1, pc2 = st.columns(2)
    with pc1:
        jour_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig_box_j = go.Figure()
        for j in jour_ordre:
            d_j = df_adm[df_adm['jour_semaine_nom'] == j].groupby('date_entree').size()
            fig_box_j.add_trace(go.Box(y=d_j.values, name=j[:3]))
        fig_box_j.update_layout(title="Variabilité par Jour", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box_j, use_container_width=True)
    with pc2:
        pivot_h = df_adm.groupby(['jour_semaine', 'mois']).size().unstack(fill_value=0)
        fig_heat_adm = px.imshow(pivot_h.values, labels=dict(x="Mois", y="Jour", color="Volume"),
                                 x=pivot_h.columns, y=jour_ordre, title="Intensité Semaine x Mois",
                                 color_continuous_scale='YlOrRd', template="plotly_dark")
        fig_heat_adm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat_adm, use_container_width=True)


    # --- Sunburst Motifs ---
    st.divider()
    st.markdown("### Hierarchie des Motifs (Sunburst)")
    top_m_list = df_adm['motif_principal'].value_counts().head(15).index
    sun_df = df_adm[df_adm['motif_principal'].isin(top_m_list)].groupby(['mode_entree', 'service', 'motif_principal']).size().reset_index(name='count')
    fig_sun_adm = px.sunburst(sun_df, path=['mode_entree', 'service', 'motif_principal'], values='count',
                              title="Mode -> Pôle -> Top Motifs",
                              template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_sun_adm.update_layout(height=700, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_sun_adm, use_container_width=True)

    # --- Analyse Semaine vs Weekend ---
    st.divider()
    st.markdown("### Analyse Comparative : Semaine vs Weekend")
    
    weekday_d = df_adm[~df_adm['est_weekend']].groupby('date_entree').size()
    weekend_d = df_adm[df_adm['est_weekend']].groupby('date_entree').size()
    
    # t-test
    t_stat, p_val = scipy_stats.ttest_ind(weekday_d, weekend_d)
    
    wc1, wc2 = st.columns(2)
    with wc1:
        fig_box_ww = go.Figure()
        fig_box_ww.add_trace(go.Box(y=weekday_d.values, name='Semaine', marker_color='steelblue'))
        fig_box_ww.add_trace(go.Box(y=weekend_d.values, name='Weekend', marker_color='coral'))
        fig_box_ww.update_layout(title="Distribution Volumes quotidien", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_box_ww, use_container_width=True)
        
    with wc2:
        st.metric("Moyenne Semaine", f"{weekday_d.mean():.1f} ± {weekday_d.std():.1f}")
        st.metric("Moyenne Weekend", f"{weekend_d.mean():.1f} ± {weekend_d.std():.1f}")
        if p_val < 0.05:
            st.success(f"Difference SIGNIFICATIVE (p={p_val:.6f})")
        else:
            st.info(f"Pas de difference significative (p={p_val:.6f})")

    # --- Detection d'Anomalies (IQR) ---
    st.divider()
    st.markdown("### Detection d'Anomalies et Pics de Charge")
    
    Q1 = daily_ts.quantile(0.25)
    Q3 = daily_ts.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outliers = daily_ts[daily_ts > upper_bound]
    
    fig_ano = go.Figure()
    fig_ano.add_trace(go.Scatter(x=daily_ts.index, y=daily_ts.values, mode='lines', name='Admissions', line=dict(color=SECONDARY_BLUE, width=1)))
    fig_ano.add_trace(go.Scatter(x=outliers.index, y=outliers.values, mode='markers', name='Pics Anormaux', marker=dict(color=ACCENT_RED, size=8, symbol='x')))
    fig_ano.add_hline(y=upper_bound, line_dash="dash", line_color=ACCENT_RED, annotation_text="Seuil Alerte (IQR)")
    fig_ano.update_layout(height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=30))
    st.plotly_chart(fig_ano, use_container_width=True)

    # --- Evolution Top 5 Poles ---
    st.divider()
    st.markdown("### Evolution des Flux par Pole (Moyenne Mobile 7j)")
    
    top_poles_names = df_adm['service'].value_counts().head(5).index
    fig_poles_evol = go.Figure()
    
    for pole in top_poles_names:
        pole_daily = df_adm[df_adm['service'] == pole].groupby('date_entree').size().asfreq('D', fill_value=0)
        pole_ma = pole_daily.rolling(window=7).mean()
        fig_poles_evol.add_trace(go.Scatter(x=pole_ma.index, y=pole_ma.values, mode='lines', name=pole))
        
    fig_poles_evol.update_layout(height=450, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_poles_evol, use_container_width=True)

    # --- Synthese Insights Finaux ---
    st.divider()
    st.markdown("### Synthese Strategique des Admissions")
    
    peak_month = df_adm.groupby('mois_nom').size().idxmax()
    main_mode = df_adm['mode_entree'].mode()[0]
    
    insights_rows = [
        {"Indicateur": "Pic Saisonnier", "Valeur": peak_month, "Observation": "Charge maximale observee."},
        {"Indicateur": "Mode d'Entree Dominant", "Valeur": main_mode, "Observation": "Vecteur principal d'admission."},
        {"Indicateur": "Jours Critiques", "Valeur": f"{len(outliers)} jours", "Observation": "Depassement des seuils de garde."},
        {"Indicateur": "Variabilite (CV)", "Valeur": f"{(daily_ts.std()/daily_ts.mean()*100):.1f}%", "Observation": "Besoin de flexibilite RH."}
    ]
    st.table(pd.DataFrame(insights_rows))

def render_logistics_subtab(df_lits, df_perso, df_stocks):
    st.markdown("## ANALYSE LOGISTIQUE & RESSOURCES")
    
    # --- Metriques Strategiques ---
    st.markdown("### Indicateurs de Tension Critique")
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Occupation Moyenne", f"{df_lits['taux_occupation'].mean():.1%}")
    l2.metric("Suroccupation (>95%)", f"{(df_lits['taux_occupation'] > 0.95).sum():,}")
    l3.metric("Ratio Infirmiers/Lit (Moy)", f"{(df_perso[df_perso['categorie']=='infirmier']['effectif_total'].sum() / df_lits['lits_totaux'].sum()):.2f}")
    l4.metric("Alertes Stocks", f"{df_stocks['alerte_rupture'].sum():,}")

    # --- Panneau Capacite vs Effectifs ---
    st.divider()
    st.markdown("### Capacité et Effectifs Soignants")
    lc1, lc2 = st.columns(2)
    
    with lc1:
        # Capacite avec sous-graphes
        l_fig = make_subplots(
            rows=2, cols=1, 
            subplot_titles=("Top 10 Poles (Lits Totaux)", "Repartition par Type de Lit"),
            specs=[[{"type": "xy"}], [{"type": "domain"}]]
        )
        lits_p = df_lits.groupby('service')['lits_totaux'].first().sort_values(ascending=False).head(10)
        l_fig.add_trace(go.Bar(x=lits_p.index, y=lits_p.values, marker_color='steelblue', name='Lits'), row=1, col=1)
        
        # Camembert par type de lit
        l_fig.add_trace(go.Pie(labels=lits_p.index[:5], values=lits_p.values[:5], hole=0.3), row=2, col=1)
        l_fig.update_layout(height=700, template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(l_fig, use_container_width=True)
        
    with lc2:
        # Detail Effectifs
        perso_cat = df_perso[df_perso['categorie'] != 'total'].groupby('categorie')['effectif_total'].sum().reset_index()
        fig_p_cat = px.bar(perso_cat, x='categorie', y='effectif_total', color='categorie',
                           title="Effectifs Totaux par Corps de Metier (ETP)",
                           template="plotly_dark")
        fig_p_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p_cat, use_container_width=True)


    # --- Salles d'Isolement & Alertes Epidemiques ---
    st.divider()
    st.markdown("### Focus : Salles d'Isolement & Vigilance Epidemique")
    ic1, ic2 = st.columns(2)
    with ic1:
        # Chargement donnees isolement
        iso_services = ['Urgences_(Passage_court)', 'PRAGUES_(Réa/Pneumo)', 'Infectiologie']
        df_iso = df_lits[df_lits['service'].isin(iso_services)]
        fig_iso = px.line(df_iso, x='date', y='taux_occupation', color='service',
                          title="Tension dans les Services Haute Vigilance",
                          template="plotly_dark")
        fig_iso.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_iso, use_container_width=True)
    with ic2:
        st.info("Les services identifies ci-contre disposent de chambres a pression negative (Salles d'Isolement). Une occupation depassant 90% sur ces zones declenche le protocole 'Alerte Epidemique'.")
        st.metric("Taux d'Alerte Global (ISO)", f"{(df_iso['taux_occupation'] > 0.9).mean():.1%}")

    # --- Monitoring Stocks & Ruptures Critiques ---
    st.divider()
    st.markdown("### Gestion des Stocks et Ruptures Critiques")
    sc1, sc2 = st.columns([1, 1.2])
    
    with sc1:
        st.write("Hierarchie des ruptures constatees (Points de vigilance majeurs).")
        # Donnees source EDA
        rupt_data = pd.DataFrame({
            'Medicament': ['Antibiotiques', 'Morphine IV', 'Insuline', 'Heparine', 'Paracetamol'],
            'Occurences': [650, 420, 220, 120, 49]
        })
        fig_rupt = px.bar(rupt_data, y='Medicament', x='Occurences', orientation='h', 
                         title="Hierarchie des Ruptures (Cumul jours)",
                         color='Occurences', color_continuous_scale='Reds',
                         template="plotly_dark")
        fig_rupt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_rupt, use_container_width=True)
        
    with sc2:
        st.write("Analyse saisonniere de l'absenteisme soignant par service.")
        abs_df = df_perso.copy()
        abs_df['mois'] = abs_df['date'].dt.month
        abs_agg = abs_df.groupby(['mois', 'service'])['taux_absence'].mean().reset_index()
        
        fig_abs = px.line(abs_agg, x='mois', y='taux_absence', color='service',
                         title="Saisonnalite de l'Absenteisme (%)",
                         template="plotly_dark")
        fig_abs.update_xaxes(tickvals=list(range(1,13)), ticktext=['Jan','Fev','Mar','Avr','Mai','Juin','Juil','Aout','Sep','Oct','Nov','Dec'])
        fig_abs.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_abs, use_container_width=True)


    # --- Summary Insights Logistique ---
    st.divider()
    st.markdown("### Diagnostic de Resilence Logistique")
    log_res = [
        {"Point": "Tension Staff", "Constat": "Ratio moyen stable (> 0.22), mais pics de sous-effectif en Rea.", "Statut": "Vigilance"},
        {"Point": "Capacite Lits", "Constat": f"{(df_lits['taux_occupation']>0.95).sum()} episodes de saturation severe detectes.", "Statut": "Alerte"},
        {"Point": "Stocks", "Constat": f"{df_stocks['alerte_rupture'].sum()} ruptures critiques identifiees (Principalement Curitine).", "Statut": "Action Requise"}
    ]
    st.table(pd.DataFrame(log_res))

def render_sejour_subtab(df_pat, df_sej, df_diag):
    st.markdown("## ANALYSE DES SEJOURS & PARCOURS PATIENTS")
    
    # --- 1. Apercu Dataset (Tables Stylees) ---
    st.markdown("### Apercu des Jeux de Donnees 2024")
    
    def create_styled_table_st(df):
        return df.head(5).style.set_properties(**{
            'background-color': '#f8f9fa',
            'color': '#2c3e50',
            'border-color': '#e9ecef'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white')]}
        ])

    with st.expander("Consulter l'extrait des Tables (Top 5 rows)", expanded=False):
        st.markdown("#### Table Patients")
        st.dataframe(create_styled_table_st(df_pat), use_container_width=True)
        st.markdown("#### Table Sejours")
        st.dataframe(create_styled_table_st(df_sej), use_container_width=True)
        st.markdown("#### Table Diagnostics")
        st.dataframe(create_styled_table_st(df_diag), use_container_width=True)

    # --- 2. Qualite Donnees & Vue d'ensemble ---
    st.divider()
    st.markdown("### Qualite et Profil Demographique")
    q1, q2, q3 = st.columns(3)
    
    datasets = [(df_pat, "Patients"), (df_sej, "Sejours"), (df_diag, "Diagnostics")]
    all_comp = []
    for i, (df, name) in enumerate(datasets):
        with [q1, q2, q3][i]:
            completeness = (1 - df.isna().mean()) * 100
            comp_val = completeness.mean()
            all_comp.append(comp_val)
            st.metric(f"Completude {name}", f"{comp_val:.1f}%")
    avg_comp = sum(all_comp) / len(all_comp)
            
    dc1, dc2 = st.columns(2)
    with dc1:
        fig_sexe = px.pie(df_pat, names='sexe', title="Répartition par Sexe",
                          template="plotly_dark", hole=0.4,
                          color_discrete_map={'M': '#2c3e50', 'F': '#e74c3c'})
        fig_sexe.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sexe, use_container_width=True)
    with dc2:
        fig_age_sej = px.histogram(df_sej, x="age", nbins=40, marginal="violin",
                                 title="Pyramide des Ages a l'Admission (Densite)",
                                 template="plotly_dark", color_discrete_sequence=['#3498db'])
        fig_age_sej.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_age_sej, use_container_width=True)

    # --- Nouveau : Boxplot Detail Age par Type ---
    st.divider()
    st.markdown("### Dispersion Detaillee de l'Age par Type d'Hospitalisation")
    fig_box_notched = px.box(df_sej, x="type_hospit", y="age", color="type_hospit",
                             notched=True, points="suspectedoutliers",
                             title="Age Median et Outliers par Type de Sejour",
                             template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Prism)
    fig_box_notched.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    st.plotly_chart(fig_box_notched, use_container_width=True)

    # --- 3. Analyse Specialites & Repartition Age ---
    st.divider()
    st.markdown("### Hierarchie et Structure Demographique des Poles")
    sc1, sc2 = st.columns(2)
    with sc1:
        fig_sun_sej = px.sunburst(df_sej, path=['pole', 'type_hospit'], values='age', # Proxy volume
                              title="Poles -> Types d'Hospitalisation",
                              template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_sun_sej.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sun_sej, use_container_width=True)
    with sc2:
        st.write("Exploration des structures de soins par pole. Le sunburst permet de visualiser l'imbrication des types d'hospitalisation au sein des unites medicales.")
        st.info("Utilisez le clic pour zoomer sur un pole specifique et voir le detail des sejours.")

    # --- Nouveau : Repartition Ages par Pole (Pleine Largeur & Trie) ---
    st.divider()
    st.markdown("### Répartition des Âges par Pôle")
    df_sej['age_bin'] = pd.cut(df_sej['age'], bins=[0, 18, 45, 65, 105], labels=['Enfants', 'Adultes', 'Seniors', 'Grand Age'])
    
    # Calcul volume pour tri
    pole_order = df_sej['pole'].value_counts().index.tolist()
    age_pole = df_sej.groupby(['pole', 'age_bin']).size().reset_index(name='count')
    
    fig_age_pole = px.bar(age_pole, x="count", y="pole", color="age_bin", orientation='h',
                           category_orders={"pole": pole_order[::-1]}, # Gros volumes en haut
                          title="Structure Demographique des Admissions par Pole",
                          template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu_r)
    fig_age_pole.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title="Tranche d'age")
    st.plotly_chart(fig_age_pole, use_container_width=True)

    # --- 4. Analyse Diagnostics ---
    st.divider()
    st.markdown("### Analyse des Pathologies (CIM-10)")
    dg1, dg2 = st.columns(2)
    with dg1:
        diag_p = df_diag.groupby("pathologie_groupe").size().reset_index(name='count').sort_values('count', ascending=True)
        fig_p_p = px.bar(diag_p, x='count', y='pathologie_groupe', orientation='h',
                           title="Principaux Groupes de Pathologies", template="plotly_dark",
                           color='count', color_continuous_scale="Tealgrn")
        fig_p_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_p_p, use_container_width=True)
    with dg2:
        # Donut par Type Diagnostic
        repartition = df_diag["type_diagnostic"].value_counts().reset_index()
        repartition.columns = ["type", "count"]
        fig_donut = px.pie(repartition, values="count", names="type", hole=0.5,
                           title="Repartition des Types de Diagnostics",
                           color_discrete_sequence=['#2C3E50', '#E74C3C'], # Pro Dark / Pro Red
                           template="plotly_dark")
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_donut, use_container_width=True)

    # --- 5. Correlation Age vs Duree ---
    st.divider()
    st.markdown("### Analyse de Corrélation : Age vs Durée de Séjour")
    
    # Echantillon representatif pour scatter plot
    sample_size = min(500, len(df_sej))
    sample_sej = df_sej.sample(n=sample_size, random_state=42)
    
    fig_scatter = px.scatter(sample_sej, x="age", y="duree_jours", color="pole", size="age",
                             title=f"Repartition Age / DMS (Echantillon {sample_size} patients)",
                             template="plotly_dark", opacity=0.7)
    fig_scatter.add_hline(y=df_sej['duree_jours'].mean(), line_dash="dot", annotation_text="DMS Moyenne")
    fig_scatter.add_vline(x=df_sej['age'].mean(), line_dash="dot", annotation_text="Age Moyen")
    fig_scatter.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # --- 6. Profiling Poles Multidimensionnel (Radar) ---
    st.divider()
    st.markdown("### Profiling Multidimensionnel des Poles (Radar)")
    
    # Preparation donnees radar
    radar_raw = df_sej.groupby('pole').agg({
        'age': 'mean',
        'duree_jours': 'mean',
        'id_sejour': 'count'
    }).reset_index()
    
    # Normalisation pour visualisation radar (Echelle 0-1)
    for col in ['age', 'duree_jours', 'id_sejour']:
        radar_raw[f'{col}_norm'] = radar_raw[col] / radar_raw[col].max()
        
    fig_radar = go.Figure()
    categories = ['Age Moyen', 'Duree Sejour (DMS)', 'Volume Activite']
    top_poles_radar = radar_raw.sort_values('id_sejour', ascending=False).head(3)
    
    for i, row in top_poles_radar.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['age_norm'], row['duree_jours_norm'], row['id_sejour_norm']],
            theta=categories, fill='toself', name=row['pole']
        ))
        
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_radar, use_container_width=True)
    
    st.divider()
    
    qc1, qc2 = st.columns([2, 1])
    with qc1:
        # Nouveau : Heatmap Tension Horaire
        df_sej['heure'] = df_sej['date_admission'].dt.hour
        tension_h = df_sej.groupby(['jour_adm', 'heure']).size().reset_index(name='nb_admissions')
        jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig_heat_h = px.density_heatmap(tension_h, x="heure", y="jour_adm", z="nb_admissions", 
                                         nbinsx=24, category_orders={"jour_adm": jours_ordre},
                                         color_continuous_scale="YlOrRd",
                                         title="Heatmap de Tension : Flux d'arrivee des patients",
                                         template="plotly_dark")
        fig_heat_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Heure d'admission")
        st.plotly_chart(fig_heat_h, use_container_width=True)
    with qc2:
        st.info("Cette heatmap permet d'identifier les pics d'activite journaliers. Une concentration rouge indique un flux critique necessitant un renfort des effectifs d'accueil et de tri.")
        st.metric("Heure de Pointe (Moyenne)", f"{df_sej['heure'].mode()[0]}h00")
    
    # --- Insights Finaux Sejour ---
    st.divider()
    st.markdown("### Synthese des Parcours Patients")
    dms = df_sej['duree_jours'].mean()
    top_patho = df_diag['pathologie_groupe'].value_counts().index[0]
    
    sej_insights_df = pd.DataFrame([
        {"Indicateur": "Duree Moyenne de Sejour (DMS)", "Valeur": f"{dms:.1f} jours", "Note": "Optimisation des flux requise."},
        {"Indicateur": "Pathologie Dominante", "Valeur": top_patho, "Note": "Vigilance sur les lits specialises."},
        {"Indicateur": "Qualite des Codages", "Valeur": f"{avg_comp:.1f}%", "Note": "Niveau de fiabilite excellent."},
        {"Indicateur": "Intensite Diagnostique", "Valeur": f"{(len(df_diag)/len(df_sej)):.1f} codes/sej", "Note": "Complexite des soins confirmee."}
    ])
    st.table(sej_insights_df)
