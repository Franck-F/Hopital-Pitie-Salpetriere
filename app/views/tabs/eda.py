import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import scipy.stats as scipy_stats
from config import SECONDARY_BLUE, ACCENT_RED, PRIMARY_BLUE


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
    st.markdown(f"""
        <h2 style='background: linear-gradient(135deg, {SECONDARY_BLUE} 0%, {PRIMARY_BLUE} 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ANALYSE DES ADMISSIONS 2024-2025
        </h2>
    """, unsafe_allow_html=True)
    
    # Dashboard KPIs compact
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Admissions", f"{len(df_adm):,}")
    with kpi2:
        st.metric("Services/Poles", f"{df_adm['service'].nunique()}")
    with kpi3:
        st.metric("Moyenne Quotidienne", f"{daily_ts.mean():.1f}")
    with kpi4:
        st.metric("Pic Journalier", f"{daily_ts.max()}")
    
    # Detection anomalies en haut (important)
    Q1 = daily_ts.quantile(0.25)
    Q3 = daily_ts.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outliers = daily_ts[daily_ts > upper_bound]
    
    st.markdown("### Detection d'Anomalies et Pics de Charge")
    fig_ano = go.Figure()
    fig_ano.add_trace(go.Scatter(x=daily_ts.index, y=daily_ts.values, mode='lines', name='Admissions', line=dict(color=SECONDARY_BLUE, width=1)))
    fig_ano.add_trace(go.Scatter(x=outliers.index, y=outliers.values, mode='markers', name='Pics Anormaux', marker=dict(color=ACCENT_RED, size=8, symbol='x')))
    fig_ano.add_hline(y=upper_bound, line_dash="dash", line_color=ACCENT_RED, annotation_text="Seuil Alerte (IQR)")
    fig_ano.update_layout(height=350, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=30))
    st.plotly_chart(fig_ano, use_container_width=True)
    
    # Grille 2x2 pour distributions
    st.markdown("### Distributions Principales")
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 Poles
        pole_c = df_adm['service'].value_counts().head(10)
        fig_poles = px.bar(x=pole_c.values, y=pole_c.index, orientation='h',
                          title='Top 10 Poles/Services', template="plotly_dark",
                          color_discrete_sequence=[SECONDARY_BLUE])
        fig_poles.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_poles, use_container_width=True)
        
    with col2:
        # Modes d'entree
        mode_c = df_adm['mode_entree'].value_counts()
        fig_modes = px.pie(values=mode_c.values, names=mode_c.index,
                          title="Modes d'Entree", template="plotly_dark", hole=0.4)
        fig_modes.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_modes, use_container_width=True)
    
    # Expanders pour analyses detaillees
    with st.expander("Patterns Temporels et Saisonnalite"):
        pc1, pc2 = st.columns(2)
        with pc1:
            jour_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fig_box_j = go.Figure()
            for j in jour_ordre:
                d_j = df_adm[df_adm['jour_semaine_nom'] == j].groupby('date_entree').size()
                fig_box_j.add_trace(go.Box(y=d_j.values, name=j[:3]))
            fig_box_j.update_layout(title="Variabilite par Jour", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box_j, use_container_width=True)
        with pc2:
            pivot_h = df_adm.groupby(['jour_semaine', 'mois']).size().unstack(fill_value=0)
            fig_heat_adm = px.imshow(pivot_h.values, labels=dict(x="Mois", y="Jour", color="Volume"),
                                     x=pivot_h.columns, y=jour_ordre, title="Intensite Semaine x Mois",
                                     color_continuous_scale='YlOrRd', template="plotly_dark")
            fig_heat_adm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_heat_adm, use_container_width=True)
    
    with st.expander("Analyse Semaine vs Weekend"):
        weekday_d = df_adm[~df_adm['est_weekend']].groupby('date_entree').size()
        weekend_d = df_adm[df_adm['est_weekend']].groupby('date_entree').size()
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
    
    with st.expander("Hierarchie des Motifs (Sunburst)"):
        top_m_list = df_adm['motif_principal'].value_counts().head(15).index
        sun_df = df_adm[df_adm['motif_principal'].isin(top_m_list)].groupby(['mode_entree', 'service', 'motif_principal']).size().reset_index(name='count')
        fig_sun_adm = px.sunburst(sun_df, path=['mode_entree', 'service', 'motif_principal'], values='count',
                                  title="Mode -> Pole -> Top Motifs",
                                  template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_sun_adm.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sun_adm, use_container_width=True)
    
    with st.expander("Evolution par Pole (Top 5)"):
        top_poles_names = df_adm['service'].value_counts().head(5).index
        fig_poles_evol = go.Figure()
        for pole in top_poles_names:
            pole_daily = df_adm[df_adm['service'] == pole].groupby('date_entree').size().asfreq('D', fill_value=0)
            pole_ma = pole_daily.rolling(window=7).mean()
            fig_poles_evol.add_trace(go.Scatter(x=pole_ma.index, y=pole_ma.values, mode='lines', name=pole))
        fig_poles_evol.update_layout(height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_poles_evol, use_container_width=True)
    
    with st.expander("Repartition Detaillee (Table)"):
        type_counts = df_adm['service'].value_counts().reset_index()
        type_counts.columns = ['Service', 'Nombre d\'admissions']
        type_counts['Pourcentage (%)'] = (type_counts['Nombre d\'admissions'] / len(df_adm) * 100).round(2)
        st.dataframe(
            type_counts.style.background_gradient(subset=['Nombre d\'admissions'], cmap='Blues'),
            use_container_width=True,
            hide_index=True
        )

def render_logistics_subtab(df_lits, df_perso, df_stocks):
    st.markdown(f"""
        <h2 style='background: linear-gradient(135deg, {SECONDARY_BLUE} 0%, {PRIMARY_BLUE} 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ANALYSE LOGISTIQUE & RESSOURCES
        </h2>
    """, unsafe_allow_html=True)
    
    # KPIs compact
    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Occupation Moyenne", f"{df_lits['taux_occupation'].mean():.1%}")
    l2.metric("Suroccupation (>95%)", f"{(df_lits['taux_occupation'] > 0.95).sum():,}")
    l3.metric("Ratio Inf/Lit", f"{(df_perso[df_perso['categorie']=='infirmier']['effectif_total'].sum() / df_lits['lits_totaux'].sum()):.2f}")
    l4.metric("Alertes Stocks", f"{df_stocks['alerte_rupture'].sum():,}")
    
    # Grille 2x2 principale
    st.markdown("### Capacite et Effectifs")
    lc1, lc2 = st.columns(2)
    
    with lc1:
        # Visualisation interactive avec couleurs par service
        lits_service = df_lits.groupby('service')['lits_totaux'].first().sort_values(ascending=False).reset_index()
        fig_lits = px.bar(lits_service, x='service', y='lits_totaux', 
                         title="Capacite lits par service",
                         text='lits_totaux', color='service',
                         color_discrete_sequence=px.colors.sequential.Plasma_r,
                         template="plotly_dark")
        fig_lits.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_lits.update_layout(xaxis_tickangle=45, height=450, showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_lits, use_container_width=True)

        
    with lc2:
        perso_cat = df_perso[df_perso['categorie'] != 'total'].groupby('categorie')['effectif_total'].sum().reset_index()
        fig_p_cat = px.bar(perso_cat, x='categorie', y='effectif_total', color='categorie',
                           title="Effectifs par Corps de Metier (ETP)",
                           template="plotly_dark")
        fig_p_cat.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_p_cat, use_container_width=True)

    
    # Expanders pour details
    with st.expander("Salles d'Isolement & Vigilance Epidemique"):
        ic1, ic2 = st.columns(2)
        with ic1:
            iso_services = ['Urgences_(Passage_court)', 'PRAGUES_(Réa/Pneumo)', 'Infectiologie']
            df_iso = df_lits[df_lits['service'].isin(iso_services)]
            fig_iso = px.line(df_iso, x='date', y='taux_occupation', color='service',
                              title="Tension dans les Services Haute Vigilance",
                              template="plotly_dark")
            fig_iso.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_iso, use_container_width=True)
        with ic2:
            st.info("Les services identifies disposent de chambres a pression negative. Une occupation depassant 90% declenche le protocole 'Alerte Epidemique'.")
            st.metric("Taux d'Alerte Global (ISO)", f"{(df_iso['taux_occupation'] > 0.9).mean():.1%}")
    
    with st.expander("Gestion des Stocks et Ruptures"):
        sc1, sc2 = st.columns(2)
        with sc1:
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
            abs_df = df_perso.copy()
            abs_df['mois'] = abs_df['date'].dt.month
            abs_agg = abs_df.groupby(['mois', 'service'])['taux_absence'].mean().reset_index()
            fig_abs = px.line(abs_agg, x='mois', y='taux_absence', color='service',
                             title="Saisonnalite de l'Absenteisme (%)",
                             template="plotly_dark")
            fig_abs.update_xaxes(tickvals=list(range(1,13)), ticktext=['Jan','Fev','Mar','Avr','Mai','Juin','Juil','Aout','Sep','Oct','Nov','Dec'])
            fig_abs.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_abs, use_container_width=True)

def render_sejour_subtab(df_pat, df_sej, df_diag):
    st.markdown(f"""
        <h2 style='background: linear-gradient(135deg, {SECONDARY_BLUE} 0%, {PRIMARY_BLUE} 100%); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ANALYSE DES SEJOURS & PARCOURS PATIENTS
        </h2>
    """, unsafe_allow_html=True)
    
    # KPIs compact
    datasets = [(df_pat, "Patients"), (df_sej, "Sejours"), (df_diag, "Diagnostics")]
    all_comp = []
    for df, name in datasets:
        completeness = (1 - df.isna().mean()) * 100
        comp_val = completeness.mean()
        all_comp.append(comp_val)
    avg_comp = sum(all_comp) / len(all_comp)
    
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Completude Moyenne", f"{avg_comp:.1f}%")
    q2.metric("Total Sejours", f"{len(df_sej):,}")
    q3.metric("DMS Moyenne", f"{df_sej['duree_jours'].mean():.1f} jours")
    q4.metric("Codes/Sejour", f"{(len(df_diag)/len(df_sej)):.1f}")
    
    # Grille 2x2 principale
    st.markdown("### Profil Demographique")
    dc1, dc2 = st.columns(2)
    
    with dc1:
        fig_sexe = px.pie(df_pat, names='sexe', title="Repartition par Sexe",
                          template="plotly_dark", hole=0.4,
                          color_discrete_map={'M': '#2c3e50', 'F': '#e74c3c'})
        fig_sexe.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sexe, use_container_width=True)
        
    with dc2:
        fig_age_sej = px.histogram(df_sej, x="age", nbins=40, marginal="violin",
                                 title="Pyramide des Ages a l'Admission",
                                 template="plotly_dark", color_discrete_sequence=['#3498db'])
        fig_age_sej.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_age_sej, use_container_width=True)
    
    # Repartition ages par pole (important, donc visible)
    st.markdown("### Repartition des Ages par Pole")
    df_sej['age_bin'] = pd.cut(df_sej['age'], bins=[0, 18, 45, 65, 105], labels=['Enfants', 'Adultes', 'Seniors', 'Grand Age'])
    pole_order = df_sej['pole'].value_counts().index.tolist()
    age_pole = df_sej.groupby(['pole', 'age_bin']).size().reset_index(name='count')
    fig_age_pole = px.bar(age_pole, x="count", y="pole", color="age_bin", orientation='h',
                           category_orders={"pole": pole_order[::-1]},
                          title="Structure Demographique des Admissions par Pole",
                          template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu_r)
    fig_age_pole.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title="Tranche d'age")
    st.plotly_chart(fig_age_pole, use_container_width=True)
    
    # Expanders pour analyses detaillees
    with st.expander("Analyse des Pathologies (CIM-10)"):
        dg1, dg2 = st.columns(2)
        with dg1:
            diag_p = df_diag.groupby("pathologie_groupe").size().reset_index(name='count').sort_values('count', ascending=True)
            fig_p_p = px.bar(diag_p, x='count', y='pathologie_groupe', orientation='h',
                               title="Principaux Groupes de Pathologies", template="plotly_dark",
                               color='count', color_continuous_scale="Tealgrn")
            fig_p_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_p_p, use_container_width=True)
        with dg2:
            repartition = df_diag["type_diagnostic"].value_counts().reset_index()
            repartition.columns = ["type", "count"]
            fig_donut = px.pie(repartition, values="count", names="type", hole=0.5,
                               title="Repartition des Types de Diagnostics",
                               color_discrete_sequence=['#2C3E50', '#E74C3C'],
                               template="plotly_dark")
            fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True)
    
    with st.expander("Correlation Age vs Duree de Sejour"):
        sample_size = min(500, len(df_sej))
        sample_sej = df_sej.sample(n=sample_size, random_state=42)
        fig_scatter = px.scatter(sample_sej, x="age", y="duree_jours", color="pole", size="age",
                                 title=f"Repartition Age / DMS (Echantillon {sample_size} patients)",
                                 template="plotly_dark", opacity=0.7)
        fig_scatter.add_hline(y=df_sej['duree_jours'].mean(), line_dash="dot", annotation_text="DMS Moyenne")
        fig_scatter.add_vline(x=df_sej['age'].mean(), line_dash="dot", annotation_text="Age Moyen")
        fig_scatter.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with st.expander("Profiling Multidimensionnel (Radar)"):
        radar_raw = df_sej.groupby('pole').agg({
            'age': 'mean',
            'duree_jours': 'mean',
            'id_sejour': 'count'
        }).reset_index()
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
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with st.expander("Heatmap Tension Horaire"):
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
        st.info("Cette heatmap permet d'identifier les pics d'activite journaliers. Une concentration rouge indique un flux critique.")
        st.metric("Heure de Pointe (Moyenne)", f"{df_sej['heure'].mode()[0]}h00")
