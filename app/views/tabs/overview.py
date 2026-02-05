import streamlit as st
import plotly.express as px
from config import SECONDARY_BLUE, ACCENT_RED, PRIMARY_BLUE


def render_overview(df_adm, daily_ts, df_lits=None, df_perso=None, df_equip=None, df_stocks=None):
    st.markdown("<h2 style='font-weight:800;'>Panorama de l'Activité</h2>", unsafe_allow_html=True)
    
    # KPIs Admissions
    st.markdown("### Activité Hospitalière")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Admissions", f"{len(df_adm):,}")
    m2.metric("Moyenne Quotidienne", f"{daily_ts.mean():.1f}")
    m3.metric("Jour de Pic", f"{daily_ts.max()}")
    
    # Graphique principal
    fig_main = px.line(daily_ts.reset_index(), x='date_entree', y='admissions', 
                       title="Flux d'admissions quotidiens ", 
                       template="plotly_dark", color_discrete_sequence=[SECONDARY_BLUE])
    fig_main.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_main, use_container_width=True)
    
    # KPIs Logistiques
    if df_perso is not None and df_equip is not None and df_stocks is not None:
        st.divider()
        st.markdown("### Ressources et Logistique")
        
        # Calculs des metriques
        latest_date = df_perso['date'].max()
        
        # Effectifs
        df_perso_latest = df_perso[df_perso['date'] == latest_date]
        total_staff = df_perso_latest[df_perso_latest['categorie'] == 'total']['effectif_total'].sum()
        total_present = df_perso_latest[df_perso_latest['categorie'] == 'total']['effectif_present'].sum()
        taux_presence = (total_present / total_staff * 100) if total_staff > 0 else 0
        
        # Equipements
        df_equip_latest = df_equip[df_equip['date'] == latest_date]
        total_equipements = df_equip_latest['quantite_totale'].sum()
        equipements_fonctionnels = df_equip_latest['quantite_fonctionnelle'].sum()
        taux_fonctionnel = (equipements_fonctionnels / total_equipements * 100) if total_equipements > 0 else 0
        
        # Stocks medicaments
        df_stocks_latest = df_stocks[df_stocks['date'] == latest_date]
        total_medicaments = len(df_stocks_latest)
        alertes_rupture = df_stocks_latest['alerte_rupture'].sum()
        
        # Affichage des bulles KPI
        kpi1, kpi2, kpi3 = st.columns(3)
        
        with kpi1:
            presence_color = ACCENT_RED if taux_presence < 85 else SECONDARY_BLUE
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(0,91,161,0.2) 0%, rgba(0,210,255,0.1) 100%); 
                            border-radius: 20px; padding: 25px; text-align: center; border: 2px solid {presence_color};'>
                    <p style='margin: 0; font-size: 0.85rem; color: #8899A6; text-transform: uppercase; letter-spacing: 1px;'>Effectifs Personnel</p>
                    <h1 style='margin: 10px 0 5px 0; color: {SECONDARY_BLUE}; font-size: 2.5rem; font-weight: 800;'>{int(total_staff)}</h1>
                    <p style='margin: 0; font-size: 0.9rem; color: #f0f4f8;'>ETP disponibles</p>
                    <p style='margin: 10px 0 0 0; font-size: 0.85rem; color: {presence_color}; font-weight: 600;'>Taux présence : {taux_presence:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with kpi2:
            equip_color = ACCENT_RED if taux_fonctionnel < 80 else SECONDARY_BLUE
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(0,91,161,0.2) 0%, rgba(0,210,255,0.1) 100%); 
                            border-radius: 20px; padding: 25px; text-align: center; border: 2px solid {equip_color};'>
                    <p style='margin: 0; font-size: 0.85rem; color: #8899A6; text-transform: uppercase; letter-spacing: 1px;'>Matériels</p>
                    <h1 style='margin: 10px 0 5px 0; color: {SECONDARY_BLUE}; font-size: 2.5rem; font-weight: 800;'>{int(total_equipements)}</h1>
                    <p style='margin: 0; font-size: 0.9rem; color: #f0f4f8;'>équipements totaux</p>
                    <p style='margin: 10px 0 0 0; font-size: 0.85rem; color: {equip_color}; font-weight: 600;'>Fonctionnels : {taux_fonctionnel:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with kpi3:
            stock_color = ACCENT_RED if alertes_rupture > 0 else SECONDARY_BLUE
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(0,91,161,0.2) 0%, rgba(0,210,255,0.1) 100%); 
                            border-radius: 20px; padding: 25px; text-align: center; border: 2px solid {stock_color};'>
                    <p style='margin: 0; font-size: 0.85rem; color: #8899A6; text-transform: uppercase; letter-spacing: 1px;'>Stocks Médicaments</p>
                    <h1 style='margin: 10px 0 5px 0; color: {SECONDARY_BLUE}; font-size: 2.5rem; font-weight: 800;'>{total_medicaments}</h1>
                    <p style='margin: 0; font-size: 0.9rem; color: #f0f4f8;'>références suivies</p>
                    <p style='margin: 10px 0 0 0; font-size: 0.85rem; color: {stock_color}; font-weight: 600;'>Alertes rupture : {int(alertes_rupture)}</p>
                </div>
            """, unsafe_allow_html=True)

