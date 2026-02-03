import streamlit as st
import plotly.express as px
from config import SECONDARY_BLUE
from utils import get_logistique_data, predict_future_admissions


def render_simulator(daily_ts, model_lgbm):
    st.markdown("## SIMULATEUR DE CRISE & RESILIENCE")
    
    c_sim_1, c_sim_2 = st.columns([1, 2])
    
    with c_sim_1:
        st.markdown("### Parametres du Scenario")
        intensite = st.slider("Intensite du Choc (% augmentation)", 0, 50, 10)
        ressource_type = st.radio("Ressource Critique Focus", ["Lits (Rea/Med)", "Effectif Soignant", "Stocks Pharma"])
        
        sim_days = st.number_input("Horizon de Simulation (Jours)", 7, 30, 14)
        
        if st.button("Lancer la Simulation"):
            st.session_state.run_sim = True
            
    with c_sim_2:
        if st.session_state.get('run_sim', False) and model_lgbm:
            daily_series_sim = daily_ts.asfreq('D', fill_value=0)
            _, future_preds = predict_future_admissions(daily_series_sim, model_lgbm)
            avg_predicted = future_preds.mean()
            
            # Calcul stress
            stress_load = avg_predicted * (1 + intensite/100)
            
            # Initialisation metriques
            utilization = 0.0
            depletion_days = 30.0
            fig_sim = None
            
            # Affichage Metriques Baseline
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Baseline Modèle", f"{avg_predicted:.1f}/j")
            m_col2.metric("Charge Simulée", f"{stress_load:.1f}/j", delta=f"+{intensite}%", delta_color="inverse")
            
            # Decompactage & Filtrage Temps Reel
            df_lits_raw, df_perso_raw, _, df_stocks_raw = get_logistique_data()
            latest_date = df_perso_raw['date'].max()
            
            df_lits = df_lits_raw[df_lits_raw['date'] == latest_date]
            df_perso = df_perso_raw[df_perso_raw['date'] == latest_date]
            df_stocks = df_stocks_raw[df_stocks_raw['date'] == latest_date]
            
            # Logique Ressources
            if "Lits" in ressource_type:
                # Top 10 poles par capacite pour l'instantanné
                total_capacity = df_lits.nlargest(10, 'lits_totaux')['lits_totaux'].sum()
                occup_base = df_lits.nlargest(10, 'lits_totaux')['lits_occupes'].sum()
                utilization = (occup_base / total_capacity) * (1 + intensite/200)
                
                m_col3.metric("Saturation Lits", f"{min(utilization*100, 100):.1f}%")
                
                # Viz dummy saturation chart
                fig_sim = px.bar(x=['Capacite Actuelle', 'Besoin Simule'], 
                                 y=[total_capacity, total_capacity * utilization],
                                 title="Impact Saturation Lits",
                                 template="plotly_dark", color_discrete_sequence=[SECONDARY_BLUE, 'red'])
                
                if utilization > 1.0:
                    st.error(f"RUPTURE CAPACITAIRE : Manque {int((utilization-1)*total_capacity)} lits")
                    
            elif "Effectif" in ressource_type:
                total_staff = df_perso['effectif_total'].sum()
                # Tension realiste: (Nouveaux patients simules * facteur) / effectif dispo
                utilization = (stress_load * 0.8) / (total_staff / 20) 
                m_col3.metric("Tension Staff", f"{min(utilization*100, 100):.1f}%")
                
                fig_sim = px.bar(df_perso, 
                                x='categorie', y='effectif_total', title=f"Effectifs Réels au {latest_date.strftime('%d/%m/%Y')}",
                                template="plotly_dark", color_discrete_sequence=[SECONDARY_BLUE])
            else:
                # Logique Stocks
                depletion_days = 30 / (1 + intensite/100)
                m_col3.metric("Autonomie Stocks", f"{depletion_days:.1f} Jours")
                
                if depletion_days < 7:
                    st.error("RUPTURE IMMINENTE (< 7 jours)")
                    
                # Pour line chart, usage historique mais filtrage Top 5 medicaments
                df_viz = df_stocks_raw.copy()
                top_meds = df_viz[df_viz['date'] == latest_date].nlargest(5, 'conso_jour')['medicament'].tolist()
                df_viz = df_viz[df_viz['medicament'].isin(top_meds)].sort_values('date')
                
                fig_sim = px.line(df_viz, x='date', y='conso_jour', color='medicament', title="Dynamique Consommation", template="plotly_dark")
            
            if fig_sim:
                fig_sim.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sim, use_container_width=True)
            
            # Message Status Final
            is_critical = False
            if "Lits" in ressource_type or "Effectif" in ressource_type:
                if utilization > 0.9: is_critical = True
            elif "Stocks" in ressource_type and depletion_days < 10:
                is_critical = True
                
            if is_critical:
                st.markdown("### ⚠️ Plan Blanc Recommandé")
                st.warning("Les seuils de sécurités sont dépassés par le scénario.")
            else:
                st.success("✅ Résilience Confirmée : Situation gérable avec les ressources actuelles.")
