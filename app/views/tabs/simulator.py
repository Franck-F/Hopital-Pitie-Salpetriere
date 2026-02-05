import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from config import SECONDARY_BLUE, ACCENT_RED, PRIMARY_BLUE
from utils import get_logistique_data, predict_future_admissions


def render_simulator(daily_ts, model_lgbm):
    # Header avec style premium
    st.markdown("""
        <div style='text-align: center; margin-bottom: 40px;'>
            <h1 style='font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #00d2ff 0%, #005ba1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                SIMULATEUR DE CRISE
            </h1>
            <p style='font-size: 1.2rem; color: #8899A6; margin-top: -10px;'>
                Anticipez les scénarios critiques et évaluez la résilience du système
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Panel de controle avec design moderne
    st.markdown("<div style='background: rgba(255,255,255,0.03); border-radius: 20px; padding: 30px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 30px;'>", unsafe_allow_html=True)
    
    col_params, col_preview = st.columns([1, 1.5])
    
    with col_params:
        st.markdown("### Configuration du Scénario")
        
        # Type de crise avec scenarios realistes
        scenario_type = st.selectbox(
            "Type de Crise",
            [
                "Épidémie (Grippe/COVID)", 
                "Pic d'Activité Saisonnière",
                "Grève du Personnel",
                "Canicule/Vague de Chaleur", 
                "Accident de Masse",
                "Crise Sanitaire Majeure",
                "Personnalise"
            ],
            help="Sélectionnez un scénario pré-configuré ou personnalisé"
        )
        
        # Presets selon le type avec paramètres réalistes
        if scenario_type == "Épidémie (Grippe/COVID)":
            default_intensity = 40
            default_resources = ["Lits (Rea/Med)", "Effectif Soignant"]
            default_days = 28
            scenario_desc = "Augmentation progressive des admissions respiratoires"
        elif scenario_type == "Pic d'Activité Saisonnière":
            default_intensity = 20
            default_resources = ["Lits (Rea/Med)"]
            default_days = 14
            scenario_desc = "Surcharge temporaire liée aux variations saisonnières"
        elif scenario_type == "Grève du Personnel":
            default_intensity = 15
            default_resources = ["Effectif Soignant"]
            default_days = 7
            scenario_desc = "Réduction de l'effectif disponible impactant la capacité"
        elif scenario_type == "Canicule/Vague de Chaleur":
            default_intensity = 35
            default_resources = ["Lits (Rea/Med)", "Effectif Soignant"]
            default_days = 10
            scenario_desc = "Afflux de patients vulnérables (déshydratation, malaises)"
        elif scenario_type == "Accident de Masse":
            default_intensity = 60
            default_resources = ["Lits (Rea/Med)", "Stocks Pharma"]
            default_days = 3
            scenario_desc = "Afflux massif et soudain de victimes traumatisées"
        elif scenario_type == "Crise Sanitaire Majeure":
            default_intensity = 50
            default_resources = ["Lits (Rea/Med)", "Effectif Soignant", "Stocks Pharma"]
            default_days = 21
            scenario_desc = "Crise multi-dimensionnelle impactant toutes les ressources"
        else:
            default_intensity = 10
            default_resources = ["Lits (Rea/Med)"]
            default_days = 14
            scenario_desc = "Configuration personnalisée"
        
        # Affichage description du scénario
        if scenario_type != "Personnalisé":
            st.info(f"**Scénario** : {scenario_desc}")
        
        st.divider()
        
        intensite = st.slider(
            "Intensité du pic (%)",
            0, 100, default_intensity,
            help="Augmentation prévue du flux d'admissions"
        )
        
        # Sélection multiple de ressources critiques
        ressources_types = st.multiselect(
            "Ressources Critiques (sélection multiple)",
            ["Lits (Rea/Med)", "Effectif Soignant", "Stocks Pharma"],
            default=default_resources,
            help="Sélectionnez une ou plusieurs ressources à analyser simultanément"
        )
        
        # Validation de la sélection
        if not ressources_types:
            st.warning("Veuillez sélectionner au moins une ressource critique")
            ressources_types = ["Lits (Rea/Med)"]  # Fallback
        
        sim_days = st.number_input(
            "Horizon (Jours)",
            7, 60, default_days,
            help="Durée de la simulation"
        )
        
        st.divider()
        
        run_button = st.button(
            "LANCER LA SIMULATION",
            use_container_width=True,
            type="primary"
        )
        
        if run_button:
            st.session_state.run_sim = True
            st.session_state.sim_params = {
                'intensite': intensite,
                'ressources': ressources_types,  # Liste de ressources
                'days': sim_days,
                'scenario': scenario_type
            }
    
    with col_preview:
        st.markdown("### Aperçu des Paramètres")
        
        # Jauge d'intensite visuelle
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=intensite,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Intensite du pic", 'font': {'size': 20, 'color': '#f0f4f8'}},
            delta={'reference': 20, 'increasing': {'color': ACCENT_RED}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': "#f0f4f8"},
                'bar': {'color': SECONDARY_BLUE},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.2)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(0, 255, 0, 0.2)'},
                    {'range': [30, 60], 'color': 'rgba(255, 165, 0, 0.2)'},
                    {'range': [60, 100], 'color': 'rgba(255, 0, 0, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': ACCENT_RED, 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#f0f4f8", 'family': "Outfit"},
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Résultats de la simulation
    if st.session_state.get('run_sim', False) and model_lgbm:
        params = st.session_state.get('sim_params', {})
        intensite = params.get('intensite', 10)
        ressources_types = params.get('ressources', ['Lits (Rea/Med)'])  # Liste de ressources
        sim_days = params.get('days', 14)
        scenario_type = params.get('scenario', 'Personnalise')
        
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(0,210,255,0.1) 0%, rgba(0,91,161,0.1) 100%); 
                        border-radius: 20px; padding: 20px; border: 1px solid rgba(0,210,255,0.3); margin-bottom: 30px;'>
                <h2 style='margin: 0; color: {SECONDARY_BLUE};'> Résultats de Simulation : {scenario_type}</h2>
                <p style='margin: 5px 0 0 0; color: #8899A6;'>Analyse de {len(ressources_types)} ressource(s) critique(s)</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Predictions ML
        daily_series_sim = daily_ts.asfreq('D', fill_value=0)
        future_dates, future_preds = predict_future_admissions(daily_series_sim, model_lgbm, days=sim_days)
        avg_predicted = future_preds.mean()
        stress_load = avg_predicted * (1 + intensite/100)
        
        # KPIs en haut
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.markdown(f"""
                <div style='background: rgba(0,91,161,0.2); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(0,210,255,0.3);'>
                    <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Baseline Modele</p>
                    <h2 style='margin: 5px 0; color: {SECONDARY_BLUE};'>{avg_predicted:.1f}</h2>
                    <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>admissions/jour</p>
                </div>
            """, unsafe_allow_html=True)
        
        with kpi2:
            st.markdown(f"""
                <div style='background: rgba(200,16,46,0.2); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(200,16,46,0.3);'>
                    <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Charge Simulee</p>
                    <h2 style='margin: 5px 0; color: {ACCENT_RED};'>{stress_load:.1f}</h2>
                    <p style='margin: 0; font-size: 0.8rem; color: {ACCENT_RED};'>+{intensite}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with kpi3:
            st.markdown(f"""
                <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                    <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Durée Simulation</p>
                    <h2 style='margin: 5px 0; color: {SECONDARY_BLUE};'>{sim_days}</h2>
                    <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>jours</p>
                </div>
            """, unsafe_allow_html=True)
        
        with kpi4:
            st.markdown(f"""
                <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                    <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Ressources Analysées</p>
                    <h2 style='margin: 5px 0; color: {SECONDARY_BLUE};'>{len(ressources_types)}</h2>
                    <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>dimensions</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Chargement donnees logistiques
        df_lits_raw, df_perso_raw, _, df_stocks_raw = get_logistique_data()
        latest_date = df_perso_raw['date'].max()
        
        df_lits = df_lits_raw[df_lits_raw['date'] == latest_date]
        df_perso = df_perso_raw[df_perso_raw['date'] == latest_date]
        df_stocks = df_stocks_raw[df_stocks_raw['date'] == latest_date]
        
        # Analyse de chaque ressource sélectionnée
        is_critical = False
        critical_resources = []
        
        # Boucle sur chaque ressource selectionnee
        for idx, ressource_type in enumerate(ressources_types):
            st.divider()
            
            # Header pour chaque ressource
            st.markdown(f"""
                <div style='background: rgba(0,210,255,0.1); border-radius: 15px; padding: 15px; margin: 20px 0; border-left: 4px solid {SECONDARY_BLUE};'>
                    <h3 style='margin: 0; color: {SECONDARY_BLUE};'> Analyse : {ressource_type}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            if "Lits" in ressource_type:
                total_capacity = df_lits.nlargest(10, 'lits_totaux')['lits_totaux'].sum()
                occup_base = df_lits.nlargest(10, 'lits_totaux')['lits_occupes'].sum()
                utilization = (occup_base / total_capacity) * (1 + intensite/200)
                
                kpi_col1, kpi_col2 = st.columns(2)
                
                with kpi_col1:
                    color = ACCENT_RED if utilization > 0.9 else SECONDARY_BLUE
                    st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                            <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Saturation Lits</p>
                            <h2 style='margin: 5px 0; color: {color};'>{min(utilization*100, 100):.1f}%</h2>
                            <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>Top 10 poles</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with kpi_col2:
                    deficit = max(0, int((utilization-1)*total_capacity))
                    st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                            <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Deficit Estime</p>
                            <h2 style='margin: 5px 0; color: {ACCENT_RED if deficit > 0 else SECONDARY_BLUE};'>{deficit}</h2>
                            <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>lits manquants</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                if utilization > 0.9:
                    is_critical = True
                    critical_resources.append(ressource_type)
                
                # Visualisation capacite
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Gauge de saturation
                    fig_sat = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=min(utilization*100, 100),
                        title={'text': "Taux de Saturation Projete", 'font': {'size': 18, 'color': '#f0f4f8'}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickcolor': "#f0f4f8"},
                            'bar': {'color': ACCENT_RED if utilization > 0.9 else SECONDARY_BLUE},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "rgba(255,255,255,0.2)",
                            'steps': [
                                {'range': [0, 70], 'color': 'rgba(0, 255, 0, 0.2)'},
                                {'range': [70, 90], 'color': 'rgba(255, 165, 0, 0.2)'},
                                {'range': [90, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
                            ],
                            'threshold': {
                                'line': {'color': ACCENT_RED, 'width': 4},
                                'thickness': 0.75,
                                'value': 95
                            }
                        }
                    ))
                    fig_sat.update_layout(
                        height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font={'color': "#f0f4f8", 'family': "Outfit"}
                    )
                    st.plotly_chart(fig_sat, use_container_width=True)
                
                with viz_col2:
                    # Comparaison capacite
                    df_comp = pd.DataFrame({
                        'Etat': ['Capacite Totale', 'Occupation Actuelle', 'Besoin Simule'],
                        'Lits': [total_capacity, occup_base, total_capacity * utilization]
                    })
                    
                    fig_comp = px.bar(df_comp, x='Etat', y='Lits',
                                     title="Analyse Capacitaire",
                                     template="plotly_dark",
                                     color='Etat',
                                     color_discrete_map={
                                         'Capacite Totale': PRIMARY_BLUE,
                                         'Occupation Actuelle': SECONDARY_BLUE,
                                         'Besoin Simule': ACCENT_RED
                                     })
                    fig_comp.update_layout(
                        height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
            elif "Effectif" in ressource_type:
                total_staff = df_perso['effectif_total'].sum()
                utilization = (stress_load * 0.8) / (total_staff / 20)
                
                kpi_col1, kpi_col2 = st.columns(2)
                
                with kpi_col1:
                    color = ACCENT_RED if utilization > 0.9 else SECONDARY_BLUE
                    st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                            <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Tension Staff</p>
                            <h2 style='margin: 5px 0; color: {color};'>{min(utilization*100, 100):.1f}%</h2>
                            <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>charge prevue</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with kpi_col2:
                    st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                            <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Effectif Total</p>
                            <h2 style='margin: 5px 0; color: {SECONDARY_BLUE};'>{int(total_staff)}</h2>
                            <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>ETP disponibles</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                if utilization > 0.9:
                    is_critical = True
                    critical_resources.append(ressource_type)
                
                # Filtre par pôle/service
                services_list = ['Ensemble de l\'hôpital'] + sorted(df_perso['service'].unique().tolist())
                selected_service = st.selectbox(
                    "Filtrer par Pôle/Service",
                    services_list,
                    key=f"service_filter_{idx}"
                )
                
                # Répartition par catégorie (exclure les lignes 'total')
                if selected_service == 'Ensemble de l\'hôpital':
                    df_perso_detail = df_perso[df_perso['categorie'] != 'total']
                    chart_title = f"Répartition des Effectifs - Ensemble de l'Hôpital au {latest_date.strftime('%d/%m/%Y')}"
                else:
                    df_perso_detail = df_perso[(df_perso['service'] == selected_service) & (df_perso['categorie'] != 'total')]
                    chart_title = f"Répartition des Effectifs - {selected_service} au {latest_date.strftime('%d/%m/%Y')}"
                
                fig_staff = px.bar(df_perso_detail, x='categorie', y='effectif_total',
                                  title=chart_title,
                                  template="plotly_dark",
                                  color='categorie',
                                  color_discrete_sequence=px.colors.qualitative.Set3)
                fig_staff.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                    xaxis_title="Catégorie de Personnel",
                    yaxis_title="Effectif Total (ETP)"
                )
                st.plotly_chart(fig_staff, use_container_width=True)
                
            else:  # Stocks
                depletion_days = 30 / (1 + intensite/100)
                
                kpi_col1, kpi_col2 = st.columns(2)
                
                with kpi_col1:
                    color = ACCENT_RED if depletion_days < 10 else SECONDARY_BLUE
                    st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                            <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Autonomie Stocks</p>
                            <h2 style='margin: 5px 0; color: {color};'>{depletion_days:.1f}</h2>
                            <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>jours restants</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with kpi_col2:
                    alertes = df_stocks['alerte_rupture'].sum()
                    st.markdown(f"""
                        <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.1);'>
                            <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Alertes Actuelles</p>
                            <h2 style='margin: 5px 0; color: {ACCENT_RED if alertes > 0 else SECONDARY_BLUE};'>{int(alertes)}</h2>
                            <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>ruptures detectees</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                if depletion_days < 10:
                    is_critical = True
                    critical_resources.append(ressource_type)
                
                # Evolution consommation
                df_viz = df_stocks_raw.copy()
                top_meds = df_viz[df_viz['date'] == latest_date].nlargest(5, 'conso_jour')['medicament'].tolist()
                df_viz = df_viz[df_viz['medicament'].isin(top_meds)].sort_values('date')
                
                fig_stocks = px.line(df_viz, x='date', y='conso_jour', color='medicament',
                                    title="Evolution Consommation (Top 5 Medicaments)",
                                    template="plotly_dark")
                fig_stocks.update_layout(
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_stocks, use_container_width=True)
        
        # Verdict final
        st.divider()
        
        if is_critical:
            critical_list = ", ".join(critical_resources) if critical_resources else "ressources multiples"
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(200,16,46,0.2) 0%, rgba(255,0,0,0.1) 100%); 
                            border-radius: 20px; padding: 30px; border: 2px solid {ACCENT_RED}; margin-top: 30px;'>
                    <h2 style='margin: 0 0 15px 0; color: {ACCENT_RED};'>PLAN BLANC RECOMMANDE</h2>
                    <p style='font-size: 1.1rem; margin: 0 0 10px 0; color: #f0f4f8;'>
                        Les seuils de securite sont depasses par le scenario simule. Activation des procedures d'urgence conseillee.
                    </p>
                    <p style='font-size: 0.95rem; margin: 0; color: #ff6b6b; font-weight: 600;'>
                        Ressources critiques : {critical_list}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background: linear-gradient(135deg, rgba(0,210,255,0.2) 0%, rgba(0,255,0,0.1) 100%); 
                            border-radius: 20px; padding: 30px; border: 2px solid {SECONDARY_BLUE}; margin-top: 30px;'>
                    <h2 style='margin: 0 0 15px 0; color: {SECONDARY_BLUE};'> RESILIENCE CONFIRMEE</h2>
                    <p style='font-size: 1.1rem; margin: 0; color: #f0f4f8;'>
                        Le systeme peut absorber la charge simulee avec les ressources actuelles. Situation gerable.
                    </p>
                </div>
            """, unsafe_allow_html=True)
