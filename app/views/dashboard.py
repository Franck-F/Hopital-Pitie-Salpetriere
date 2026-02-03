import streamlit as st
import pandas as pd
from config import LOGO_PATH, SECONDARY_BLUE
from utils import (
    get_admission_data, get_logistique_data, get_patient_sejour_data, 
    load_champion_model
)
from views.tabs.overview import render_overview
from views.tabs.eda import render_eda
from views.tabs.ml import render_ml
from views.tabs.simulator import render_simulator
from views.tabs.team import render_team


def show_dashboard():
    # --- Data Integration ---
    df_adm = get_admission_data()
    df_lits, df_perso, df_equip, df_stocks = get_logistique_data()
    df_pat, df_sej, df_diag = get_patient_sejour_data()
    model_lgbm = load_champion_model()
    
    # Global Time Series
    daily_ts = df_adm.groupby('date_entree').size().rename('admissions')
    
    # --- Header ---
    if hasattr(st, 'logo'):
        st.logo(LOGO_PATH, icon_image=LOGO_PATH)
        
    st.markdown(f"""
        <div style='display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 30px; padding-top: 10px;'>
            <h1 style='margin: 0; font-weight: 800; letter-spacing: -1px; background: linear-gradient(to right, #ffffff, {SECONDARY_BLUE}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>PITIE-SALPETRIERE <span style='font-weight: 300; font-size: 0.8em; color: #8899A6;'>VISION 2026</span></h1>
        </div>
    """, unsafe_allow_html=True)
    
    # --- Sidebar ---
    with st.sidebar:
        st.markdown("<h2 style='color:#f0f4f8;'>Vision 2026</h2>", unsafe_allow_html=True)
        st.divider()
        focus = st.selectbox("Focus Intelligence", ["Activité Globale", "Alertes Pics", "Optimisation Services"])
        st.divider()
        
        # Contenu Sidebar Personnalise
        if focus == "Alertes Pics":
            st.error("3 alertes détectées")
            st.info("Pic prévu : Lundi prochain (+15%)")
        elif focus == "Optimisation Services":
            st.success("Staff : Optimisé")
            st.warning("Lits : Tension en Réa")
        
        st.divider()
        if st.button("Quitter le Dashboard"):
            st.session_state.page = 'landing'
            st.rerun()
            
    # --- Tabs Layout ---
    tab_acc, tab_exp, tab_ml, tab_sim, tab_tea = st.tabs([
        "TABLEAU DE BORD", "EXPLORATION DATA", "PREVISIONS ML", "SIMULATEUR", "EQUIPE"
    ])

    
    with tab_acc:
        render_overview(df_adm, daily_ts)
        
    with tab_exp:
        render_eda(df_adm, daily_ts, df_lits, df_perso, df_equip, df_stocks, df_pat, df_sej, df_diag)
        
    with tab_ml:
        render_ml(daily_ts, model_lgbm)
        
    with tab_sim:
        render_simulator(daily_ts, model_lgbm)
        
    with tab_tea:
        render_team()
