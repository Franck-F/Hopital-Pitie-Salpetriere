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
    # Data Integration 
    df_adm = get_admission_data()
    df_lits, df_perso, df_equip, df_stocks = get_logistique_data()
    df_pat, df_sej, df_diag = get_patient_sejour_data()
    model_lgbm = load_champion_model()
    
    # Global Time Series
    daily_ts = df_adm.groupby('date_entree').size().rename('admissions')
    
    # Header 
    if hasattr(st, 'logo'):
        st.logo(LOGO_PATH, icon_image=LOGO_PATH)
        
    st.markdown(f"""
        <div style='display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 30px; padding-top: 10px;'>
            <h1 style='margin: 0; font-weight: 800; letter-spacing: -1px; background: linear-gradient(to right, #ffffff, {SECONDARY_BLUE}); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>PITIE-SALPETRIERE <span style='font-weight: 300; font-size: 0.8em; color: #8899A6;'>VISION 2026</span></h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar 
    with st.sidebar:
        st.markdown("<h2 style='color:#f0f4f8;'>Vision 2026</h2>", unsafe_allow_html=True)
        st.divider()
        focus = st.selectbox("Focus Intelligence", ["Activité Globale", "Alertes Pics", "Optimisation Services"])
        st.divider()
        
        # Contenu Sidebar Personnalisé
        if focus == "Alertes Pics":
            st.markdown("#### Alertes Détectées")
            
            # Alerte 1 - Pic d'admissions
            st.error("**Pic d'Admissions Prévu**")
            st.markdown("""
                - **Date** : Lundi 10 Février
                - **Intensité** : +15% vs moyenne
                - **Services impactés** : Urgences, Médecine
                - **Action** : Renforcer effectifs
            """)
            
            # Alerte 2 - Tension lits
            st.warning("**Tension Lits Réanimation**")
            st.markdown("""
                - **Taux actuel** : 92% occupation
                - **Seuil critique** : 95%
                - **Marge** : 3 lits disponibles
                - **Action** : Préparer plan de débordement
            """)
            
            # Alerte 3 - Stock médicaments
            st.info("**Alerte Stock Médicaments**")
            st.markdown("""
                - **Références en rupture** : 2
                - **Références critiques** : 5 (< 7 jours)
                - **Action** : Commande urgente requise
            """)
            
        elif focus == "Optimisation Services":
            st.markdown("#### Recommandations d'Optimisation")
            
            # Optimisation 1 - Staff
            st.success("**Effectifs Personnel**")
            st.markdown("""
                - **Statut** :  Optimisé
                - **Taux présence** : 87%
                - **Répartition** : Équilibrée
                - **Suggestion** : Maintenir niveau actuel
            """)
            
            # Optimisation - Lits
            st.warning("**Capacité Lits**")
            st.markdown("""
                - **Statut** :  Tension en Réanimation
                - **Occupation Réa** : 92%
                - **Occupation Médecine** : 78%
                - **Suggestion** : Transférer 2-3 patients stables vers Médecine
            """)
            
            # Optimisation - Flux
            st.info("**Flux Patients**")
            st.markdown("""
                - **Durée moyenne séjour** : 5.2 jours
                - **Objectif** : 4.8 jours
                - **Potentiel gain** : 8% de capacité
                - **Suggestion** : Accélérer sorties matinales
            """)
        
        st.divider()
        
        # Section Mentions Legales
        st.markdown("<h4 style='color:#8899A6; font-size: 0.9rem; margin-top: 20px;'>Informations Legales</h4>", unsafe_allow_html=True)
        
        with st.expander("CGU - Conditions Générales d'Utilisation"):
            st.markdown("""
                **Application de Gestion Hospitalière**
                
                Cette application est destinée exclusivement à un usage interne 
                par le personnel autorisé de l'Hôpital Pitié-Salpêtrière.
                
                - L'accès est réservé aux professionnels de santé habilités
                - Les données affichées sont confidentielles
                - Toute utilisation non autorisée est interdite
                - Les prédictions ML sont des outils d'aide à la décision
                
                Version 2026 - AP-HP
            """)
        
        with st.expander("RGPD - Protection des Données"):
            st.markdown("""
                **Conformité RGPD**
                
                Les données personnelles sont traitées conformément au 
                Règlement Général sur la Protection des Données (RGPD).
                
                - Finalité : Gestion et optimisation des flux hospitaliers
                - Base légale : Mission d'intérêt public (santé publique)
                - Durée de conservation : Selon réglementation en vigueur
                - Droits : Accès, rectification, limitation, opposition
                
                Contact DPO : dpo@pitié.fr
            """)
        
        with st.expander("Mentions Legales"):
            st.markdown("""
                **Éditeur**
                
                Assistance Publique - Hôpitaux de Paris (AP-HP)
                Hôpital Pitié-Salpêtrière
                47-83 Boulevard de l'Hôpital, 75013 Paris
                
                **Hébergement**
                
                Données hébergées sur infrastructure sécurisée AP-HP
                Conforme aux normes HDS (Hébergeur de Données de Santé)
                
                **Propriété Intellectuelle**
                
                Tous droits réservés - AP-HP 2026
            """)
        
        st.divider()
        if st.button("Quitter le Dashboard"):
            st.session_state.page = 'landing'
            st.rerun()
            
    # Onglets 
    tab_acc, tab_exp, tab_ml, tab_sim, tab_tea = st.tabs([
        "TABLEAU DE BORD", "EXPLORATION DATA", "PREVISIONS ML", "SIMULATEUR", "EQUIPE"
    ])

    
    with tab_acc:
        render_overview(df_adm, daily_ts, df_lits, df_perso, df_equip, df_stocks)
        
    with tab_exp:
        render_eda(df_adm, daily_ts, df_lits, df_perso, df_equip, df_stocks, df_pat, df_sej, df_diag)
        
    with tab_ml:
        render_ml(daily_ts, model_lgbm)
        
    with tab_sim:
        render_simulator(daily_ts, model_lgbm)
        
    with tab_tea:
        render_team()
