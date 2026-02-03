import streamlit as st
from config import SECONDARY_BLUE, PRIMARY_BLUE, ACCENT_RED


def render_team():
    # Header premium
    st.markdown(f"""
        <div style='text-align: center; margin-bottom: 40px;'>
            <h1 style='font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, {SECONDARY_BLUE} 0%, {PRIMARY_BLUE} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                EQUIPE & PROJET
            </h1>
            <p style='font-size: 1.2rem; color: #8899A6; margin-top: -10px;'>
                Vision 2026 - Hopital Pitie-Salpetriere
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Section Equipe
    st.markdown("<div style='background: rgba(255,255,255,0.03); border-radius: 20px; padding: 30px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 30px;'>", unsafe_allow_html=True)
    
    st.markdown("## 👥 Equipe Data Science & Innovation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(0,91,161,0.2) 0%, rgba(0,210,255,0.1) 100%); 
                        border-radius: 15px; padding: 25px; border: 1px solid rgba(0,210,255,0.3); margin-bottom: 20px;'>
                <h3 style='margin: 0 0 15px 0; color: {SECONDARY_BLUE};'>Lead Data Scientist</h3>
                <p style='margin: 5px 0; font-size: 1.1rem;'><strong>Franck F.</strong></p>
                <p style='margin: 5px 0; color: #8899A6;'>Modelisation Predictive & ML</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 25px; border: 1px solid rgba(255,255,255,0.1);'>
                <h3 style='margin: 0 0 15px 0; color: {SECONDARY_BLUE};'>Contact</h3>
                <p style='margin: 5px 0;'><a href='mailto:pitié-salpetriere@vision.fr'><strong>pitié-salpetriere@vision.fr</strong></a></p>
                <p style='margin: 5px 0;'><a href='https://hopital-pitie-salpetrieregit-jsfpemvrjtde9tma3f7yq6.streamlit.app/'><strong>AP-HP Pitie-Salpetriere</strong></a></p>
                <p style='margin: 5px 0;'><strong>Paris, France</strong></p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 25px; border: 1px solid rgba(255,255,255,0.1); height: 100%;'>
                <h3 style='margin: 0 0 15px 0; color: {SECONDARY_BLUE};'>Mission du Projet</h3>
                <p style='margin: 10px 0; line-height: 1.8; color: #f0f4f8;'>
                    Developper une plateforme decisionnelle intelligente pour anticiper les flux hospitaliers, 
                    optimiser l'allocation des ressources et renforcer la resilience operationnelle de l'hopital 
                    face aux crises sanitaires.
                </p>
                <p style='margin: 15px 0 0 0; padding: 15px; background: rgba(0,210,255,0.1); border-radius: 10px; border-left: 4px solid {SECONDARY_BLUE};'>
                    <strong>Objectif :</strong> Reduire les tensions capacitaires et ameliorer la qualite des soins 
                    grace a l'intelligence artificielle.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Section Projet
    st.markdown("<div style='background: rgba(255,255,255,0.03); border-radius: 20px; padding: 30px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 30px;'>", unsafe_allow_html=True)
    
    st.markdown("## Caracteristiques du Projet")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.markdown(f"""
            <div style='background: rgba(0,91,161,0.2); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(0,210,255,0.3);'>
                <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Donnees Analysees</p>
                <h2 style='margin: 5px 0; color: {SECONDARY_BLUE};'>100K+</h2>
                <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>admissions/an</p>
            </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""
            <div style='background: rgba(0,91,161,0.2); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(0,210,255,0.3);'>
                <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Precision ML</p>
                <h2 style='margin: 5px 0; color: {SECONDARY_BLUE};'>95%+</h2>
                <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>R² Score</p>
            </div>
        """, unsafe_allow_html=True)
    
    with kpi3:
        st.markdown(f"""
            <div style='background: rgba(0,91,161,0.2); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(0,210,255,0.3);'>
                <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Lits Geres</p>
                <h2 style='margin: 5px 0; color: {SECONDARY_BLUE};'>1.8K</h2>
                <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>capacite totale</p>
            </div>
        """, unsafe_allow_html=True)
    
    with kpi4:
        st.markdown(f"""
            <div style='background: rgba(0,91,161,0.2); border-radius: 15px; padding: 20px; text-align: center; border: 1px solid rgba(0,210,255,0.3);'>
                <p style='margin: 0; font-size: 0.9rem; color: #8899A6;'>Periode Couverte</p>
                <h2 style='margin: 5px 0; color: {SECONDARY_BLUE};'>2024-25</h2>
                <p style='margin: 0; font-size: 0.8rem; color: #8899A6;'>donnees reelles</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Section Stack Technique
    st.markdown("<div style='background: rgba(255,255,255,0.03); border-radius: 20px; padding: 30px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 30px;'>", unsafe_allow_html=True)
    
    st.markdown("## Stack Technique")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; border: 1px solid rgba(255,255,255,0.1);'>
                <h4 style='margin: 0 0 15px 0; color: {SECONDARY_BLUE};'>Frontend & Viz</h4>
                <ul style='list-style: none; padding: 0; margin: 0;'>
                    <li style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <code style='background: rgba(0,210,255,0.1); padding: 4px 8px; border-radius: 5px;'>Streamlit</code>
                        <span style='color: #8899A6; font-size: 0.85rem;'> - Interface</span>
                    </li>
                    <li style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <code style='background: rgba(0,210,255,0.1); padding: 4px 8px; border-radius: 5px;'>Plotly</code>
                        <span style='color: #8899A6; font-size: 0.85rem;'> - Graphiques</span>
                    </li>
                    <li style='padding: 8px 0;'>
                        <code style='background: rgba(0,210,255,0.1); padding: 4px 8px; border-radius: 5px;'>CSS Custom</code>
                        <span style='color: #8899A6; font-size: 0.85rem;'> - Design</span>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with tech_col2:
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; border: 1px solid rgba(255,255,255,0.1);'>
                <h4 style='margin: 0 0 15px 0; color: {SECONDARY_BLUE};'>Data & ML</h4>
                <ul style='list-style: none; padding: 0; margin: 0;'>
                    <li style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <code style='background: rgba(0,210,255,0.1); padding: 4px 8px; border-radius: 5px;'>Pandas</code>
                        <span style='color: #8899A6; font-size: 0.85rem;'> - Manipulation</span>
                    </li>
                    <li style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <code style='background: rgba(0,210,255,0.1); padding: 4px 8px; border-radius: 5px;'>NumPy</code>
                        <span style='color: #8899A6; font-size: 0.85rem;'> - Calculs</span>
                    </li>
                    <li style='padding: 8px 0;'>
                        <code style='background: rgba(0,210,255,0.1); padding: 4px 8px; border-radius: 5px;'>SciPy</code>
                        <span style='color: #8899A6; font-size: 0.85rem;'> - Stats</span>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with tech_col3:
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); border-radius: 15px; padding: 20px; border: 1px solid rgba(255,255,255,0.1);'>
                <h4 style='margin: 0 0 15px 0; color: {SECONDARY_BLUE};'>Modelisation</h4>
                <ul style='list-style: none; padding: 0; margin: 0;'>
                    <li style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <code style='background: rgba(0,210,255,0.1); padding: 4px 8px; border-radius: 5px;'>LightGBM</code>
                        <span style='color: #8899A6; font-size: 0.85rem;'> - Predictions</span>
                    </li>
                    <li style='padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                        <code style='background: rgba(0,210,255,0.1); padding: 4px 8px; border-radius: 5px;'>Scikit-learn</code>
                        <span style='color: #8899A6; font-size: 0.85rem;'> - ML Pipeline</span>
                    </li>
                    <li style='padding: 8px 0;'>
                        <code style='background: rgba(0,210,255,0.1); padding: 4px 8px; border-radius: 5px;'>Joblib</code>
                        <span style='color: #8899A6; font-size: 0.85rem;'> - Serialisation</span>
                    </li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Section Fonctionnalites
    st.markdown("<div style='background: rgba(255,255,255,0.03); border-radius: 20px; padding: 30px; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 30px;'>", unsafe_allow_html=True)
    
    st.markdown("## Fonctionnalites Principales")
    
    feat_col1, feat_col2 = st.columns(2)
    
    with feat_col1:
        st.markdown(f"""
            <div style='margin-bottom: 15px; padding: 20px; background: rgba(0,210,255,0.05); border-radius: 12px; border-left: 4px solid {SECONDARY_BLUE};'>
                <h4 style='margin: 0 0 10px 0; color: {SECONDARY_BLUE};'>Tableau de Bord Temps Reel</h4>
                <p style='margin: 0; color: #8899A6; line-height: 1.6;'>
                    Visualisation des flux d'admissions quotidiens avec metriques cles et tendances.
                </p>
            </div>
            
            <div style='margin-bottom: 15px; padding: 20px; background: rgba(0,210,255,0.05); border-radius: 12px; border-left: 4px solid {SECONDARY_BLUE};'>
                <h4 style='margin: 0 0 10px 0; color: {SECONDARY_BLUE};'>Data Exploration Avancee</h4>
                <p style='margin: 0; color: #8899A6; line-height: 1.6;'>
                    Analyse multi-dimensionnelle des admissions, logistique et parcours patients.
                </p>
            </div>
            
            <div style='padding: 20px; background: rgba(0,210,255,0.05); border-radius: 12px; border-left: 4px solid {SECONDARY_BLUE};'>
                <h4 style='margin: 0 0 10px 0; color: {SECONDARY_BLUE};'>Previsions ML</h4>
                <p style='margin: 0; color: #8899A6; line-height: 1.6;'>
                    Modele LightGBM optimise pour predictions a 14 jours avec MAE < 1.0.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown(f"""
            <div style='margin-bottom: 15px; padding: 20px; background: rgba(0,210,255,0.05); border-radius: 12px; border-left: 4px solid {SECONDARY_BLUE};'>
                <h4 style='margin: 0 0 10px 0; color: {SECONDARY_BLUE};'>Simulateur de Crise</h4>
                <p style='margin: 0; color: #8899A6; line-height: 1.6;'>
                    Scenarios pre-configures pour tester la resilience face aux crises sanitaires.
                </p>
            </div>
            
            <div style='margin-bottom: 15px; padding: 20px; background: rgba(0,210,255,0.05); border-radius: 12px; border-left: 4px solid {SECONDARY_BLUE};'>
                <h4 style='margin: 0 0 10px 0; color: {SECONDARY_BLUE};'>Detection d'Anomalies</h4>
                <p style='margin: 0; color: #8899A6; line-height: 1.6;'>
                    Identification automatique des pics de charge via methode IQR.
                </p>
            </div>
            
            <div style='padding: 20px; background: rgba(0,210,255,0.05); border-radius: 12px; border-left: 4px solid {SECONDARY_BLUE};'>
                <h4 style='margin: 0 0 10px 0; color: {SECONDARY_BLUE};'>Architecture Modulaire</h4>
                <p style='margin: 0; color: #8899A6; line-height: 1.6;'>
                    Code refactorise en modules pour faciliter la maintenance et l'evolution.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown(f"""
        <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, rgba(0,91,161,0.1) 0%, rgba(0,210,255,0.05) 100%); 
                    border-radius: 20px; border: 1px solid rgba(0,210,255,0.2);'>
            <p style='margin: 0; font-size: 1.1rem; color: {SECONDARY_BLUE}; font-weight: 600;'>
                Projet Vision 2026 - Hopital Pitie-Salpetriere
            </p>
            <p style='margin: 10px 0 0 0; color: #8899A6;'>
                Developpe par  Franck - Charlotte - Gaetan - Djouhra - Farah
            </p>
        </div>
    """, unsafe_allow_html=True)
