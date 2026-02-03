import streamlit as st
from config import LOGO_PATH, HERO_BG_PATH, SECONDARY_BLUE, ACCENT_RED
from utils import get_base64_image


def show_landing_page(go_to_dashboard_callback):
    HERO_BG64 = get_base64_image(HERO_BG_PATH)
    
    st.markdown("<div class='hero-container'>", unsafe_allow_html=True)
    col_text, col_visual = st.columns([1.2, 1])
    with col_text:
        st.markdown(f"<div style='border-left: 5px solid {ACCENT_RED}; padding-left: 25px; margin-bottom: 30px;'><img src='data:image/png;base64,{get_base64_image(LOGO_PATH)}' width='300'></div>", unsafe_allow_html=True)
        st.markdown("<h1 class='wow-title'>L'excellence au service de la donnée prédictive.</h1>", unsafe_allow_html=True)
        st.markdown("<p class='wow-sub'>Anticiper les besoins, optimiser les ressources, sauver des vies. Bienvenue dans l'interface décisionnelle de l'Hôpital Pitié-Salpêtrière.</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='margin-bottom: 40px;'>
            <div class='stat-badge'><span style='color:{SECONDARY_BLUE}; font-weight:800;'>1.8K</span> lits gérés</div>
            <div class='stat-badge'><span style='color:{SECONDARY_BLUE}; font-weight:800;'>100K+</span> urgences/an</div>
            <div class='stat-badge'><span style='color:{SECONDARY_BLUE}; font-weight:800;'>95%</span> précision ML</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Callback wrapper to match st.button on_click signature if needed, 
        # or simplified just by checking return
        if st.button("Entrer dans l'Espace Décisionnel"):
            go_to_dashboard_callback()
            
    with col_visual:
        if HERO_BG64:
            st.markdown(f"<img src='data:image/png;base64,{HERO_BG64}' style='width:100%; border-radius:30px;'>", unsafe_allow_html=True)
        else:
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
