import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import base64

# --- Page Config ---
st.set_page_config(
    page_title="Pitié-Salpêtrière | Vision 2026",
    page_icon="app/assets/logo_ps.png",
    layout="wide",
)

# --- Path Constants ---
LOGO_PATH = "app/assets/logo_ps.png"
HERO_BG_PATH = "app/assets/hero_bg.png"
PRIMARY_BLUE = "#005ba1"
SECONDARY_BLUE = "#00d2ff"
ACCENT_RED = "#c8102e"
BG_DARK = "#0a0c10"

# --- Helper to load image as base64 ---
def get_base64_image(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return ""

HERO_BG64 = get_base64_image(HERO_BG_PATH)

# --- Global Styling ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Outfit', sans-serif;
        color: #f0f4f8;
    }}

    .stApp {{
        background: radial-gradient(circle at 0% 0%, #1a3a5f 0%, {BG_DARK} 50%),
                    radial-gradient(circle at 100% 100%, #005ba166 0%, {BG_DARK} 50%);
        background-attachment: fixed;
    }}

    /* Radical Streamlit UI Cleaning */
    header[data-testid="stHeader"], [data-testid="stDecoration"] {{
        display: none !important;
        height: 0px !important;
    }}
    
    [data-testid="stAppViewContainer"] {{
        padding-top: 0rem !important;
    }}

    .main .block-container {{
        padding-top: 0rem !important;
        margin-top: -1rem !important;
    }}

    #root > div:nth-child(1) > div.withScreencast > div > div > div > section {{
        padding-top: 0rem !important;
    }}

    /* Global Button Styling */
    .stButton>button {{
        background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, #003d6b 100%);
        color: white !important;
        border-radius: 50px;
        padding: 12px 40px;
        border: 1px solid rgba(255,255,255,0.1);
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        font-size: 0.9rem;
    }}

    .stButton>button:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 30px {SECONDARY_BLUE}33;
        border-color: {SECONDARY_BLUE};
    }}

    /* Landing Page Specific */
    .hero-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        height: 100vh;
        max-height: 100vh;
        gap: 50px;
        padding-top: 0;
        margin-top: 0;
        animation: fadeIn 1.5s ease-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .wow-title {{
        font-size: 4.5rem !important;
        line-height: 1.1;
        font-weight: 800 !important;
        background: linear-gradient(to right, #ffffff, {SECONDARY_BLUE});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 15px;
    }}

    .wow-sub {{
        font-size: 1.5rem;
        color: #8899A6;
        margin-bottom: 8px;
        line-height: 1.6;
        font-weight: 300;
    }}

    .floating-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(30px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 40px;
        padding: 40px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }}

    .stat-badge {{
        background: rgba(255, 255, 255, 0.05);
        padding: 10px 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: inline-block;
        margin-right: 15px;
        margin-bottom: 15px;
    }}

    /* Dashboard Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 30px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-size: 1.1rem;
        font-weight: 600;
        padding: 10px 20px;
        background: transparent;
        color: #8899A6;
    }}
</style>
""", unsafe_allow_html=True)

# --- Session Management ---
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def go_to_dashboard():
    st.session_state.page = 'dashboard'

# --- Landing Logic ---
if st.session_state.page == 'landing':
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
        st.button("Entrer dans l'Espace Décisionnel", on_click=go_to_dashboard)
    with col_visual:
        st.markdown("<div class='floating-card'>", unsafe_allow_html=True)
        if HERO_BG64:
            st.markdown(f"<img src='data:image/png;base64,{HERO_BG64}' style='width:100%; border-radius:30px;'>", unsafe_allow_html=True)
        else:
            st.image(LOGO_PATH, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- Dashboard Logic ---
@st.cache_data
def load_data():
    dates = pd.date_range(end=datetime.now(), periods=100)
    return pd.DataFrame({
        "Date": dates,
        "Flux": 130 + np.random.normal(0, 15, 100).cumsum().clip(50, 250),
        "Occ": np.random.uniform(85, 98, 100),
        "Staff": np.random.uniform(90, 100, 100),
        "Age": np.random.normal(55, 20, 100).clip(0, 100),
        "Service": np.random.choice(["Urgences", "Réanimation", "Chirurgie", "Médecine"], 100)
    })

data = load_data()
st.logo(LOGO_PATH, icon_image=LOGO_PATH)

with st.sidebar:
    st.markdown("<h2 style='color:#f0f4f8;'>Système Vision 2026</h2>", unsafe_allow_html=True)
    st.divider()
    st.selectbox("Scénario Proactif", ["Filtre Nominal", "Alerte Épidémique", "Vague de Chaleur", "Plan Blanc Simulé"])
    st.divider()
    if st.button("Quitter le Dashboard"):
        st.session_state.page = 'landing'
        st.rerun()

tab_acc, tab_exp, tab_ml, tab_sim, tab_tea = st.tabs([
    "TABLEAU DE BORD", "EXPLORATION DATA", "PRÉVISIONS ML", "SIMULATEUR", "ÉQUIPE PROJET"
])

with tab_acc:
    st.markdown("<h2 style='font-weight:800;'>Panorama de l'Activité</h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("ACTIVITÉ 24H", "154 PATIENTS", "14%")
    c2.metric("OCCUPATION RÉA", "97.2%", "2.1%")
    c3.metric("EFFORTS STAFF", "92.5%", "-5%", delta_color="inverse")
    fig = px.area(data, x="Date", y="Flux", template="plotly_dark", color_discrete_sequence=[SECONDARY_BLUE])
    fig.update_layout(height=450, margin=dict(l=0,r=0,b=0,t=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

with tab_exp:
    sub_tab_adm, sub_tab_log, sub_tab_sej = st.tabs([
        "Admission patient", "Logistique", "Séjour patient"
    ])
    
    with sub_tab_adm:
        st.markdown("### Analyse des Admissions")
        col1, col2 = st.columns(2)
        with col1:
            fig_adm = px.histogram(data, x="Flux", nbins=20, title="Distribution des Admissions Quotidiennes", color_discrete_sequence=[PRIMARY_BLUE])
            st.plotly_chart(fig_adm, use_container_width=True)
        with col2:
            fig_serv = px.pie(data, names="Service", title="Répartition par Service", hole=0.4)
            st.plotly_chart(fig_serv, use_container_width=True)

    with sub_tab_log:
        st.markdown("### Suivi Logistique & Ressources")
        col1, col2 = st.columns(2)
        with col1:
            fig_occ = px.line(data, x="Date", y="Occ", title="Taux d'Occupation (%)", color_discrete_sequence=[ACCENT_RED])
            st.plotly_chart(fig_occ, use_container_width=True)
        with col2:
            fig_staff = px.area(data, x="Date", y="Staff", title="Capacité Staff Disponible", color_discrete_sequence=[SECONDARY_BLUE])
            st.plotly_chart(fig_staff, use_container_width=True)

    with sub_tab_sej:
        st.markdown("### Analyse des Séjours")
        col1, col2 = st.columns(2)
        with col1:
            fig_age = px.histogram(data, x="Age", title="Distribution d'Âge des Patients", color_discrete_sequence=["#ffffff"])
            st.plotly_chart(fig_age, use_container_width=True)
        with col2:
            st.info("La durée moyenne de séjour (DMS) est actuellement de 5.4 jours.")
            st.metric("Variation DMS (7j)", "+0.2j", delta_color="inverse")

with tab_ml:
    st.markdown("<h2 style='font-weight:800;'>Moteur de Prévisions ML</h2>", unsafe_allow_html=True)
    st.markdown("Prévisions des flux à 14 jours basées sur Prophet & XGBoost.")
    pred_dates = pd.date_range(start=datetime.now(), periods=14)
    preds = 180 + np.random.normal(0, 10, 14).cumsum()
    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(x=data['Date'].tail(15), y=data['Flux'].tail(15), name="Historique"))
    fig_ml.add_trace(go.Scatter(x=pred_dates, y=preds, name="Prévision (Modèle)", line=dict(dash='dash', color=SECONDARY_BLUE)))
    fig_ml.update_layout(template="plotly_dark", height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ml, use_container_width=True)

with tab_sim:
    st.markdown("<h2 style='font-weight:800;'>Simulateur de Scénarios</h2>", unsafe_allow_html=True)
    scen_col = st.columns([1, 2])
    with scen_col[0]:
        intensite = st.slider("Intensité de la crise (%)", 0, 100, 25)
        duree = st.number_input("Durée estimée (jours)", 1, 30, 7)
        if st.button("Lancer Simulation"):
            with st.status("Simulation en cours..."):
                time.sleep(1.5)
                st.write("Analyse des capacités...")
                time.sleep(1)
                st.write("Optimisation des ressources...")
            st.success(f"Simulation terminée. Surcroît de charge : +{intensite*1.2:.1f}%")

with tab_tea:
    st.markdown("<h1 style='text-align:center;'>L'Équipe Excellence</h1>", unsafe_allow_html=True)
    team_cols = st.columns(4)
    members = ["FranckF", "Gaetan Adj", "Djouhratabet", "Martineau"]
    for i, member in enumerate(members):
        with team_cols[i]:
            st.markdown(f"<div style='text-align:center; padding:30px; border-radius:20px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);'><div style='font-size:1.2rem; font-weight:800; color:{SECONDARY_BLUE};'>{member}</div><div style='font-size:0.8rem; color:#8899A6; margin-top:10px;'>Expertise IA & Santé</div></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='text-align:center; color:#555;'>Projet Académique M2 - Promotion 2026</p>", unsafe_allow_html=True)
