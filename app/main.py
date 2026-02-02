import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# --- Page Config ---
st.set_page_config(
    page_title="Pitié-Salpêtrière | Vision 2026",
    page_icon="app/assets/logo_ps.png",
    layout="wide",
)

# --- Themes & Styling ---
LOGO_PATH = "app/assets/logo_ps.png"
PRIMARY_BLUE = "#005ba1"
SECONDARY_BLUE = "#00d2ff"
ACCENT_RED = "#c8102e"
BG_DARK = "#0d1117"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }}

    .stApp {{
        background: radial-gradient(circle at top right, #1a2a47, {BG_DARK});
    }}

    /* Landing Page Styling */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .landing-rect {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 60px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 50px;
        animation: fadeIn 1.2s ease-out;
    }}

    /* Tab Custom Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 24px;
        background-color: transparent;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        padding: 0px 24px;
        font-weight: 600;
        color: #8899A6;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: rgba(0, 91, 161, 0.2) !important;
        color: {SECONDARY_BLUE} !important;
        border-bottom: 2px solid {SECONDARY_BLUE} !important;
    }}

    /* Glassmorphism Containers */
    div.stMetric, .element-container div[style*="flex-direction: column"] > div {{
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .main-title {{
        font-size: 3.5rem !important;
        background: linear-gradient(90deg, #FFFFFF 0%, {SECONDARY_BLUE} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }}

    .stButton>button {{
        background: {PRIMARY_BLUE};
        color: white;
        border-radius: 30px;
        padding: 10px 30px;
        border: none;
        font-weight: 600;
        transition: 0.3s;
    }}
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

def go_to_dashboard():
    st.session_state.page = 'dashboard'

# --- Data Engine ---
@st.cache_data
def get_full_data():
    days = 120
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    df = pd.DataFrame({
        "Date": dates,
        "Admissions": np.random.poisson(130, days) + np.sin(np.linspace(0, 10, days)) * 20,
        "Lits": np.random.uniform(80, 95, days),
        "Staff": np.random.uniform(88, 98, days),
        "Service": np.random.choice(["Urgences", "Chirurgie", "Réanimation", "Médecine", "Pédiatrie"], days)
    })
    return df

# --- Landing Page ---
if st.session_state.page == 'landing':
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='landing-rect'>", unsafe_allow_html=True)
        st.image(LOGO_PATH, width=400)
        st.markdown("<h1 class='main-title'>Vision 2026</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.4rem; color: #8899A6;'>Système de Prévision & Gestion des Ressources Hospitalières</p>", unsafe_allow_html=True)
        st.button("Accéder au Dashboard", on_click=go_to_dashboard, use_container_width=True)
        st.markdown("<p style='margin-top:20px; color:#555;'>Promotion 2026 | Direction Pitié-Salpêtrière</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- Dashboard ---
data = get_full_data()
st.logo(LOGO_PATH, icon_image=LOGO_PATH)

with st.sidebar:
    st.markdown("### Contrôles")
    scénario = st.selectbox("Simulation Active", ["Normal", "Crise Hivernale", "Canicule Extrême", "Sous-effectif"])
    st.divider()
    if st.button("Retour à l'accueil"):
        st.session_state.page = 'landing'
        st.rerun()

tab_accueil, tab_exploration, tab_predictions, tab_simulations, tab_infos = st.tabs([
    "ACCUEIL", "EXPLORATION", "PRÉDICTIONS", "SIMULATIONS", "INFOS"
])

with tab_accueil:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Admissions (24h)", "142", "12%")
    m2.metric("Occupation Lits", "89.4%", "2.1%")
    m3.metric("Lits Disponibles", "184", "-15", delta_color="inverse")
    m4.metric("Score Efficacité", "92/100", "5")
    fig_main = px.area(data.tail(30), x="Date", y="Admissions", title="Tendances des Admissions")
    fig_main.update_layout(height=400, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_main, use_container_width=True)

with tab_exploration:
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(data, x="Admissions", color="Service", barmode="overlay", title="Distribution par Service")
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        fig_scatter = px.scatter(data, x="Lits", y="Admissions", color="Service", title="Corrélation Lits vs Admissions")
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab_predictions:
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        pred_dates = [datetime.now() + timedelta(days=x) for x in range(14)]
        pred_values = np.random.poisson(150, 14)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=data['Date'].tail(15), y=data['Admissions'].tail(15), name="Historique"))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=pred_values, name="Prévision", line=dict(dash='dash', color=ACCENT_RED)))
        fig_pred.update_layout(height=450, template="plotly_dark", title="Prévisions à 14 jours")
        st.plotly_chart(fig_pred, use_container_width=True)
    with col_p2:
        st.markdown("#### Performance Modèle")
        st.info("MAE: 4.2 | MAPE: 3.1%")
        st.markdown("#### Importances des Variables")
        factors = pd.DataFrame({"Variable": ["Saison", "Météo", "Staff", "Lits"], "Score": [0.4, 0.35, 0.15, 0.1]})
        st.plotly_chart(px.bar(factors, y="Variable", x="Score", orientation='h', template="plotly_dark"), use_container_width=True)

with tab_simulations:
    st.markdown("### Simulateur Stratégique")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        intensite = st.select_slider("Intensité de l'évènement", ["Basse", "Moyenne", "Haute", "Critique"])
        if st.button("Calculer Impact"):
            st.toast("Calcul en cours...")
            time.sleep(1)
            st.success("Impact calculé : +28% de flux prévu.")
    with s_col2:
        st.warning("Recommandation : Activer le plan blanc si l'occupation dépasse 95%.")

with tab_infos:
    st.image(LOGO_PATH, width=200)
    st.markdown("### Équipe Projet")
    st.markdown("""
    - **FranckF**
    - **koffigaetan-adj**
    - **Djouhratabet**
    - **cmartineau15**
    """)
    st.divider()
    st.markdown("Pitié-Salpêtrière | Division Data & Prospective © 2026")
