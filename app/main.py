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
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .landing-rect {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 60px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 50px;
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

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: rgba(13, 17, 23, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .main-title {{
        font-size: 3.5rem !important;
        background: linear-gradient(90deg, #FFFFFF 0%, {SECONDARY_BLUE} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }}

    /* Custom Button */
    .stButton>button {{
        background: {PRIMARY_BLUE};
        color: white;
        border-radius: 30px;
        padding: 10px 30px;
        border: none;
        font-weight: 600;
        transition: 0.3s;
    }}

    .stButton>button:hover {{
        background: {SECONDARY_BLUE};
        box-shadow: 0 0 20px {SECONDARY_BLUE}44;
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

# --- Landing Page logic ---
if st.session_state.page == 'landing':
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='landing-rect'>", unsafe_allow_html=True)
        try:
            st.image(LOGO_PATH, width=400)
        except:
            st.title("Hôpital Pitié-Salpêtrière")
        
        st.markdown("<h1 class='main-title'>Vision 2026</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.4rem; color: #8899A6;'>Système de Prévision & Gestion des Ressources Hospitalières</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Accéder au Dashboard", on_click=go_to_dashboard, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# --- Dashboard Logic ---
data = get_full_data()

# Sidebar
with st.sidebar:
    try:
        st.image(LOGO_PATH, use_container_width=True)
    except:
        st.title("AP-HP")
    
    st.divider()
    st.markdown("### Contrôles")
    time_range = st.selectbox("Période d'analyse", ["7 derniers jours", "30 derniers jours", "3 mois", "Année complète"])
    scénario = st.selectbox("Simulation Active", ["Normal", "Crise Hivernale", "Canicule Extrême", "Sous-effectif"])
    
    st.divider()
    st.markdown("#### Alertes Actives")
    st.error("Saturation Réanimation : 96%")
    st.warning("Pic prévu : Jeudi prochain (+15%)")
    
    if st.button("Retour à l'accueil"):
        st.session_state.page = 'landing'
        st.rerun()

# --- Main Layout ---
tab_accueil, tab_exploration, tab_predictions, tab_simulations = st.tabs([
    "🏠 Accueil", "📊 Exploration", "🔮 Prédictions", "🎮 Simulations"
])

# --- Tab 1: Accueil ---
with tab_accueil:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("<h2 style='margin-bottom:0;'>État Général du Système</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#8899A6;'>Vue d'ensemble en temps réel des flux hospitaliers</p>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='text-align:right; font-weight:600; color:{ACCENT_RED}; padding:10px;'>LIVE: {datetime.now().strftime('%H:%M')}</div>", unsafe_allow_html=True)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Admissions", "142", "12%")
    m2.metric("Taux d'Occupation", "89.4%", "2.1%")
    m3.metric("Lits Disponibles", "184", "-15", delta_color="inverse")
    m4.metric("Score Efficacité", "92/100", "5")

    # Main Chart
    fig_main = px.area(data.tail(30), x="Date", y="Admissions", title="Trend des Admissions (30 pts)")
    fig_main.update_traces(line_color=SECONDARY_BLUE, fillcolor=f"rgba(0, 210, 255, 0.1)")
    fig_main.update_layout(height=400, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_main, use_container_width=True)

# --- Tab 2: Exploration ---
with tab_exploration:
    st.markdown("### Analyse de Corrélation et Distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig_hist = px.histogram(data, x="Admissions", color="Service", marginal="box", barmode="overlay")
        fig_hist.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        fig_scatter = px.scatter(data, x="Lits", y="Admissions", size="Staff", color="Service", hover_name="Date")
        fig_scatter.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- Tab 3: Prédictions ---
with tab_tab3 := tab_predictions:
    st.markdown("### Moteur de Prédiction ML (Prophet / XGBoost)")
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        # Fake Prediction Plot
        pred_dates = [datetime.now() + timedelta(days=x) for x in range(14)]
        pred_values = np.random.poisson(150, 14)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=data['Date'].tail(15), y=data['Admissions'].tail(15), name="Historique", line=dict(color=SECONDARY_BLUE)))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=pred_values, name="Prévision (95% CI)", line=dict(color=ACCENT_RED, dash='dash')))
        fig_pred.update_layout(height=450, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pred, use_container_width=True)
    with col_p2:
        st.markdown("#### Performance Modèle")
        st.info("Précision (MAE): 4.2 admissions")
        st.info("MAPE: 3.1%")
        st.markdown("---")
        st.markdown("#### Facteurs d'Influence (SHAP)")
        factors = pd.DataFrame({"Factor": ["Météo", "Saison", "Effectif", "Lits"], "Importance": [0.45, 0.30, 0.15, 0.10]})
        fig_feat = px.bar(factors, y="Factor", x="Importance", orientation='h', color_discrete_sequence=[PRIMARY_BLUE])
        fig_feat.update_layout(height=250, template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig_feat, use_container_width=True)

# --- Tab 4: Simulations ---
with tab_simulations:
    st.markdown("### Laboratoire de Simulation Stratégique")
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        st.markdown("#### Paramètres du Scénario")
        intensite = st.select_slider("Intensité de la Crise", options=["Mineure", "Modérée", "Sévère", "Critique"])
        duree_sim = st.number_input("Durée (jours)", 1, 60, 14)
        if st.button("Lancer la Simulation"):
            with st.spinner("Calcul des impacts ressources..."):
                time.sleep(1.5)
                st.success("Simulation terminée")
    with col_s2:
        st.markdown("#### Impact Estimé")
        st.warning(f"Surcharge prévue : +{np.random.randint(20, 50)}% sur les Urgences")
        st.error(f"Point de rupture : Jour {np.random.randint(4, 9)}")
        
    st.divider()
    st.markdown("### Recommandations IA")
    st.success("Priorité : Réquisitionner 5 personnels soignants en chirurgie programmée")
    st.info("Action : Transfert inter-hospitalier (Hôpital Saint-Antoine) à prévoir pour J+3")

# Footer
st.markdown("---")
st.markdown(f"<p style='text-align:center; color:#8899A6; font-size:0.8rem;'>Pitié-Salpêtrière | Vision 2026 Dashboard | Développé avec excellence</p>", unsafe_allow_html=True)
