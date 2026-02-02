import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(
    page_title="Pitié-Salpêtrière | Resource Management",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Glassmorphism & Premium UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1a2a47, #0d1117);
    }

    /* Glassmorphism Containers */
    div.stMetric, div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Headers */
    h1, h2, h3 {
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    .main-title {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 0px;
    }

    /* KPI Metric Adjustments */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #00d2ff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8899A6 !important;
    }

    /* Hide redundant elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Data Simulation ---
@st.cache_data
def generate_mock_data(days=30, scenario="Normal"):
    dates = [datetime.now() - timedelta(days=x) for x in range(days)]
    dates.reverse()
    
    base_admissions = 120
    noise = np.random.normal(0, 15, days)
    
    if scenario == "Epidémie (Grippe/Covid)":
        trend = np.linspace(0, 80, days)
    elif scenario == "Canicule":
        trend = np.linspace(0, 40, days)
    else:
        trend = np.zeros(days)
        
    admissions = base_admissions + trend + noise
    
    df = pd.DataFrame({
        "Date": dates,
        "Admissions": admissions.astype(int),
        "Occupation_Lits": np.random.uniform(75, 95, days),
        "Staff_Available": np.random.uniform(85, 100, days)
    })
    return df

# --- Sidebar ---
st.sidebar.markdown("<h2 style='color: #00d2ff;'>Configuration</h2>", unsafe_allow_html=True)
selected_scenario = st.sidebar.selectbox(
    "Scénario de Simulation",
    ["Normal", "Epidémie (Grippe/Covid)", "Canicule", "Grève du Personnel"]
)

st.sidebar.divider()
st.sidebar.markdown("### Ressources Critiques")
st.sidebar.slider("Capacité active des lits", 500, 2000, 1800)
st.sidebar.slider("Effectifs cibles", 50, 500, 250)

st.sidebar.divider()
st.sidebar.info("Dashboard Decisionnel v1.0.0 Alpha - Direction Pitié-Salpêtrière")

# --- Header Section ---
st.markdown("<h1 class='main-title'>Pitié-Salpêtrière</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2rem; color: #8899A6; margin-top: -10px;'>Système Global de Prévision et de Gestion des Flux</p>", unsafe_allow_html=True)

# --- Data Processing ---
data = generate_mock_data(scenario=selected_scenario)
latest = data.iloc[-1]
prev = data.iloc[-2]

# --- KPI Row ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Admissions (24h)", f"{latest['Admissions']}", delta=f"{int(latest['Admissions'] - prev['Admissions'])}")
with col2:
    st.metric("Occupation Lits", f"{latest['Occupation_Lits']:.1f}%", delta=f"{latest['Occupation_Lits'] - prev['Occupation_Lits']:.1f}%")
with col3:
    st.metric("Personnel Actif", f"{latest['Staff_Available']:.1f}%", delta=f"{latest['Staff_Available'] - prev['Staff_Available']:.1f}%", delta_color="inverse")
with col4:
    alert_level = "CRITIQUE" if latest['Occupation_Lits'] > 90 else "NOMINAL"
    st.metric("État du Système", alert_level)

st.divider()

# --- Main Charts ---
c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("### Évolution des Admissions et Prévisions")
    fig_line = px.line(data, x="Date", y="Admissions", template="plotly_dark")
    fig_line.update_traces(line_color='#00d2ff', line_width=3, fill='tozeroy', fillcolor='rgba(0, 210, 255, 0.1)')
    fig_line.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        height=400,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig_line, use_container_width=True)

with c2:
    st.markdown("### Répartition par Service")
    services = pd.DataFrame({
        "Service": ["Urgences", "Réanimation", "Chirurgie", "Médecine Interne", "Pédiatrie"],
        "Flux": [45, 15, 20, 12, 8]
    })
    fig_donut = px.pie(services, values='Flux', names='Service', hole=0.7, 
                       color_discrete_sequence=['#00d2ff', '#3a7bd5', '#1a2a47', '#00bfa5', '#607d8b'])
    fig_donut.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    # Add center text
    fig_donut.add_annotation(text="Total Flux", showarrow=False, font_size=14, font_color="#8899A6")
    st.plotly_chart(fig_donut, use_container_width=True)

# --- Lower Section ---
st.markdown("### Analyse de Capacité Sectorielle")
c3, c4 = st.columns(2)

with c3:
    # Heatmap of occupancy by hour/day
    st.markdown("#### Intensité d'Occupation (Dernière Semaine)")
    days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
    hours = [f'{h}h' for h in range(0, 24, 4)]
    z_data = np.random.uniform(60, 100, (len(days), len(hours)))
    
    fig_hm = go.Figure(data=go.Heatmap(
        z=z_data, x=hours, y=days,
        colorscale=[[0, '#1a2a47'], [0.5, '#3a7bd5'], [1, '#00d2ff']],
        showscale=False
    ))
    fig_hm.update_layout(
        height=300, 
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="#8899A6"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

with c4:
    st.markdown("#### Recommandations de Déploiement")
    st.markdown(f"""
    <div style='background: rgba(0, 210, 255, 0.05); border-left: 5px solid #00d2ff; padding: 20px; border-radius: 5px;'>
        <p style='color: #00d2ff; font-weight: 600; margin-bottom: 10px;'>Protocoles Suggérés (Scénario : {selected_scenario})</p>
        <ul style='color: #E0E0E0; font-size: 0.95rem;'>
            <li>Activer la réserve sanitaire (Seuil {'>'} 85% atteint)</li>
            <li>Prioriser les sorties en Médecine Interne pour libérer 15 lits</li>
            <li>Renforcer les gardes de nuit aux Urgences (+2 IDE)</li>
            <li>Déporter les chirurgies non-urgentes de 24h</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.divider()
st.markdown("<p style='text-align: center; color: #8899A6; font-size: 0.8rem;'>© 2026 Pitié-Salpêtrière | Division Data & Prospective</p>", unsafe_allow_html=True)
