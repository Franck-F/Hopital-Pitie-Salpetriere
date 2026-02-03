import streamlit as st

# --- Constantes de Chemins ---
LOGO_PATH = "app/assets/logo_ps.png"
HERO_BG_PATH = "app/assets/hero_bg.png"
DATA_ADMISSION_PATH = "data/raw/admissions_hopital_pitie_2024_2025.csv"

# --- Constantes de Couleurs ---
PRIMARY_BLUE = "#005ba1"
SECONDARY_BLUE = "#00d2ff"
ACCENT_RED = "#c8102e"
BG_DARK = "#0a0c10"

# --- Configuration de la Page ---
def setup_page_config():
    st.set_page_config(
        page_title="Pitie-Salpetriere | Vision 2026",
        layout="wide",
    )

# --- Style Global ---
def get_global_style():
    return f"""
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

    /* Nettoyage UI Streamlit Radical - Maintien Visibilite Toggle */
    header[data-testid="stHeader"] {{
        background: rgba(0,0,0,0) !important;
        color: white !important;
    }}
    
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    .main .block-container {{
        padding-top: 2rem !important;
    }}

    /* Ajustements Responsifs */
    @media (max-width: 900px) {{
        .hero-container {{
            flex-direction: column !important;
            text-align: center;
            max-height: none !important;
            padding-bottom: 50px;
        }}
        .wow-title {{
            font-size: 2.5rem !important;
        }}
        .wow-sub {{
            font-size: 1.1rem !important;
        }}
    }}

    /* Style Global des Boutons */
    .stButton>button {{
        background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, #003d6b 100%);
        color: white !important;
        border-radius: 25px;
        padding: 12px 40px;
        border: 1px solid rgba(255,255,255,0.1);
        font-weight: 600;
        letter-spacing: 1px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        font-size: 0.9rem;
    }}

    /* Specifique Page d'Accueil */
    .hero-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
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
        padding: 20px;
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

    /* Onglets Dashboard */
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
"""

def apply_global_style():
    st.markdown(get_global_style(), unsafe_allow_html=True)

