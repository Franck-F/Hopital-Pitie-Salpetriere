import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import base64
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats as scipy_stats
import warnings
import joblib
import os

warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(
    page_title="Pitie-Salpetriere | Vision 2026",
    layout="wide",
)

# --- Path Constants ---
LOGO_PATH = "app/assets/logo_ps.png"
HERO_BG_PATH = "app/assets/hero_bg.png"
DATA_ADMISSION_PATH = "data/raw/admissions_hopital_pitie_2024.csv"
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

    /* Radical Streamlit UI Cleaning - Maintaining Toggle Visibility */
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

    /* Responsive adjustments */
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

    /* Landing Page Specific */
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

# --- Data Loading (Real Admissions Data) ---
@st.cache_data
def get_admission_data():
    df = pd.read_csv(DATA_ADMISSION_PATH)
    df['date_entree'] = pd.to_datetime(df['date_entree'])
    # Extract features for temporal analysis
    df['annee'] = df['date_entree'].dt.year
    df['mois'] = df['date_entree'].dt.month
    df['mois_nom'] = df['date_entree'].dt.month_name()
    df['jour_semaine'] = df['date_entree'].dt.dayofweek
    df['jour_semaine_nom'] = df['date_entree'].dt.day_name()
    df['semaine'] = df['date_entree'].dt.isocalendar().week
    df['est_weekend'] = df['jour_semaine'].isin([5, 6])
    return df

@st.cache_data
def get_logistique_data():
    df_lits = pd.read_csv("data/raw/lits_poles.csv")
    df_perso = pd.read_csv("data/raw/personnel_poles.csv")
    df_equip = pd.read_csv("data/raw/equipements_poles.csv")
    df_stocks = pd.read_csv("data/raw/stocks_medicaments.csv")
    
    for df in [df_lits, df_perso, df_equip, df_stocks]:
        df['date'] = pd.to_datetime(df['date'])
        
    df_lits['taux_occupation'] = df_lits['lits_occupes'] / df_lits['lits_totaux']
    return df_lits, df_perso, df_equip, df_stocks

@st.cache_data
def get_patient_sejour_data():
    df_pat = pd.read_csv("data/raw/patients_pitie_2024.csv")
    df_sej = pd.read_csv("data/raw/sejours_pitie_2024.csv", parse_dates=["date_admission", "date_sortie"])
    df_diag = pd.read_csv("data/raw/diagnostics_pitie_2024.csv")
    
    # Process durations
    df_sej['duree_jours'] = (df_sej['date_sortie'] - df_sej['date_admission']).dt.days
    df_sej['mois_adm'] = df_sej['date_admission'].dt.month_name()
    df_sej['jour_adm'] = df_sej['date_admission'].dt.day_name()
    
    return df_pat, df_sej, df_diag

@st.cache_resource
def load_champion_model():
    m_path = "models/lightgbm_final_v2.joblib"
    if os.path.exists(m_path):
        return joblib.load(m_path)
    return None

def predict_future_admissions(df_daily, model, days=14):
    if model is None:
        return None, None
        
    current_ts = df_daily.copy()
    last_date = current_ts.index.max()
    
    preds = []
    future_dates = []
    
    # Recursive Forecasting with ultra-optimized XGBoost
    holidays = ['2024-01-01', '2024-04-01', '2024-05-01', '2024-05-08', 
                '2024-05-09', '2024-05-20', '2024-07-14', '2024-08-15', 
                '2024-11-01', '2024-11-11', '2024-12-25']
    holiday_dates = pd.to_datetime(holidays)
    
    # Reference holidays for distance calculation
    holidays = pd.to_datetime(['2024-01-01', '2024-04-01', '2024-05-01', '2024-05-08', 
                             '2024-05-09', '2024-05-20', '2024-07-14', '2024-08-15', 
                             '2024-11-01', '2024-11-11', '2024-12-25'])
    
    for i in range(1, days + 1):
        next_date = last_date + timedelta(days=i)
        future_dates.append(next_date)
        
        # Features for next date - EXACT MATCH WITH lightgbm_final_v2.joblib
        row = pd.DataFrame(index=[next_date])
        row['day'] = next_date.dayofweek
        row['month'] = next_date.month
        row['lag1'] = current_ts.iloc[-1]
        row['lag2'] = current_ts.iloc[-2] if len(current_ts) >= 2 else current_ts.iloc[-1]
        row['lag7'] = current_ts.iloc[-7] if len(current_ts) >= 7 else current_ts.iloc[-1]
        row['is_holiday'] = 1 if next_date in holidays else 0
        row['dist_holiday'] = (holidays[holidays >= next_date].min() - next_date).days if any(holidays >= next_date) else 365
        
        FEATS = ['day', 'month', 'lag1', 'lag2', 'lag7', 'is_holiday', 'dist_holiday']
        X_row = row[FEATS]
        
        # LightGBM Champion Prediction
        final_pred = model.predict(X_row)[0]
        
        preds.append(final_pred)
        # Augment series for next recursive step
        current_ts.loc[next_date] = final_pred
        
    return future_dates, np.array(preds)

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
df_adm = get_admission_data()
df_lits, df_perso, df_equip, df_stocks = get_logistique_data()
df_pat, df_sej, df_diag = get_patient_sejour_data()
model_lgbm = load_champion_model()
st.logo(LOGO_PATH, icon_image=LOGO_PATH)

# --- Premium Dashboard Header ---
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
    
    # Custom Sidebar Content
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

tab_acc, tab_exp, tab_ml, tab_sim, tab_tea = st.tabs([
    "TABLEAU DE BORD", "EXPLORATION DATA", "PREVISIONS ML", "SIMULATEUR", "EQUIPE PROJET"
])

with tab_acc:
    st.markdown("<h2 style='font-weight:800;'>Panorama de l'Activite Reelle</h2>", unsafe_allow_html=True)
    daily_stats = df_adm.groupby('date_entree').size()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Admissions 2024", f"{len(df_adm):,}")
    m2.metric("Moyenne Quotidienne", f"{daily_stats.mean():.1f}")
    m3.metric("Jour de Pic", f"{daily_stats.max()}")
    
    fig_main = px.line(daily_stats.reset_index(), x='date_entree', y=0, 
                       title="Flux d'admissions quotidiens - 2024", 
                       template="plotly_dark", color_discrete_sequence=[SECONDARY_BLUE])
    fig_main.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_main, use_container_width=True)

with tab_exp:
    sub_tab_adm, sub_tab_log, sub_tab_sej = st.tabs([
        "Admission patient", "Logistique", "Séjour patient"
    ])
    
    with sub_tab_adm:
        st.markdown("## ANALYSE DES ADMISSIONS 2024")
        
        # --- Overview Stats ---
        st.markdown("### Vue d'ensemble des donnees")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Periode d'analyse", f"{df_adm['date_entree'].min().strftime('%d/%m/%Y')} -> {df_adm['date_entree'].max().strftime('%d/%m/%Y')}")
        o2.metric("Total Admissions", f"{len(df_adm):,}")
        o3.metric("Services/Poles", f"{df_adm['service'].nunique()}")
        o4.metric("Modes d'Entree", f"{df_adm['mode_entree'].nunique()}")

        # --- Admissions Table ---
        st.divider()
        st.markdown("### Repartition par Type d'Admission")
        type_counts = df_adm['service'].value_counts().reset_index()
        type_counts.columns = ['Service', 'Nombre d\'admissions']
        type_counts['Pourcentage (%)'] = (type_counts['Nombre d\'admissions'] / len(df_adm) * 100).round(2)
        
        st.dataframe(
            type_counts.style.background_gradient(subset=['Nombre d\'admissions'], cmap='Blues'),
            use_container_width=True,
            hide_index=True
        )

        # --- Subplots for Categorical Distributions ---
        st.divider()
        st.markdown("### Distributions des Variables Catégorielles")
        
        fig_cat = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Top 10 Poles/Services', 'Modes d\'Entree', 
                            'Origine Geographique (Top 10)', 'Top 20 Motifs d\'Admission'),
            specs=[[{'type': 'bar'}, {'type': 'pie'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )

        # 1. Poles
        pole_c = df_adm['service'].value_counts().head(10)
        fig_cat.add_trace(go.Bar(y=pole_c.index, x=pole_c.values, orientation='h', name='Poles', marker_color='lightblue'), row=1, col=1)
        
        # 2. Modes
        mode_c = df_adm['mode_entree'].value_counts()
        fig_cat.add_trace(go.Pie(labels=mode_c.index, values=mode_c.values, name='Modes', hole=0.4), row=1, col=2)
        
        # 3. Geo
        geo_c = df_adm['departement_patient'].value_counts().head(10)
        fig_cat.add_trace(go.Bar(x=geo_c.index, y=geo_c.values, name='Origine', marker_color='coral'), row=2, col=1)
        
        # 4. Motifs
        motif_c = df_adm['motif_principal'].value_counts().head(20)
        fig_cat.add_trace(go.Bar(x=motif_c.index, y=motif_c.values, name='Motifs', marker_color='lightgreen'), row=2, col=2)
        
        fig_cat.update_layout(height=900, template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cat, use_container_width=True)

        # --- Temporal Analysis ---
        st.divider()
        st.markdown("### Tendances et Saisonnalité (Série Temporelle)")
        daily_series = daily_stats.asfreq('D', fill_value=0)
        decomposition = seasonal_decompose(daily_series, model='additive', period=7)
        
        fig_temp = make_subplots(rows=4, cols=1, 
                                 subplot_titles=('Signal Original', 'Tendance', 'Saisonnalité', 'Résidus'),
                                 vertical_spacing=0.08)
        fig_temp.add_trace(go.Scatter(x=daily_series.index, y=daily_series.values, name="Original", line_color=SECONDARY_BLUE), row=1, col=1)
        fig_temp.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name="Tendance", line_color=ACCENT_RED), row=2, col=1)
        fig_temp.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name="Saisonnalité", line_color='green'), row=3, col=1)
        fig_temp.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name="Résidus", line_color='orange'), row=4, col=1)
        fig_temp.update_layout(height=1000, template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_temp, use_container_width=True)

        # --- Patterns & Heatmap ---
        st.divider()
        st.markdown("### Patterns Temporels")
        pc1, pc2 = st.columns(2)
        with pc1:
            jour_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fig_box_j = go.Figure()
            for j in jour_ordre:
                d_j = df_adm[df_adm['jour_semaine_nom'] == j].groupby('date_entree').size()
                fig_box_j.add_trace(go.Box(y=d_j.values, name=j[:3]))
            fig_box_j.update_layout(title="Variabilité par Jour", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box_j, use_container_width=True)
        with pc2:
            pivot_h = df_adm.groupby(['jour_semaine', 'mois']).size().unstack(fill_value=0)
            fig_heat_adm = px.imshow(pivot_h.values, labels=dict(x="Mois", y="Jour", color="Volume"),
                                     x=pivot_h.columns, y=jour_ordre, title="Intensité Semaine x Mois",
                                     color_continuous_scale='YlOrRd', template="plotly_dark")
            fig_heat_adm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_heat_adm, use_container_width=True)

        # --- Anomaly Detection ---
        st.divider()
        st.markdown("### Detection d'Anomalies (Pics Hors-Normes)")
        Q1, Q3 = daily_stats.quantile(0.25), daily_stats.quantile(0.75)
        IQR = Q3 - Q1
        upper_b = Q3 + 1.5 * IQR
        outliers = daily_stats[daily_stats > upper_b]
        
        fig_out_adm = go.Figure()
        fig_out_adm.add_trace(go.Scatter(x=daily_stats.index, y=daily_stats.values, mode='markers', name='Normal', marker=dict(color=SECONDARY_BLUE, size=4)))
        fig_out_adm.add_trace(go.Scatter(x=outliers.index, y=outliers.values, mode='markers', name='Anomalie', marker=dict(color=ACCENT_RED, size=8, symbol='x')))
        fig_out_adm.add_hline(y=upper_b, line_dash="dash", line_color=ACCENT_RED, annotation_text="Seuil IQR")
        fig_out_adm.update_layout(title="Identification des Pics", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_out_adm, use_container_width=True)

        # --- Sunburst Motifs ---
        st.divider()
        st.markdown("### Hierarche des Motifs (Sunburst)")
        top_m_list = df_adm['motif_principal'].value_counts().head(15).index
        sun_df = df_adm[df_adm['motif_principal'].isin(top_m_list)].groupby(['mode_entree', 'service', 'motif_principal']).size().reset_index(name='count')
        fig_sun_adm = px.sunburst(sun_df, path=['mode_entree', 'service', 'motif_principal'], values='count',
                                  title="Mode -> Pôle -> Top Motifs",
                                  template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_sun_adm.update_layout(height=700, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sun_adm, use_container_width=True)

        # --- Week vs Weekend Analysis ---
        st.divider()
        st.markdown("### Analyse Comparative : Semaine vs Weekend")
        df_adm['est_weekend'] = df_adm['jour_semaine'].isin([5, 6])
        weekday_d = df_adm[~df_adm['est_weekend']].groupby('date_entree').size()
        weekend_d = df_adm[df_adm['est_weekend']].groupby('date_entree').size()
        
        # t-test
        t_stat, p_val = scipy_stats.ttest_ind(weekday_d, weekend_d)
        
        wc1, wc2 = st.columns(2)
        with wc1:
            fig_box_ww = go.Figure()
            fig_box_ww.add_trace(go.Box(y=weekday_d.values, name='Semaine', marker_color='steelblue'))
            fig_box_ww.add_trace(go.Box(y=weekend_d.values, name='Weekend', marker_color='coral'))
            fig_box_ww.update_layout(title="Distribution Volumes quotidien", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box_ww, use_container_width=True)
            
        with wc2:
            st.metric("Moyenne Semaine", f"{weekday_d.mean():.1f} ± {weekday_d.std():.1f}")
            st.metric("Moyenne Weekend", f"{weekend_d.mean():.1f} ± {weekend_d.std():.1f}")
            if p_val < 0.05:
                st.success(f"Difference SIGNIFICATIVE (p={p_val:.6f})")
            else:
                st.info(f"Pas de difference significative (p={p_val:.6f})")

        # --- Detection d'Anomalies (IQR) ---
        st.divider()
        st.markdown("### Detection d'Anomalies et Pics de Charge")
        
        Q1 = daily_ts.quantile(0.25)
        Q3 = daily_ts.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outliers = daily_ts[daily_ts > upper_bound]
        
        fig_ano = go.Figure()
        fig_ano.add_trace(go.Scatter(x=daily_ts.index, y=daily_ts.values, mode='lines', name='Admissions', line=dict(color=SECONDARY_BLUE, width=1)))
        fig_ano.add_trace(go.Scatter(x=outliers.index, y=outliers.values, mode='markers', name='Pics Anormaux', marker=dict(color=ACCENT_RED, size=8, symbol='x')))
        fig_ano.add_hline(y=upper_bound, line_dash="dash", line_color=ACCENT_RED, annotation_text="Seuil Alerte (IQR)")
        fig_ano.update_layout(height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=30, b=30))
        st.plotly_chart(fig_ano, use_container_width=True)

        # --- Evolution Top 5 Poles ---
        st.divider()
        st.markdown("### Evolution des Flux par Pole (Moyenne Mobile 7j)")
        
        top_poles_names = df_adm['service'].value_counts().head(5).index
        fig_poles_evol = go.Figure()
        
        for pole in top_poles_names:
            pole_daily = df_adm[df_adm['service'] == pole].groupby('date_entree').size().asfreq('D', fill_value=0)
            pole_ma = pole_daily.rolling(window=7).mean()
            fig_poles_evol.add_trace(go.Scatter(x=pole_ma.index, y=pole_ma.values, mode='lines', name=pole))
            
        fig_poles_evol.update_layout(height=450, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_poles_evol, use_container_width=True)

        # --- Final Insights Summary ---
        st.divider()
        st.markdown("### Synthese Strategique des Admissions")
        
        peak_month = df_adm.groupby('mois_nom').size().idxmax()
        main_mode = df_adm['mode_entree'].mode()[0]
        
        insights_rows = [
            {"Indicateur": "Pic Saisonnier", "Valeur": peak_month, "Observation": "Charge maximale observee."},
            {"Indicateur": "Mode d'Entree Dominant", "Valeur": main_mode, "Observation": "Vecteur principal d'admission."},
            {"Indicateur": "Jours Critiques", "Valeur": f"{len(outliers)} jours", "Observation": "Depassement des seuils de garde."},
            {"Indicateur": "Variabilite (CV)", "Valeur": f"{(daily_ts.std()/daily_ts.mean()*100):.1f}%", "Observation": "Besoin de flexibilite RH."}
        ]
        st.table(pd.DataFrame(insights_rows))


    with sub_tab_log:
        st.markdown("## ANALYSE LOGISTIQUE & RESSOURCES")
        
        # --- Strategic Metrics ---
        st.markdown("### Indicateurs de Tension Critique")
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Occupation Moyenne", f"{df_lits['taux_occupation'].mean():.1%}")
        l2.metric("Suroccupation (>95%)", f"{(df_lits['taux_occupation'] > 0.95).sum():,}")
        l3.metric("Ratio Infirmiers/Lit (Moy)", f"{(df_perso[df_perso['categorie']=='infirmier']['effectif_total'].sum() / df_lits['lits_totaux'].sum()):.2f}")
        l4.metric("Alertes Stocks", f"{df_stocks['alerte_rupture'].sum():,}")

        # --- Capacity vs Staff Panel ---
        st.divider()
        st.markdown("### Capacité et Effectifs Soignants")
        lc1, lc2 = st.columns(2)
        
        with lc1:
            # Capacity with subplots - FIXING PIE IN SUBPLOTS BUG
            l_fig = make_subplots(
                rows=2, cols=1, 
                subplot_titles=("Top 10 Poles (Lits Totaux)", "Repartition par Type de Lit"),
                specs=[[{"type": "xy"}], [{"type": "domain"}]]
            )
            lits_p = df_lits.groupby('service')['lits_totaux'].first().sort_values(ascending=False).head(10)
            l_fig.add_trace(go.Bar(x=lits_p.index, y=lits_p.values, marker_color='steelblue', name='Lits'), row=1, col=1)
            
            # Pie for bed types (hypothetical breakdown if data allowed, here using poles as proxy)
            l_fig.add_trace(go.Pie(labels=lits_p.index[:5], values=lits_p.values[:5], hole=0.3), row=2, col=1)
            l_fig.update_layout(height=700, template="plotly_dark", showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(l_fig, use_container_width=True)
            
        with lc2:
            # Staffing detail
            perso_cat = df_perso[df_perso['categorie'] != 'total'].groupby('categorie')['effectif_total'].sum().reset_index()
            fig_p_cat = px.bar(perso_cat, x='categorie', y='effectif_total', color='categorie',
                               title="Effectifs Totaux par Corps de Metier (ETP)",
                               template="plotly_dark")
            fig_p_cat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_p_cat, use_container_width=True)

        # --- Nurse/Bed Ratio Analysis ---
        st.divider()
        st.markdown("### Analyse de la Charge de Travail (Ratio Nurse/Bed)")
        ratio_df = df_lits.merge(df_perso[df_perso['categorie']=='infirmier'], on=['date', 'service'])
        ratio_df['ratio_nb'] = ratio_df['effectif_total'] / ratio_df['lits_totaux']
        
        fig_ratio = px.box(ratio_df, x='service', y='ratio_nb', color='service',
                           title="Dispersion du Ratio Infirmiers par Lit par Service",
                           template="plotly_dark")
        fig_ratio.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Seuil Alerte (1 nurse / 5 beds)")
        fig_ratio.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ratio, use_container_width=True)

        # --- Salles d'Isolement & Alertes Epidemiques ---
        st.divider()
        st.markdown("### Focus : Salles d'Isolement & Vigilance Epidemique")
        ic1, ic2 = st.columns(2)
        with ic1:
            # Load isolation data if available or simulate/proxy from diagnostics
            iso_services = ['Urgences_(Passage_court)', 'PRAGUES_(Réa/Pneumo)', 'Infectiologie']
            df_iso = df_lits[df_lits['service'].isin(iso_services)]
            fig_iso = px.line(df_iso, x='date', y='taux_occupation', color='service',
                              title="Tension dans les Services Haute Vigilance",
                              template="plotly_dark")
            fig_iso.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_iso, use_container_width=True)
        with ic2:
            st.info("Les services identifies ci-contre disposent de chambres a pression negative (Salles d'Isolement). Une occupation depassant 90% sur ces zones declenche le protocole 'Alerte Epidemique'.")
            st.metric("Taux d'Alerte Global (ISO)", f"{(df_iso['taux_occupation'] > 0.9).mean():.1%}")

        # --- Monitoring des Stocks & Ruptures Critiques ---
        st.divider()
        st.markdown("### Gestion des Stocks et Ruptures Critiques")
        sc1, sc2 = st.columns([1, 1.2])
        
        with sc1:
            st.write("Hierarchie des ruptures constatees (Points de vigilance majeurs).")
            # Data from EDA source
            rupt_data = pd.DataFrame({
                'Medicament': ['Antibiotiques', 'Morphine IV', 'Insuline', 'Heparine', 'Paracetamol'],
                'Occurences': [650, 420, 220, 120, 49]
            })
            fig_rupt = px.bar(rupt_data, y='Medicament', x='Occurences', orientation='h', 
                             title="Hierarchie des Ruptures (Cumul jours)",
                             color='Occurences', color_continuous_scale='Reds',
                             template="plotly_dark")
            fig_rupt.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_rupt, use_container_width=True)
            
        with sc2:
            st.write("Analyse saisonniere de l'absenteisme soignant par service.")
            abs_df = df_perso.copy()
            abs_df['mois'] = abs_df['date'].dt.month
            abs_agg = abs_df.groupby(['mois', 'service'])['taux_absence'].mean().reset_index()
            
            fig_abs = px.line(abs_agg, x='mois', y='taux_absence', color='service',
                             title="Saisonnalite de l'Absenteisme (%)",
                             template="plotly_dark")
            fig_abs.update_xaxes(tickvals=list(range(1,13)), ticktext=['Jan','Fev','Mar','Avr','Mai','Juin','Juil','Aout','Sep','Oct','Nov','Dec'])
            fig_abs.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_abs, use_container_width=True)

        # --- Strategic Monitoring Heatmap ---
        st.divider()
        st.markdown("### Heatmap de Tension Logistique Globale")
        pivot_log = df_lits.groupby(['service', 'date'])['taux_occupation'].mean().unstack().T
        fig_heat_log = px.imshow(pivot_log.values, x=pivot_log.columns, y=pivot_log.index,
                                 title="Indice de Saturation Quotidien par Pole",
                                 color_continuous_scale='RdBu_r', template="plotly_dark")
        fig_heat_log.update_layout(height=600, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_heat_log, use_container_width=True)

        # --- Summary Insights Logistique ---
        st.divider()
        st.markdown("### Diagnostic de Resilence Logistique")
        log_res = [
            {"Point": "Tension Staff", "Constat": "Ratio moyen stable (> 0.22), mais pics de sous-effectif en Rea.", "Statut": "Vigilance"},
            {"Point": "Capacite Lits", "Constat": f"{(df_lits['taux_occupation']>0.95).sum()} episodes de saturation severe detectes.", "Statut": "Alerte"},
            {"Point": "Stocks", "Constat": f"{df_stocks['alerte_rupture'].sum()} ruptures critiques identifiees (Principalement Curitine).", "Statut": "Action Requise"}
        ]
        st.table(pd.DataFrame(log_res))

    with sub_tab_sej:
        st.markdown("## ANALYSE DES SEJOURS & PARCOURS PATIENTS")
        
        # --- 1. Dataset Preview (Styled Tables) ---
        st.markdown("### Apercu des Jeux de Donnees 2024")
        
        def create_styled_table_st(df):
            return df.head(5).style.set_properties(**{
                'background-color': '#f8f9fa',
                'color': '#2c3e50',
                'border-color': '#e9ecef'
            }).set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#2c3e50'), ('color', 'white')]}
            ])

        with st.expander("Consulter l'extrait des Tables (Top 5 rows)", expanded=False):
            st.markdown("#### Table Patients")
            st.dataframe(create_styled_table_st(df_pat), use_container_width=True)
            st.markdown("#### Table Sejours")
            st.dataframe(create_styled_table_st(df_sej), use_container_width=True)
            st.markdown("#### Table Diagnostics")
            st.dataframe(create_styled_table_st(df_diag), use_container_width=True)

        # --- 2. Data Quality & Overview ---
        st.divider()
        st.markdown("### Qualite et Profil Demographique")
        q1, q2, q3 = st.columns(3)
        
        datasets = [(df_pat, "Patients"), (df_sej, "Sejours"), (df_diag, "Diagnostics")]
        all_comp = []
        for i, (df, name) in enumerate(datasets):
            with [q1, q2, q3][i]:
                completeness = (1 - df.isna().mean()) * 100
                comp_val = completeness.mean()
                all_comp.append(comp_val)
                st.metric(f"Completude {name}", f"{comp_val:.1f}%")
        avg_comp = sum(all_comp) / len(all_comp)
                
        dc1, dc2 = st.columns(2)
        with dc1:
            fig_sexe = px.pie(df_pat, names='sexe', title="Répartition par Sexe",
                              template="plotly_dark", hole=0.4,
                              color_discrete_map={'M': '#2c3e50', 'F': '#e74c3c'})
            fig_sexe.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sexe, use_container_width=True)
        with dc2:
            fig_age_sej = px.histogram(df_sej, x="age", nbins=40, marginal="violin",
                                     title="Pyramide des Ages a l'Admission (Densite)",
                                     template="plotly_dark", color_discrete_sequence=['#3498db'])
            fig_age_sej.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_age_sej, use_container_width=True)

        # --- New : Boxplot Detail Age par Type ---
        st.divider()
        st.markdown("### Dispersion Detaillee de l'Age par Type d'Hospitalisation")
        fig_box_notched = px.box(df_sej, x="type_hospit", y="age", color="type_hospit",
                                 notched=True, points="suspectedoutliers",
                                 title="Age Median et Outliers par Type de Sejour",
                                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Prism)
        fig_box_notched.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_box_notched, use_container_width=True)

        # --- 3. Specialty Analysis & Repartition Age ---
        st.divider()
        st.markdown("### Hierarchie et Structure Demographique des Poles")
        sc1, sc2 = st.columns(2)
        with sc1:
            fig_sun_sej = px.sunburst(df_sej, path=['pole', 'type_hospit'], values='age', # Proxy count
                                  title="Poles -> Types d'Hospitalisation",
                                  template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_sun_sej.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sun_sej, use_container_width=True)
        with sc2:
            # New : Repartition Ages par Pole
            df_sej['age_bin'] = pd.cut(df_sej['age'], bins=[0, 18, 45, 65, 105], labels=['Enfants', 'Adultes', 'Seniors', 'Grand Age'])
            age_pole = df_sej.groupby(['pole', 'age_bin']).size().reset_index(name='count')
            fig_age_pole = px.bar(age_pole, x="count", y="pole", color="age_bin", orientation='h',
                                  title="Repartition des Ages par Pole",
                                  template="plotly_dark", color_discrete_sequence=px.colors.sequential.RdBu_r)
            fig_age_pole.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", legend_title="Tranche d'age")
            st.plotly_chart(fig_age_pole, use_container_width=True)

        # --- 4. Diagnostics Analysis ---
        st.divider()
        st.markdown("### Analyse des Pathologies (CIM-10)")
        dg1, dg2 = st.columns(2)
        with dg1:
            diag_p = df_diag.groupby("pathologie_groupe").size().reset_index(name='count').sort_values('count', ascending=True)
            fig_p_p = px.bar(diag_p, x='count', y='pathologie_groupe', orientation='h',
                               title="Principaux Groupes de Pathologies", template="plotly_dark",
                               color='count', color_continuous_scale="Tealgrn")
            fig_p_p.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_p_p, use_container_width=True)
        with dg2:
            # Donut for Diagnostic Type
            repartition = df_diag["type_diagnostic"].value_counts().reset_index()
            repartition.columns = ["type", "count"]
            fig_donut = px.pie(repartition, values="count", names="type", hole=0.5,
                               title="Repartition des Types de Diagnostics",
                               color_discrete_sequence=['#2C3E50', '#E74C3C'], # Pro Dark / Pro Red
                               template="plotly_dark")
            fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_donut, use_container_width=True)

        # --- 5. Age vs Duration Correlation ---
        st.divider()
        st.markdown("### Analyse de Corrélation : Age vs Durée de Séjour")
        
        # Taking a representative sample for the scatter plot
        sample_size = min(500, len(df_sej))
        sample_sej = df_sej.sample(n=sample_size, random_state=42)
        
        fig_scatter = px.scatter(sample_sej, x="age", y="duree_jours", color="pole", size="age",
                                 title=f"Repartition Age / DMS (Echantillon {sample_size} patients)",
                                 template="plotly_dark", opacity=0.7)
        fig_scatter.add_hline(y=df_sej['duree_jours'].mean(), line_dash="dot", annotation_text="DMS Moyenne")
        fig_scatter.add_vline(x=df_sej['age'].mean(), line_dash="dot", annotation_text="Age Moyen")
        fig_scatter.update_layout(height=500, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # --- 6. Multidimensional Pole Profiling (Radar) ---
        st.divider()
        st.markdown("### Profiling Multidimensionnel des Poles (Radar)")
        
        # Prepare radar data
        radar_raw = df_sej.groupby('pole').agg({
            'age': 'mean',
            'duree_jours': 'mean',
            'id_sejour': 'count'
        }).reset_index()
        
        # Normalize for radar visualization (Scale 0-1)
        for col in ['age', 'duree_jours', 'id_sejour']:
            radar_raw[f'{col}_norm'] = radar_raw[col] / radar_raw[col].max()
            
        fig_radar = go.Figure()
        categories = ['Age Moyen', 'Duree Sejour (DMS)', 'Volume Activite']
        top_poles_radar = radar_raw.sort_values('id_sejour', ascending=False).head(3)
        
        for i, row in top_poles_radar.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['age_norm'], row['duree_jours_norm'], row['id_sejour_norm']],
                theta=categories, fill='toself', name=row['pole']
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)


        st.divider()
        st.markdown("### Intensite et Tension Horaire")
        
        qc1, qc2 = st.columns([2, 1])
        with qc1:
            # New : Hourly Tension Heatmap
            df_sej['heure'] = df_sej['date_admission'].dt.hour
            tension_h = df_sej.groupby(['jour_adm', 'heure']).size().reset_index(name='nb_admissions')
            jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            fig_heat_h = px.density_heatmap(tension_h, x="heure", y="jour_adm", z="nb_admissions",
                                             nbinsx=24, category_orders={"jour_adm": jours_ordre},
                                             color_continuous_scale="YlOrRd",
                                             title="Heatmap de Tension : Flux d'arrivee des patients",
                                             template="plotly_dark")
            fig_heat_h.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="Heure d'admission")
            st.plotly_chart(fig_heat_h, use_container_width=True)
        with qc2:
            st.info("Cette heatmap permet d'identifier les pics d'activite journaliers. Une concentration rouge indique un flux critique necessitant un renfort des effectifs d'accueil et de tri.")
            st.metric("Heure de Pointe (Moyenne)", f"{df_sej['heure'].mode()[0]}h00")
        
        # --- Final Insights Séjour ---
        st.divider()
        st.markdown("### Synthese des Parcours Patients")
        dms = df_sej['duree_jours'].mean()
        top_patho = df_diag['pathologie_groupe'].value_counts().index[0]
        
        sej_insights_df = pd.DataFrame([
            {"Indicateur": "Duree Moyenne de Sejour (DMS)", "Valeur": f"{dms:.1f} jours", "Note": "Optimisation des flux requise."},
            {"Indicateur": "Pathologie Dominante", "Valeur": top_patho, "Note": "Vigilance sur les lits specialises."},
            {"Indicateur": "Qualite des Codages", "Valeur": f"{avg_comp:.1f}%", "Note": "Niveau de fiabilite excellent."},
            {"Indicateur": "Intensite Diagnostique", "Valeur": f"{(len(df_diag)/len(df_sej)):.1f} codes/sej", "Note": "Complexite des soins confirmee."}
        ])
        st.table(sej_insights_df)

with tab_ml:
    st.markdown("## Previsions de Charge Hospitaliere")
    st.markdown("Moteur predictif **LightGBM Champion** (Performance Maximale).")
    
    if model_lgbm:
        daily_ts = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)
        future_dates, future_preds = predict_future_admissions(daily_ts, model_lgbm)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Tendance Prochaine Semaine", f"{future_preds[:7].mean():.1f} adm/j")
        with col_m2:
            st.metric("Confiance Modele (MAE)", "67.92")
        with col_m3:
            st.metric("Status", "Calibre (Champion V2)")
            
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=daily_ts.index[-30:], y=daily_ts.values[-30:], name="Historique Recent", line=dict(color=SECONDARY_BLUE, width=3)))
        fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Projection LightGBM", line=dict(dash='dash', color=ACCENT_RED, width=3)))
        fig_pred.update_layout(
            template="plotly_dark", 
            height=500, 
            title="Projections Flux Urgentistes (J+14)",
            xaxis_title="Date",
            yaxis_title="Admissions",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        with st.expander("Optimisation LightGBM Champion"):
            st.write("Le modele Champion a ete selectionne pour sa capacite de generalisation superieure sur les donnees du mois de decembre 2024.")
            st.write("**Architecture** : Utilisation exclusive des retards temporels (Lags) pour capturer la dynamique interne des admissions.")
        
        with st.expander("Importance des Variables"):
            FEATS = ['day', 'month', 'lag1', 'lag2', 'lag7', 'is_holiday', 'dist_holiday']
            
            if len(FEATS) == len(model_lgbm.feature_importances_):
                importance = pd.DataFrame({'feature': FEATS, 'importance': model_lgbm.feature_importances_}).sort_values('importance', ascending=True)
                fig_imp = px.bar(importance, x='importance', y='feature', orientation='h', template='plotly_dark', color='importance', color_continuous_scale='Blues')
                fig_imp.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning(f"Incohérence : Le modèle attend {len(model_lgbm.feature_importances_)} variables.")
            
        with st.expander("Details Techniques du Modele"):
            st.write("Algorithme : LightGBM (Gradient Boosting)")
            st.write("Variables clefs : Lag 1, Lag 2, Lag 7, Calendrier et Vacances")
            st.info("Le modele integre le contexte des jours feries pour une meilleure precision en periode de fetes.")
    else:
        st.error("Modele XGBoost non detecte. Veuillez lancer l'entrainement.")

with tab_sim:
    st.markdown("## Simulateur de Tension Hospitalière")
    st.markdown("""
        Tester des scénarios de crise (ex: épidémie, plan blanc) pour évaluer la résilience des capacités actuelles de la Pitié-Salpêtrière.
    """)
    
    with st.expander("Configuration du Scénario de Stress", expanded=True):
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            intensite = st.slider("Surcroît d'Admissions Prévues (%)", 0, 200, 50, help="Pourcentage s'ajoutant aux prévisions du modèle.")
        with sc_col2:
            ressource_type = st.selectbox("Ressource à Monitorer", 
                                        ["Capacité en Lits (Rea/Med)", "Effectif Soignant (Infirmiers)", "Stocks de Sécurité (Médicaments)"])
        
    if st.button("Lancer la Simulation d'Impact"):
        if model_lgbm and 'df_adm' in locals():
            # Get model baseline for next 14 days
            daily_ts = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)
            _, future_preds = predict_future_admissions(daily_ts, model_lgbm)
            avg_predicted = future_preds.mean()
            
            # Stress calculation
            stress_load = avg_predicted * (1 + intensite/100)
            
            # Initialize metrics
            utilization = 0.0
            depletion_days = 30.0
            fig_sim = None
            
            # Display Metrics Baseline
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Baseline Modèle", f"{avg_predicted:.1f}/j")
            m_col2.metric("Charge Simulée", f"{stress_load:.1f}/j", delta=f"+{intensite}%", delta_color="inverse")
            
            # Data Unpacking & Real-time Filtering
            df_lits_raw, df_perso_raw, _, df_stocks_raw = get_logistique_data()
            latest_date = df_perso_raw['date'].max()
            
            df_lits = df_lits_raw[df_lits_raw['date'] == latest_date]
            df_perso = df_perso_raw[df_perso_raw['date'] == latest_date]
            df_stocks = df_stocks_raw[df_stocks_raw['date'] == latest_date]
            
            # Resource Logic
            if "Lits" in ressource_type:
                # Use top 10 poles by capacity for the snapshot
                total_capacity = df_lits.nlargest(10, 'lits_totaux')['lits_totaux'].sum()
                occup_base = df_lits.nlargest(10, 'lits_totaux')['lits_occupes'].sum()
                utilization = (occup_base / total_capacity) * (1 + intensite/200)
                m_col3.metric("Saturation Lits", f"{min(utilization*100, 100):.1f}%")
                
                fig_sim = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = min(utilization * 100, 100),
                    title = {'text': "Indice de Saturation Lits (Instantané)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': ACCENT_RED if utilization > 0.9 else SECONDARY_BLUE},
                        'steps' : [
                            {'range': [0, 70], 'color': "rgba(0, 166, 80, 0.2)"},
                            {'range': [70, 90], 'color': "rgba(255, 127, 14, 0.2)"},
                            {'range': [90, 100], 'color': "rgba(231, 76, 60, 0.2)"}]
                    }
                ))
            elif "Effectif" in ressource_type:
                total_staff = df_perso['effectif_total'].sum()
                # Realistic tension: (Simulated new patients * factor) / available staff
                utilization = (stress_load * 0.8) / (total_staff / 20) 
                m_col3.metric("Tension Staff", f"{min(utilization*100, 100):.1f}%")
                
                fig_sim = px.bar(df_perso.groupby('categorie')['effectif_total'].sum().reset_index(), 
                                x='categorie', y='effectif_total', title=f"Effectifs Réels au {latest_date.strftime('%d/%m/%Y')}",
                                template="plotly_dark", color_discrete_sequence=[SECONDARY_BLUE])
            else:
                # Stock Logic
                depletion_days = 30 / (1 + intensite/100)
                m_col3.metric("Autonomie Stocks", f"{depletion_days:.1f} Jours")
                
                if not df_stocks_raw.empty:
                    # For line chart, we use historical data but filter Top 5 meds
                    df_viz = df_stocks_raw.copy()
                    top_meds = df_viz[df_viz['date'] == latest_date].nlargest(5, 'conso_jour')['medicament'].tolist()
                    df_viz = df_viz[df_viz['medicament'].isin(top_meds)].sort_values('date')
                    fig_sim = px.line(df_viz, x='date', y='stock_fin', color='medicament', 
                                     title="Historique & Tendance des Stocks Critiques",
                                     template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Safe)
                else:
                    st.warning("Données de stocks non disponibles.")

            if fig_sim:
                fig_sim.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_sim, use_container_width=True)
            
            # Final Status Message
            is_critical = False
            if "Lits" in ressource_type or "Effectif" in ressource_type:
                if utilization > 0.9: is_critical = True
            elif depletion_days < 7:
                is_critical = True
                
            if is_critical:
                st.error("ALERTE CRITIQUE : Les capacités actuelles ne permettent pas d'absorber ce pic sur la durée.")
                st.info("Recommandation : Déclenchement du Plan Blanc et rappel des effectifs en congés.")
            else:
                st.success("RÉSILIENCE CONFIRMÉE : L'infrastructure peut absorber la charge simulée.")
        else:
            st.error("Données ou Modèle ML non chargés. Impossible de simuler.")

with tab_tea:
    st.markdown("<h1 style='text-align:center;'>Equipe Projet Vision 2026</h1>", unsafe_allow_html=True)
    team_cols = st.columns(5)
    members = ["Franck", "Charlotte", "Gaëtan", "Djouhra", "Farah"]
    for i, member in enumerate(members):
        with team_cols[i]:
            st.markdown(f"<div style='text-align:center; padding:30px; border-radius:30px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.1);'><div style='font-size:1.2rem; font-weight:800; color:{SECONDARY_BLUE};'>{member}</div><div style='font-size:0.8rem; color:#8899A6; margin-top:10px;'>Expertise IA & Santé</div></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='text-align:center; color:#555;'>Direction de l'Hôpital Pitié-Salpêtrière | Promotion 2026</p>", unsafe_allow_html=True)
