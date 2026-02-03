import streamlit as st
import plotly.express as px
from config import SECONDARY_BLUE


def render_overview(df_adm, daily_ts):
    st.markdown("<h2 style='font-weight:800;'>Panorama de l'Activite Reelle</h2>", unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Admissions 2024", f"{len(df_adm):,}")
    m2.metric("Moyenne Quotidienne", f"{daily_ts.mean():.1f}")
    m3.metric("Jour de Pic", f"{daily_ts.max()}")
    
    fig_main = px.line(daily_ts.reset_index(), x='date_entree', y='admissions', 
                       title="Flux d'admissions quotidiens - 2024", 
                       template="plotly_dark", color_discrete_sequence=[SECONDARY_BLUE])
    fig_main.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_main, use_container_width=True)
