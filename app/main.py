import streamlit as st
import warnings
from config import setup_page_config, apply_global_style
from views.landing import show_landing_page
from views.dashboard import show_dashboard


warnings.filterwarnings('ignore')

# --- Initialisation ---
setup_page_config()
apply_global_style()

# --- Gestion de Session ---
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

# --- Fonctions de Navigation ---
def go_to_dashboard():
    st.session_state.page = 'dashboard'
    # st.rerun() is handled by streamlit automatically on state change usually, 
    

# --- Logique de Routing ---
if st.session_state.page == 'landing':
    show_landing_page(go_to_dashboard)
elif st.session_state.page == 'dashboard':
    show_dashboard()