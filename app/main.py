"""Application Streamlit - Tableau de bord hospitalier."""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_generator import HospitalDataGenerator
from src.analyzer import HospitalAnalyzer
from src.predictor import AdmissionPredictor


# Configuration de la page
st.set_page_config(
    page_title="Tableau de bord - Hôpital Pitié-Salpêtrière",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_or_generate_data():
    """Charge ou génère les données."""
    try:
        admissions = pd.read_csv('data/raw/admissions.csv')
        resources = pd.read_csv('data/raw/resources.csv')
    except FileNotFoundError:
        st.warning("Génération des données... Cela peut prendre quelques secondes.")
        generator = HospitalDataGenerator()
        admissions = generator.generate_admissions()
        resources = generator.generate_resources()
        
        # Sauvegarde
        admissions.to_csv('data/raw/admissions.csv', index=False)
        resources.to_csv('data/raw/resources.csv', index=False)
    
    return admissions, resources


def main():
    """Application principale."""
    
    # En-tête
    st.title("🏥 Hôpital Pitié-Salpêtrière")
    st.markdown("### Système de Prévision et de Gestion des Ressources")
    st.markdown("---")
    
    # Chargement des données
    with st.spinner("Chargement des données..."):
        admissions, resources = load_or_generate_data()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Sélectionner une page",
        ["Vue d'ensemble", "Analyse des admissions", "Prédictions", "Ressources"]
    )
    
    # Initialisation de l'analyseur
    analyzer = HospitalAnalyzer(admissions)
    
    if page == "Vue d'ensemble":
        show_overview(analyzer, admissions, resources)
    
    elif page == "Analyse des admissions":
        show_admissions_analysis(analyzer)
    
    elif page == "Prédictions":
        show_predictions(admissions)
    
    elif page == "Ressources":
        show_resources(resources)


def show_overview(analyzer, admissions, resources):
    """Affiche la vue d'ensemble."""
    st.header("📊 Vue d'ensemble")
    
    # Statistiques principales
    stats = analyzer.get_summary_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total admissions",
            f"{stats['total_admissions']:,}",
            help="Nombre total d'admissions sur la période"
        )
    
    with col2:
        st.metric(
            "Durée moyenne de séjour",
            f"{stats['duree_sejour_moyenne']:.1f} jours",
            help="Durée moyenne d'hospitalisation"
        )
    
    with col3:
        st.metric(
            "Taux d'urgences",
            f"{stats['taux_urgences']:.1f}%",
            help="Pourcentage d'admissions en urgence"
        )
    
    with col4:
        st.metric(
            "Âge moyen",
            f"{stats['age_moyen']:.0f} ans",
            help="Âge moyen des patients"
        )
    
    st.markdown("---")
    
    # Graphiques
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Évolution des admissions")
        fig_time = analyzer.plot_admissions_over_time()
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        st.subheader("Répartition par service")
        fig_service = analyzer.plot_service_distribution()
        st.plotly_chart(fig_service, use_container_width=True)
    
    # Périodes de pic
    st.markdown("---")
    st.subheader("🔴 Périodes de pic d'activité")
    peaks = analyzer.identify_peak_periods(threshold_percentile=90)
    st.dataframe(
        peaks[['date_admission', 'nb_admissions', 'nb_urgences']].head(10),
        use_container_width=True
    )


def show_admissions_analysis(analyzer):
    """Affiche l'analyse détaillée des admissions."""
    st.header("📈 Analyse des admissions")
    
    # Statistiques par service
    st.subheader("Statistiques par service")
    service_stats = analyzer.get_service_stats()
    st.dataframe(service_stats, use_container_width=True)
    
    # Statistiques quotidiennes
    st.markdown("---")
    st.subheader("Statistiques quotidiennes")
    daily_stats = analyzer.get_daily_stats()
    st.dataframe(daily_stats.tail(30), use_container_width=True)


def show_predictions(admissions):
    """Affiche les prédictions."""
    st.header("🔮 Prédictions")
    
    # Configuration
    col1, col2 = st.columns([1, 3])
    
    with col1:
        model_type = st.selectbox(
            "Type de modèle",
            ["random_forest", "gradient_boosting"],
            format_func=lambda x: "Random Forest" if x == "random_forest" else "Gradient Boosting"
        )
        
        n_days = st.slider("Nombre de jours à prédire", 7, 90, 30)
        
        train_button = st.button("Entraîner et prédire", type="primary")
    
    with col2:
        if train_button:
            with st.spinner("Entraînement du modèle..."):
                predictor = AdmissionPredictor(model_type=model_type)
                metrics = predictor.train(admissions)
                
                # Affichage des métriques
                st.success("Modèle entraîné avec succès!")
                met_col1, met_col2, met_col3 = st.columns(3)
                with met_col1:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with met_col2:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with met_col3:
                    st.metric("R²", f"{metrics['r2']:.3f}")
            
            # Prédictions
            future_dates = pd.date_range(
                start=pd.to_datetime(admissions['date_admission']).max() + pd.Timedelta(days=1),
                periods=n_days,
                freq='D'
            )
            
            predictions = predictor.predict(future_dates)
            
            # Affichage des prédictions
            st.markdown("---")
            st.subheader("Prédictions futures")
            
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Admissions prévues': predictions.round().astype(int)
            })
            
            st.line_chart(pred_df.set_index('Date'))
            st.dataframe(pred_df, use_container_width=True)


def show_resources(resources):
    """Affiche les ressources."""
    st.header("🏥 Gestion des ressources")
    
    resources['date'] = pd.to_datetime(resources['date'])
    
    # Métriques actuelles (dernier jour)
    latest = resources.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Lits disponibles", int(latest['lits_disponibles']))
    
    with col2:
        st.metric("Infirmiers", int(latest['infirmiers']))
    
    with col3:
        st.metric("Médecins", int(latest['medecins']))
    
    with col4:
        st.metric("Taux d'occupation", f"{latest['taux_occupation']:.1%}")
    
    # Graphiques
    st.markdown("---")
    st.subheader("Évolution des ressources")
    
    # Graphique du taux d'occupation
    st.line_chart(resources.set_index('date')['taux_occupation'])
    
    # Tableau des dernières données
    st.subheader("Données récentes")
    st.dataframe(resources.tail(30), use_container_width=True)


if __name__ == "__main__":
    main()
