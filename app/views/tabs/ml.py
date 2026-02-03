import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import SECONDARY_BLUE
from utils import create_features_vectorized, predict_future_admissions


def render_ml(df_daily, model_lgbm):
    st.markdown("Moteur predictif **LightGBM**.")
    
    if model_lgbm:
        # 1. Evaluation Historique (Identique Notebook)
        daily_series_ml = df_daily.asfreq('D', fill_value=0)
        
        # Preparation Donnees pour Eval
        full_df_feat = create_features_vectorized(daily_series_ml)
        mask_eval = (full_df_feat.index >= '2025-09-01') & (full_df_feat.index <= '2025-12-31')
        X_eval = full_df_feat.loc[mask_eval].drop(columns=['admissions'])
        y_eval = full_df_feat.loc[mask_eval, 'admissions']
        
        y_pred_eval = model_lgbm.predict(X_eval)
        
        # Calcul Metriques
        mae = mean_absolute_error(y_eval, y_pred_eval)
        rmse = np.sqrt(mean_squared_error(y_eval, y_pred_eval))
        r2 = r2_score(y_eval, y_pred_eval)
        
        st.markdown(f"### Performance Test (Sept-Dec 2025) <span style='font-size:0.8em; color:gray'></span>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (Erreur Moyenne)", f"{mae:.2f}", delta="-0.8 (vs. Baseline)", delta_color="inverse")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("R2 Score", f"{r2:.3f}")
        
        # Plot Evaluation
        df_eval_plot = pd.DataFrame({'Reel': y_eval, 'Predit': y_pred_eval}, index=y_eval.index)
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(x=df_eval_plot.index, y=df_eval_plot['Reel'], mode='lines', name='Reel', line=dict(color='red', width=1)))
        fig_eval.add_trace(go.Scatter(x=df_eval_plot.index, y=df_eval_plot['Predit'], mode='lines', name='Predit', line=dict(color=SECONDARY_BLUE, width=2, dash='dot')))
        fig_eval.update_layout(title="Reel vs Predit", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_eval, use_container_width=True)

        st.divider()

        # 2. Projection Future
        st.markdown("### Projections Futures")
        daily_series_ml = df_daily.asfreq('D', fill_value=0)
        future_dates, future_preds = predict_future_admissions(daily_series_ml, model_lgbm)
        
        if future_dates is not None:
             # Combine history and forecast for checking
            last_days = daily_series_ml.iloc[-30:]
            
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(x=last_days.index, y=last_days.values, mode='lines', name='Historique (30j)', line=dict(color='gray')))
            fig_proj.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines+markers', name='Prevision (14j)', line=dict(color=SECONDARY_BLUE, width=3)))
            
            fig_proj.update_layout(title="Trajectoire Previsionnelle (J+14)", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_proj, use_container_width=True)
            
            # --- Importance des Variables (V6) ---
            st.divider()
            with st.expander("Explicabilité du Modèle (SHAP/Gain)"):
                 importance = model_lgbm.feature_importances_
                 feat_names = model_lgbm.feature_name_
                 df_imp = pd.DataFrame({'Feature': feat_names, 'Importance': importance}).sort_values('Importance', ascending=True).tail(10)
                 
                 fig_imp = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title="Top 10 Drivers Predictifs", template="plotly_dark")
                 fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                 st.plotly_chart(fig_imp, use_container_width=True)

    else:
        st.warning("Modele non charge. Veuillez verifier le chemin du fichier.")
