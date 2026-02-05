import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import base64
from datetime import timedelta
from config import DATA_ADMISSION_PATH


# Utilitaire pour charger image en base64 
def get_base64_image(path):
    try:
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return ""

# Chargement des Donnees 
@st.cache_data
def get_admission_data():
    df = pd.read_csv(DATA_ADMISSION_PATH)
    df['date_entree'] = pd.to_datetime(df['date_entree'])
    # Extraction features pour analyse temporelle
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
    
    # Calcul des durees
    df_sej['duree_jours'] = (df_sej['date_sortie'] - df_sej['date_admission']).dt.days
    df_sej['mois_adm'] = df_sej['date_admission'].dt.month_name()
    df_sej['jour_adm'] = df_sej['date_admission'].dt.day_name()
    
    return df_pat, df_sej, df_diag

@st.cache_resource
def load_champion_model():
    m_path = "models/lightgbm_final_v6_2425.joblib"
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
    
    holidays = pd.to_datetime(['2024-01-01', '2024-05-01', '2024-07-14', '2024-12-25',
                               '2025-01-01', '2025-05-01', '2025-07-14', '2025-12-25'])
    
    for i in range(1, days + 1):
        next_date = last_date + timedelta(days=i)
        future_dates.append(next_date)
        
        # Reconstruction Avancee des Features
        row = pd.DataFrame(index=[next_date])
        
        # Decalages (Lags)
        for l in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
            row[f'lag{l}'] = current_ts.iloc[-l] if len(current_ts) >= l else current_ts.iloc[-1]
            
        # Moyennes Glissantes
        for w in [3, 7]:
            row[f'roll_{w}'] = current_ts.iloc[-w:].mean() if len(current_ts) >= w else current_ts.iloc[-1]
            
        # Calendrier
        row['day'] = next_date.dayofweek
        row['month'] = next_date.month
        
        # Saisonnier (Cyclique)
        row['sin_day'] = np.sin(2 * np.pi * next_date.dayofyear / 365.25)
        row['cos_day'] = np.cos(2 * np.pi * next_date.dayofyear / 365.25)
        
        # Jours Feries
        row['is_holiday'] = 1 if next_date in holidays else 0
        
        FEATS = ['lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'lag6', 'lag7', 'lag14', 'lag21', 'lag28', 
                 'roll_3', 'roll_7', 'day', 'month', 'sin_day', 'cos_day', 'is_holiday']
        
        X_row = row[FEATS]
        final_pred = model.predict(X_row)[0]
        
        preds.append(final_pred)
        current_ts.loc[next_date] = final_pred
        
    return future_dates, np.array(preds)

def create_features_vectorized(df_ts):
    # Miroir de la Logique Notebook pour Evaluation
    df = pd.DataFrame(index=df_ts.index)
    df['admissions'] = df_ts.values
    
    for l in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
        df[f'lag{l}'] = df['admissions'].shift(l)
        
    for w in [3, 7]:
        df[f'roll_{w}'] = df['admissions'].shift(1).rolling(w).mean()
        
    df['day'] = df.index.dayofweek
    df['month'] = df.index.month
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    holidays = pd.to_datetime(['2024-01-01', '2024-05-01', '2024-07-14', '2024-12-25',
                               '2025-01-01', '2025-05-01', '2025-07-14', '2025-12-25'])
    df['is_holiday'] = df.index.isin(holidays).astype(int)
    return df.dropna()
