import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_PATH = "data/raw/admissions_hopital_pitie_2024_2025.csv"
MODEL_PATH = "models/lightgbm_final_v3_2425.joblib"

def train_v3():
    print("Chargement des donnees 2024-2025...")
    df_adm = pd.read_csv(DATA_PATH)
    df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
    dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)
    
    print("Feature Engineering Avance...")
    df = pd.DataFrame(index=dd.index)
    df['admissions'] = dd.values
    
    # 1. Lags
    for l in [1, 2, 7, 14]:
        df[f'lag{l}'] = df['admissions'].shift(l)
        
    # 2. Rolling Statistics
    for w in [7, 14]:
        df[f'roll_mean{w}'] = df['admissions'].shift(1).rolling(window=w).mean()
        
    # 3. Calendar Features
    df['day'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # 4. Seasonal Features (Cyclic)
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 5. Holidays (Simplifie pour 2024-2025)
    holidays = pd.to_datetime(['2024-01-01', '2024-05-01', '2024-07-14', '2024-12-25',
                               '2025-01-01', '2025-05-01', '2025-07-14', '2025-12-25'])
    df['is_holiday'] = df.index.isin(holidays).astype(int)
    
    df = df.dropna()
    
    X = df.drop(columns=['admissions'])
    y = df['admissions']
    
    # Split (30 derniers jours pour test)
    X_tr, y_tr = X.iloc[:-30], y.iloc[:-30]
    X_te, y_te = X.iloc[-30:], y.iloc[-30:]
    
    print(f"Entrainement LightGBM (X shape: {X_tr.shape})...")
    model = lgb.LGBMRegressor(
        objective='regression_l1',
        n_estimators=5000,
        learning_rate=0.005,
        num_leaves=63,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_tr, y_tr)
    
    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    print(f"MAE sur les 30 derniers jours : {mae:.2f}")
    
    print(f"Sauvegarde du modele : {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    
    # Sauvegarder la liste des features pour coherence dashboard
    print("Features utilisees :", X.columns.tolist())

if __name__ == "__main__":
    train_v3()
