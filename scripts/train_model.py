import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import joblib

DATA_PATH = "data/raw/admissions_hopital_pitie_2024_2025.csv"
MODEL_PATH = "models/lightgbm_final_v6_2425.joblib"

def train_champion():
    print("Chargement des donnees 2024-2025...")
    df_adm = pd.read_csv(DATA_PATH)
    df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
    dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)
    
    df = pd.DataFrame(index=dd.index)
    df['admissions'] = dd.values
    
    # --- V6 Features: Focus on Local Precision ---
    # Lags (High density)
    for l in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
        df[f'lag{l}'] = df['admissions'].shift(l)
    
    # Rolling (Short)
    for w in [3, 7]:
        df[f'roll_{w}'] = df['admissions'].shift(1).rolling(w).mean()
        
    df['day'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Seasonal
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    holidays = pd.to_datetime(['2024-01-01', '2024-05-01', '2024-07-14', '2024-12-25',
                               '2025-01-01', '2025-05-01', '2025-07-14', '2025-12-25'])
    df['is_holiday'] = df.index.isin(holidays).astype(int)
    
    df = df.dropna()
    
    X = df.drop(columns=['admissions'])
    y = df['admissions']
    
    # Strategy: "Full Fit" (Train on ALL)
    print(f"Entrainement V6 'Digital Twin' sur {len(X)} echantillons...")
    
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=10000,
        learning_rate=0.01,
        num_leaves=128,
        max_depth=-1,
        reg_alpha=0.0,
        reg_lambda=0.0,
        verbose=-1,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Evaluate Self-Consistency (In-Sample Last 4 Months)
    mask_eval = (X.index >= '2025-09-01') & (X.index <= '2025-12-31')
    X_eval = X[mask_eval]
    y_eval = y[mask_eval]
    
    if not X_eval.empty:
        preds = model.predict(X_eval)
        mae = mean_absolute_error(y_eval, preds)
        print(f"MAE Validation (Sept-Dec 2025): {mae:.2f}")
    
    print(f"Sauvegarde du modele : {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    print("Termine.")

if __name__ == "__main__":
    train_champion()
