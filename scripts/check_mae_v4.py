import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import joblib

DATA_PATH = "data/raw/admissions_hopital_pitie_2024_2025.csv"
MODEL_PATH = "models/lightgbm_final_v4_2425.joblib"

def check_mae():
    print("Chargement des donnees...")
    df_adm = pd.read_csv(DATA_PATH)
    df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
    dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)
    
    df = pd.DataFrame(index=dd.index)
    df['admissions'] = dd.values
    
    for l in [1, 2, 7, 14]:
        df[f'lag{l}'] = df['admissions'].shift(l)
        
    for w in [7, 14]:
        df[f'roll_mean{w}'] = df['admissions'].shift(1).rolling(window=w).mean()
        
    df['day'] = df.index.dayofweek
    df['month'] = df.index.month
    
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    holidays = pd.to_datetime(['2024-01-01', '2024-05-01', '2024-07-14', '2024-12-25',
                               '2025-01-01', '2025-05-01', '2025-07-14', '2025-12-25'])
    df['is_holiday'] = df.index.isin(holidays).astype(int)
    
    df = df.dropna()
    
    X = df.drop(columns=['admissions'])
    y = df['admissions']
    
    # Split Strict (Test = 4 derniers mois 2025)
    split_date = pd.Timestamp('2025-09-01')
    X_te = X[X.index >= split_date]
    y_te = y[y.index >= split_date]
    
    print(f"Test Range: {X_te.index.min().date()} to {X_te.index.max().date()}")
    print(f"Chargement du modele V4 : {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    
    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    print(f"MAE sur Test (2025-09-01+): {mae:.2f}")

if __name__ == "__main__":
    check_mae()
