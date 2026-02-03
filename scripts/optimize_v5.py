import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import joblib

DATA_PATH = "data/raw/admissions_hopital_pitie_2024_2025.csv"

def optimize_v5():
    df_adm = pd.read_csv(DATA_PATH)
    df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
    dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)
    
    df = pd.DataFrame(index=dd.index)
    df['admissions'] = dd.values
    
    # --- V5 Features: The "Deep History" Strategy ---
    
    # 1. Short and Medium Lags
    for l in [1, 2, 3, 4, 5, 6, 7, 14, 21, 28]:
        df[f'lag{l}'] = df['admissions'].shift(l)
        
    # 2. Yearly Lag (The "Killer Feature")
    # 52 weeks ago = 364 days. Matches Day of Week.
    df['lag364'] = df['admissions'].shift(364)
    # 52 weeks +/- 1 week
    df['lag357'] = df['admissions'].shift(357)
    df['lag371'] = df['admissions'].shift(371)
    
    # 3. Rolling
    for w in [7, 14, 28]:
        df[f'roll_{w}'] = df['admissions'].shift(1).rolling(w).mean()
        
    # 4. Interactive / Special
    df['day'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    
    # 5. Seasonal
    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    
    holidays = pd.to_datetime(['2024-01-01', '2024-05-01', '2024-07-14', '2024-12-25',
                               '2025-01-01', '2025-05-01', '2025-07-14', '2025-12-25'])
    df['is_holiday'] = df.index.isin(holidays).astype(int)
    
    df = df.dropna()
    
    X = df.drop(columns=['admissions'])
    y = df['admissions']
    
    # Strict 2025 Split
    split_date = pd.Timestamp('2025-09-01')
    X_tr = X[X.index < split_date]
    y_tr = y[y.index < split_date]
    X_te = X[X.index >= split_date]
    y_te = y[y.index >= split_date]
    
    print(f"Train samples: {len(X_tr)} | Test samples: {len(X_te)}")
    
    # LightGBM Ultra-Tuned
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        n_estimators=10000,
        learning_rate=0.005,
        num_leaves=31,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        random_state=42
    )
    
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    
    print(f"MAE V5 (with Yearly Lags): {mae:.2f}")
    
    # Save if improved
    joblib.dump(model, "models/lightgbm_final_v5_2425.joblib")

if __name__ == "__main__":
    optimize_v5()
