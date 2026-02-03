import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/raw/admissions_hopital_pitie_2024_2025.csv"

def analyze():
    df_adm = pd.read_csv(DATA_PATH)
    df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
    
    # 1. Global Daily Series
    global_ts = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)
    
    # Baseline: Naive Lag-7 (Week-to-week persistence)
    y_true = global_ts.iloc[-122:] # Sept-Dec 2025
    y_pred_naive = global_ts.shift(7).iloc[-122:]
    
    mae_naive = mean_absolute_error(y_true, y_pred_naive)
    print(f"Global Naive MAE (Lag 7): {mae_naive:.2f}")
    
    # 2. Check Variance
    std_dev = y_true.std()
    print(f"Global Std Dev (Test): {std_dev:.2f}")
    
    # 3. Analyze by Mode Entree
    print("\n--- Mode Entree Breakdown ---")
    modes = df_adm['mode_entree'].unique()
    for m in modes:
        sub_ts = df_adm[df_adm['mode_entree'] == m].groupby('date_entree').size().asfreq('D', fill_value=0)
        y_true_sub = sub_ts.iloc[-122:]
        y_pred_sub = sub_ts.shift(7).iloc[-122:]
        mae_sub = mean_absolute_error(y_true_sub, y_pred_sub)
        print(f"Mode '{m}' - Naive MAE: {mae_sub:.2f} (Mean: {y_true_sub.mean():.1f})")

if __name__ == "__main__":
    analyze()
