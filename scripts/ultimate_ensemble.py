import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# 1. Chargement Clean
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

# 2. Feature Engineering Multi-Signal
def create_ultimate_features(df_ts):
    df = pd.DataFrame(index=df_ts.index)
    df['admissions'] = df_ts.values
    # Fourier
    for k in range(1, 4):
        df[f'f_sin_{k}'] = np.sin(2 * np.pi * k * df.index.dayofyear / 365.25)
        df[f'f_cos_{k}'] = np.cos(2 * np.pi * k * df.index.dayofyear / 365.25)
    # Temporal
    df['day'] = df.index.dayofweek
    df['month_s'] = np.sin(2 * np.pi * df.index.month / 12)
    # Lags
    for l in [1, 2, 7, 14]:
        df[f'lag_{l}'] = df['admissions'].shift(l)
    return df.dropna()

full_df = create_ultimate_features(dd)
X = full_df.drop('admissions', axis=1)
y = full_df['admissions']

X_tr, X_te = X.iloc[:-30], X.iloc[-30:]
y_tr, y_te = y.iloc[:-30], y.iloc[-30:]

# 3. Model A : LightGBM DART
m1 = lgb.LGBMRegressor(objective='regression_l1', n_estimators=3000, learning_rate=0.01, num_leaves=127, verbose=-1, random_state=42)
m1.fit(X_tr, y_tr)
p1 = m1.predict(X_te)

# 4. Model B : XGBoost
m2 = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=1000, learning_rate=0.01, max_depth=10, random_state=42)
m2.fit(X_tr, y_tr)
p2 = m2.predict(X_te)

# 5. Blend 50/50
p_final = (0.5 * p1) + (0.5 * p2)
mae_final = mean_absolute_error(y_te, p_final)

print(f"ULTIMATE_MAE: {mae_final:.2f}")
