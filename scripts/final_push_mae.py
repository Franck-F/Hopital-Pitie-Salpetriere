import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# 1. Chargement robuste
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

# 2. Feature Engineering Selectif (Evite l'overfitting)
df = pd.DataFrame(index=dd.index)
df['admissions'] = dd.values
df['day'] = df.index.dayofweek
df['lag1'] = df['admissions'].shift(1)
df['lag7'] = df['admissions'].shift(7)
df['lag14'] = df['admissions'].shift(14)
df['inter_1_7'] = df['lag1'] * df['lag7'] # Interaction cruciale

df = df.dropna()
X = df.drop('admissions', axis=1)
y = df['admissions']

# Split 30j
X_tr, X_te = X.iloc[:-30], X.iloc[-30:]
y_tr, y_te = y.iloc[:-30], y.iloc[-30:]

# 3. Modele A : LightGBM (MAE Optimized)
m_lgb = lgb.LGBMRegressor(objective='regression_l1', n_estimators=2000, learning_rate=0.01, num_leaves=63, verbose=-1, random_state=42)
m_lgb.fit(X_tr, y_tr)
p_lgb = m_lgb.predict(X_te)

# 4. Modele B : XGBoost (Variance Stabilizer)
m_xgb = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=1000, learning_rate=0.01, max_depth=8, random_state=42)
m_xgb.fit(X_tr, y_tr)
p_xgb = m_xgb.predict(X_te)

# 5. Hybrid Blending (70/30)
p_final = (0.7 * p_lgb) + (0.3 * p_xgb)
mae_final = mean_absolute_error(y_te, p_final)

print(f"HYBRID_MAE: {mae_final:.2f}")
print(f"LGBM_ONLY: {mean_absolute_error(y_te, p_lgb):.2f}")
print(f"XGB_ONLY: {mean_absolute_error(y_te, p_xgb):.2f}")
