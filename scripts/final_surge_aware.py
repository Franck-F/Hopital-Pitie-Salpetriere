import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# 1. Chargement
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

df = pd.DataFrame(index=dd.index)
df['admissions'] = dd.values

# 2. Features cibles (Minimal lag = 7 jours max pour preserver Janvier)
df['day'] = df.index.dayofweek
df['month'] = df.index.month
df['lag1'] = df['admissions'].shift(1)
df['lag2'] = df['admissions'].shift(2)
df['lag7'] = df['admissions'].shift(7)

# Holiday context (Extrement important pour Decembre)
holidays = pd.to_datetime(['2024-01-01', '2024-04-01', '2024-05-01', '2024-05-08', 
                         '2024-05-09', '2024-05-20', '2024-07-14', '2024-08-15', 
                         '2024-11-01', '2024-11-11', '2024-12-25'])
df['is_holiday'] = df.index.isin(holidays).astype(int)
df['dist_holiday'] = [(holidays[holidays >= d].min() - d).days if any(holidays >= d) else 365 for d in df.index]

df = df.dropna()
X = df.drop('admissions', axis=1)
y = df['admissions']

X_tr, X_te = X.iloc[:-30], X.iloc[-30:]
y_tr, y_te = y.iloc[:-30], y.iloc[-30:]

# 3. LightGBM avec penalite pour la stabilite
model = lgb.LGBMRegressor(
    objective='regression_l1',
    n_estimators=3000,
    learning_rate=0.01,
    num_leaves=63,
    lambda_l1=1.0,
    lambda_l2=1.0,
    verbose=-1,
    random_state=42
)

model.fit(X_tr, y_tr)
p = model.predict(X_te)
mae = mean_absolute_error(y_te, p)

print(f"ULTIMATE_MAE_REACHED: {mae:.2f}")

# Sauvegarde Finale
import joblib
import os
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/lightgbm_final_v2.joblib')
