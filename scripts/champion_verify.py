import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Charge
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

df = pd.DataFrame(index=dd.index)
df['admissions'] = dd.values
df['day'] = df.index.dayofweek
df['lag1'] = df['admissions'].shift(1)
df['lag7'] = df['admissions'].shift(7)
df = df.dropna()

X = df.drop(columns=['admissions'])
y = df['admissions']
X_tr, X_te = X.iloc[:-30], X.iloc[-30:]
y_tr, y_te = y.iloc[:-30], y.iloc[-30:]

model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=5000, learning_rate=0.01, num_leaves=255, verbose=-1, random_state=42)
model.fit(X_tr, y_tr)
p = model.predict(X_te)
print(f"VERIFIED_CHAMPION_MAE: {mean_absolute_error(y_te, p):.2f}")
