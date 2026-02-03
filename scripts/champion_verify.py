import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Champion Setup
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

df = pd.DataFrame(index=dd.index)
df['admissions'] = dd.values
for k in range(1, 4):
    df[f'f_sin_{k}'] = np.sin(2 * np.pi * k * df.index.dayofyear / 365.25)
    df[f'f_cos_{k}'] = np.cos(2 * np.pi * k * df.index.dayofyear / 365.25)
df['day'] = df.index.dayofweek
for l in [1, 2, 7, 14]:
    df[f'lag_{l}'] = df['admissions'].shift(l)
df = df.dropna()

X = df.drop(columns=['admissions'])
y = df['admissions']
X_tr, X_te = X.iloc[:-30], X.iloc[-30:]
y_tr, y_te = y.iloc[:-30], y.iloc[-30:]

model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=5000, learning_rate=0.005, num_leaves=255, verbose=-1, random_state=42)
model.fit(X_tr, y_tr)
preds = model.predict(X_te)
print(f"CHAMPION_MAE: {mean_absolute_error(y_te, preds):.2f}")
