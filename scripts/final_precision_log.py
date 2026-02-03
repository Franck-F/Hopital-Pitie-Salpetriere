import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Load
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

# Feature Engineering
df = pd.DataFrame(index=dd.index)
df['admissions'] = dd.values
df['y_log'] = np.log1p(df['admissions'])

df['day'] = df.index.dayofweek
df['month'] = df.index.month
for l in [1, 2, 7, 14, 21, 28]:
    df[f'lag_{l}'] = df['y_log'].shift(l)

df = df.dropna()
X = df.drop(['admissions', 'y_log'], axis=1)
y_log = df['y_log']
y_real = df['admissions']

X_tr, X_te = X.iloc[:-30], X.iloc[-30:]
y_tr_log, y_te_log = y_log.iloc[:-30], y_log.iloc[-30:]
y_real_te = y_real.iloc[-30:]

# Training on Log Space
model = lgb.LGBMRegressor(objective='regression_l1', n_estimators=5000, learning_rate=0.005, num_leaves=127, verbose=-1, random_state=42)
model.fit(X_tr, y_tr_log)

# Inverse Transform
preds_log = model.predict(X_te)
preds_real = np.expm1(preds_log)

mae = mean_absolute_error(y_real_te, preds_real)
print(f"LOG_TRANSFORMED_MAE: {mae:.2f}")
