import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Data Load
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

# Feature Engineering Stability
df = pd.DataFrame(index=dd.index)
df['admissions'] = dd.values

# Winsorization 99th
cap = np.percentile(df['admissions'], 99)
df['target_stable'] = np.where(df['admissions'] > cap, cap, df['admissions'])

df['day'] = df.index.dayofweek
df['month'] = df.index.month
for l in [1, 2, 7, 14, 21, 28]:
    df[f'lag_{l}'] = df['admissions'].shift(l)
for w in [7, 28]:
    df[f'roll_{w}'] = df['admissions'].shift(1).rolling(window=w).mean()

df = df.dropna()
X = df.drop(['admissions', 'target_stable'], axis=1)
y = df['target_stable']
y_real = df['admissions']

X_tr, X_te = X.iloc[:-30], X.iloc[-30:]
y_tr, y_te = y.iloc[:-30], y.iloc[-30:]
y_real_te = y_real.iloc[-30:]

# Optimization
rs = RandomizedSearchCV(
    lgb.LGBMRegressor(objective='regression_l1', random_state=42, verbose=-1),
    param_distributions={'num_leaves': [127, 255], 'learning_rate': [0.005, 0.01], 'n_estimators': [2000, 3000]},
    n_iter=5, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_absolute_error', n_jobs=-1
)

rs.fit(X_tr, y_tr)
preds = rs.best_estimator_.predict(X_te)
mae_final = mean_absolute_error(y_real_te, preds)
print(f"STABILITY_MAE: {mae_final:.2f}")
