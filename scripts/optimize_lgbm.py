import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Data
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

# Features
df = pd.DataFrame(index=dd.index)
df['admissions'] = dd.values
for k in range(1, 4):
    df[f'f_sin_{k}'] = np.sin(2 * np.pi * k * df.index.dayofyear / 365.25)
    df[f'f_cos_{k}'] = np.cos(2 * np.pi * k * df.index.dayofyear / 365.25)
df['ewma_7'] = df['admissions'].shift(1).ewm(span=7).mean()
for l in [1, 2, 7, 14]:
    df[f'lag_{l}'] = df['admissions'].shift(l)
df['is_winter'] = df.index.month.isin([11, 12, 1, 2]).astype(int)
df['day'] = df.index.dayofweek

df = df.dropna()
X_train, X_test = df.drop('admissions', axis=1).iloc[:-90], df.drop('admissions', axis=1).iloc[-90:]
y_train, y_test = df['admissions'].iloc[:-90], df['admissions'].iloc[-90:]

# Tuning DART
rs = RandomizedSearchCV(
    lgb.LGBMRegressor(objective='regression_l1', random_state=42, verbose=-1),
    param_distributions={'num_leaves': [63, 127], 'learning_rate': [0.01, 0.05], 'n_estimators': [1000, 2000]},
    n_iter=10, cv=TimeSeriesSplit(n_splits=3), scoring='neg_mean_absolute_error', n_jobs=-1
)
rs.fit(X_train, y_train)
preds = rs.best_estimator_.predict(X_test)
print(f"ULTRA|MAE:{mean_absolute_error(y_test, preds):.2f}")
