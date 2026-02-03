import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Data Prep
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

df = pd.DataFrame(index=dd.index)
df['admissions'] = dd.values

# Advanced Features
df['day'] = df.index.dayofweek
df['is_weekend'] = (df.day >= 5).astype(int)
df['month'] = df.index.month
df['week'] = df.index.isocalendar().week.astype(int)

# Lags & Diffs
for l in [1, 2, 7, 14]:
    df[f'lag_{l}'] = df['admissions'].shift(l)
df['diff_1'] = df['lag_1'] - df['lag_2']
df['diff_7'] = df['lag_1'] - df['lag_7']

# Rolling Stats
for w in [7, 14]:
    df[f'mean_{w}'] = df['admissions'].shift(1).rolling(window=w).mean()
    df[f'std_{w}'] = df['admissions'].shift(1).rolling(window=w).std()

# Target encoding (categorical smooth)
df['day_avg'] = df.groupby('day')['admissions'].transform(lambda x: x.shift(1).expanding().mean())

df = df.dropna()
X = df.drop('admissions', axis=1)
y = df['admissions']

train_size = int(len(X) * 0.9)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Fast Optimization
param_dist = {
    'num_leaves': [31, 63, 150, 255],
    'learning_rate': [0.01, 0.05],
    'n_estimators': [1000, 2000],
    'min_child_samples': [5, 10, 20],
    'feature_fraction': [0.7, 0.9],
    'bagging_fraction': [0.7, 0.9],
    'bagging_freq': [5]
}

tscv = TimeSeriesSplit(n_splits=3)
rs = RandomizedSearchCV(
    lgb.LGBMRegressor(objective='regression_l1', random_state=42, verbose=-1),
    param_distributions=param_dist,
    n_iter=15,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

rs.fit(X_train, y_train)
preds = rs.best_estimator_.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"RES|MAE:{mae:.2f}|PARAMS:{rs.best_params_}")
