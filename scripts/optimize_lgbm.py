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
df['month'] = df.index.month
df['day'] = df.index.dayofweek
df['is_winter'] = df['month'].isin([11, 12, 1, 2]).astype(int)

for l in [1, 2, 7, 14]:
    df[f'lag_{l}'] = df['admissions'].shift(l)
for w in [7, 28]:
    df[f'mean_{w}'] = df['admissions'].shift(1).rolling(window=w).mean()
    df[f'std_{w}'] = df['admissions'].shift(1).rolling(window=w).std()

df = df.dropna()
X = df.drop('admissions', axis=1)
y = df['admissions']

# Test on last 90 days (Quarter 4)
test_days = 90
X_train, X_test = X.iloc[:-test_days], X.iloc[-test_days:]
y_train, y_test = y.iloc[:-test_days], y.iloc[-test_days:]

param_dist = {
    'num_leaves': [31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [1000, 2000],
    'feature_fraction': [0.8, 0.9]
}

tscv = TimeSeriesSplit(n_splits=4)
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

print(f"SEASONAL|MAE:{mae:.2f}|PARAMS:{rs.best_params_}")
