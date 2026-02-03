import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Chargement et Preparation
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])
dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)

df = pd.DataFrame(index=dd.index)
df['admissions'] = dd.values
df['day'] = df.index.dayofweek
df['month'] = df.index.month

# 2. Features de Base (Efficaces)
for l in [1, 2, 7, 14, 21, 28]:
    df[f'lag_{l}'] = df['admissions'].shift(l)
for w in [7, 14, 28]:
    df[f'mean_{w}'] = df['admissions'].shift(1).rolling(window=w).mean()

df = df.dropna()
X = df.drop('admissions', axis=1)
y = df['admissions']

# 3. Split : Train (Jusqu'a Oct) | Val (Nov) | Test (Dec)
X_train_full = X.iloc[:-60]
y_train_full = y.iloc[:-60]
X_val = X.iloc[-60:-30]
y_val = y.iloc[-60:-30]
X_test = X.iloc[-30:]
y_test = y.iloc[-30:]

# 4. Modele High-Capacity avec Early Stopping
model = lgb.LGBMRegressor(
    objective='regression_l1',
    n_estimators=10000,
    learning_rate=0.002, # Tres lent pour la precision
    num_leaves=255,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    verbose=-1
)

print("Entrainement avec Early Stopping sur Novembre...")
model.fit(
    X_train_full, y_train_full,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

# 5. Evaluation
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"RESULTAT FINAL MAE (Decembre) : {mae:.2f}")

# Sauvegarder si meilleur
if mae < 70:
    import joblib
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/lightgbm_final_v1.joblib')
    print("Modele sauvegarde dans models/lightgbm_final_v1.joblib")
