import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_absolute_error

# Load data
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])

# Aggregate daily
daily_data = df_adm.groupby('date_entree').size().rename('admissions').reset_index()
daily_data = daily_data.set_index('date_entree').asfreq('D', fill_value=0)

# Feature engineering
def create_features(df):
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    df['lag1'] = df['admissions'].shift(1)
    df['lag7'] = df['admissions'].shift(7)
    df['lag14'] = df['admissions'].shift(14)
    
    df['roll_mean_7'] = df['admissions'].shift(1).rolling(window=7).mean()
    df['roll_std_7'] = df['admissions'].shift(1).rolling(window=7).std()
    
    return df

features_df = create_features(daily_data).dropna()

FEATURES = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 
            'weekofyear', 'lag1', 'lag7', 'lag14', 'roll_mean_7', 'roll_std_7']
TARGET = 'admissions'

X = features_df[FEATURES]
y = features_df[TARGET]

# Train model on full data for production
reg = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3, objective='reg:squarederror')
reg.fit(X, y)

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(reg, 'models/xgboost_admissions_v1.joblib')

print("Model trained and saved to models/xgboost_admissions_v1.joblib")
print(f"Final training MAE: {mean_absolute_error(y, reg.predict(X)):.2f}")
