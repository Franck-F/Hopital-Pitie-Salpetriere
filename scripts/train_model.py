import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

# Load data
df_adm = pd.read_csv('data/raw/admissions_hopital_pitie_2024.csv')
df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])

# Aggregate daily
daily_data = df_adm.groupby('date_entree').size().rename('admissions').reset_index()
daily_data = daily_data.set_index('date_entree').asfreq('D', fill_value=0)

# Feature engineering
def create_features(df):
    df = df.copy()
    
    # Cyclical features for time
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    # Lags
    df['lag1'] = df['admissions'].shift(1)
    df['lag7'] = df['admissions'].shift(7)
    df['lag14'] = df['admissions'].shift(14)
    
    # Rolling stats
    df['roll_mean_7'] = df['admissions'].shift(1).rolling(window=7).mean()
    df['roll_std_7'] = df['admissions'].shift(1).rolling(window=7).std()
    
    return df

features_df = create_features(daily_data).dropna()

FEATURES = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'dayofyear', 
            'weekofyear', 'lag1', 'lag7', 'lag14', 'roll_mean_7', 'roll_std_7']
TARGET = 'admissions'

X = features_df[FEATURES]
y = features_df[TARGET]

# 1. Linear Model for Seasonal Scale & Trend
# Captures the general "heat" and seasonal baseline
reg_lr = LinearRegression()
reg_lr.fit(X, y)
y_lr = reg_lr.predict(X)

# 2. XGBoost for Residuals (Optimization with GridSearchCV)
residuals = y - y_lr

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', base_score=0)

param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X, residuals)
reg_xgb = grid_search.best_estimator_

print(f"Best Parameters found: {grid_search.best_params_}")

# Save Hybrid Bundle
os.makedirs('models', exist_ok=True)
joblib.dump({'lr': reg_lr, 'xgb': reg_xgb}, 'models/hybrid_admissions_v1.joblib')

print("Optimized Hybrid Model trained and saved to models/hybrid_admissions_v1.joblib")
mae_lr = mean_absolute_error(y, y_lr)
mae_hybrid = mean_absolute_error(y, y_lr + reg_xgb.predict(X))
print(f"Linear Baseline MAE: {mae_lr:.2f}")
print(f"Optimized Hybrid MAE: {mae_hybrid:.2f}")
