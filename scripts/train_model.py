import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.metrics import mean_absolute_error
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
    
    # Holiday feature (Hardcoded French 2024 for simplicity)
    holidays = ['2024-01-01', '2024-04-01', '2024-05-01', '2024-05-08', 
                '2024-05-09', '2024-05-20', '2024-07-14', '2024-08-15', 
                '2024-11-01', '2024-11-11', '2024-12-25']
    df['is_holiday'] = df.index.strftime('%Y-%m-%d').isin(holidays).astype(int)
    
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

FEATURES = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'is_holiday', 'dayofyear', 
            'weekofyear', 'lag1', 'lag7', 'lag14', 'roll_mean_7', 'roll_std_7']
TARGET = 'admissions'

X = features_df[FEATURES]
y = features_df[TARGET]

# XGBoost Optimization with GridSearchCV
# Poisson objective for counts and more depth
xgb_model = xgb.XGBRegressor(objective='count:poisson', random_state=42)

param_grid = {
    'n_estimators': [1000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [6, 10],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0.1]
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

grid_search.fit(X, y)
best_xgb = grid_search.best_estimator_

print(f"Best Parameters found: {grid_search.best_params_}")

# Save Optimized Model
os.makedirs('models', exist_ok=True)
joblib.dump(best_xgb, 'models/xgboost_optimized_v1.joblib')

print("Optimized XGBoost Model saved to models/xgboost_optimized_v1.joblib")
final_mae = mean_absolute_error(y, best_xgb.predict(X))
print(f"Final Optimization MAE: {final_mae:.2f}")
