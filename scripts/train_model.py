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
    
    # Holiday feature and proximity
    holidays = ['2024-01-01', '2024-04-01', '2024-05-01', '2024-05-08', 
                '2024-05-09', '2024-05-20', '2024-07-14', '2024-08-15', 
                '2024-11-01', '2024-11-11', '2024-12-25']
    holiday_dates = pd.to_datetime(holidays)
    df['is_holiday'] = df.index.strftime('%Y-%m-%d').isin(holidays).astype(int)
    
    # Distance to next/prev holiday
    df['days_to_holiday'] = [(holiday_dates[holiday_dates >= d].min() - d).days if any(holiday_dates >= d) else 365 for d in df.index]
    
    df['dayofyear'] = df.index.dayofyear
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['dayofmonth'] = df.index.day
    
    # Lags
    df['lag1'] = df['admissions'].shift(1)
    df['lag2'] = df['admissions'].shift(2)
    df['lag7'] = df['admissions'].shift(7)
    df['lag14'] = df['admissions'].shift(14)
    
    # Advanced Rolling stats
    df['roll_mean_3'] = df['admissions'].shift(1).rolling(window=3).mean()
    df['roll_mean_7'] = df['admissions'].shift(1).rolling(window=7).mean()
    df['roll_max_7'] = df['admissions'].shift(1).rolling(window=7).max()
    df['roll_min_7'] = df['admissions'].shift(1).rolling(window=7).min()
    df['roll_std_7'] = df['admissions'].shift(1).rolling(window=7).std()
    
    return df

features_df = create_features(daily_data).dropna()

FEATURES = ['month_sin', 'month_cos', 'day_sin', 'day_cos', 'is_holiday', 'days_to_holiday', 
            'dayofyear_sin', 'dayofyear_cos', 'weekofyear', 'dayofmonth', 'lag1', 'lag2', 'lag7', 'lag14', 
            'roll_mean_3', 'roll_mean_7', 'roll_max_7', 'roll_min_7', 'roll_std_7']
TARGET = 'admissions'

# 1. Formal Train/Test Split (Hold-out Test set = last 30 days)
train_df = features_df.iloc[:-30]
test_df = features_df.iloc[-30:]

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

# 2. XGBoost Optimization with TimeSeries Cross-Validation
# We optimize on the train set using internal validation folds
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [1000, 2000],
    'learning_rate': [0.01, 0.05],
    'max_depth': [10, 12],
    'subsample': [0.9],
    'gamma': [0.1]
}

# TimeSeriesSplit for Cross-Validation (standard for temporal data)
tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    verbose=1,
    n_jobs=-1
)

print("Starting Nested Cross-Validation...")
grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_

print(f"Best Parameters found: {grid_search.best_params_}")

# 3. Final Evaluation on Hold-out Test
y_pred = best_xgb.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)

# Save Optimized Model
os.makedirs('models', exist_ok=True)
joblib.dump(best_xgb, 'models/xgboost_optimized_v1.joblib')

print("Final Optimized XGBoost Model saved to models/xgboost_optimized_v1.joblib")
print(f"Final Cross-Val MAE (Best): {-grid_search.best_score_:.2f}")
print(f"Final Hold-out Test MAE (Dec): {test_mae:.2f}")
