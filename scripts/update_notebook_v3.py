import json
import os

NOTEBOOK_PATH = "notebooks/LigthGBM.ipynb"

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Erreur : {NOTEBOOK_PATH} introuvable.")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 1. Update Header
    header_content = [
        "# LightGBM - Vision 2024-2025 (V3 Champion)\n",
        "\n",
        "Ce notebook represente la configuration **V3 Champion** pour la Pitie-Salpetriere.\n",
        "Strategie :\n",
        "1. **Features Avancees** : Lags profonds (1, 2, 7, 14), Moyennes Mobiles (7, 14j) et Saisonnalite Cyclique (Sin/Cos).\n",
        "2. **Dataset Unifie** : Integration de la periode 2024-2025.\n",
        "3. **Optimisation** : Hyperparametres calibres pour une convergence precise sans overfitting.\n",
        "4. **Standard** : Absence totale d'emojis pour la robustesse."
    ]
    nb['cells'][0]['source'] = header_content

    # 2. Update Imports
    imports_content = [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import lightgbm as lgb\n",
        "from sklearn.metrics import mean_absolute_error\n",
        "import plotly.express as px\n",
        "import plotly.graph_objects as go\n",
        "import joblib\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')"
    ]
    nb['cells'][1]['source'] = imports_content

    # 3. Update Data Prep
    dataprep_content = [
        "# Chargement et Preparation 2024-2025\n",
        "df_adm = pd.read_csv('../data/raw/admissions_hopital_pitie_2024_2025.csv')\n",
        "df_adm['date_entree'] = pd.to_datetime(df_adm['date_entree'])\n",
        "dd = df_adm.groupby('date_entree').size().rename('admissions').asfreq('D', fill_value=0)\n",
        "\n",
        "def create_v3_features(df_ts):\n",
        "    df = pd.DataFrame(index=df_ts.index)\n",
        "    df['admissions'] = df_ts.values\n",
        "    \n",
        "    # 1. Lags\n",
        "    for l in [1, 2, 7, 14]:\n",
        "        df[f'lag{l}'] = df['admissions'].shift(l)\n",
        "        \n",
        "    # 2. Rolling Statistics\n",
        "    for w in [7, 14]:\n",
        "        df[f'roll_mean{w}'] = df['admissions'].shift(1).rolling(window=w).mean()\n",
        "        \n",
        "    # 3. Time components\n",
        "    df['day'] = df.index.dayofweek\n",
        "    df['month'] = df.index.month\n",
        "    \n",
        "    # 4. Seasonal (Cyclic)\n",
        "    df['sin_day'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)\n",
        "    df['cos_day'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)\n",
        "    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)\n",
        "    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)\n",
        "    \n",
        "    # 5. Holiday\n",
        "    holidays = pd.to_datetime(['2024-01-01', '2024-05-01', '2024-07-14', '2024-12-25',\n",
        "                               '2025-01-01', '2025-05-01', '2025-07-14', '2025-12-25'])\n",
        "    df['is_holiday'] = df.index.isin(holidays).astype(int)\n",
        "    \n",
        "    return df.dropna()\n",
        "\n",
        "full_df = create_v3_features(dd)\n",
        "X = full_df.drop(columns=['admissions'])\n",
        "y = full_df['admissions']\n",
        "\n",
        "X_tr, y_tr = X.iloc[:-30], y.iloc[:-30]\n",
        "X_te, y_te = X.iloc[-30:], y.iloc[-30:]"
    ]
    nb['cells'][2]['source'] = dataprep_content

    # 4. Update Training
    training_content = [
        "# Modele Champion V3 (Optimise pour 2024-2025)\n",
        "model = lgb.LGBMRegressor(\n",
        "    objective='regression_l1', \n",
        "    n_estimators=5000, \n",
        "    learning_rate=0.005, \n",
        "    num_leaves=63, \n",
        "    verbose=-1, \n",
        "    random_state=42\n",
        ")\n",
        "\n",
        "print(\"Entrainement du modele V3...\")\n",
        "model.fit(X_tr, y_tr)\n",
        "preds = model.predict(X_te)\n",
        "mae = mean_absolute_error(y_te, preds)\n",
        "\n",
        "print(f\"MAE SUR TEST 30 JOURS : {mae:.2f}\")\n",
        "\n",
        "print(\"Sauvegarde du modele...\")\n",
        "joblib.dump(model, '../models/lightgbm_final_v3_2425.joblib')\n",
        "print(\"Termine.\")"
    ]
    nb['cells'][3]['source'] = training_content

    # 5. Clear Outputs and Update Viz
    # Keeping the viz cell as is but clearing its execution count and outputs for a fresh look
    nb['cells'][4]['execution_count'] = None
    nb['cells'][4]['outputs'] = []

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Mise a jour de {NOTEBOOK_PATH} terminee avec succes.")

if __name__ == "__main__":
    update_notebook()
