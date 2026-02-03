import json
import os

NOTEBOOK_PATH = "notebooks/LigthGBM.ipynb"

def clean_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Erreur : {NOTEBOOK_PATH} introuvable.")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Clean execution counts and outputs for ALL code cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"Sorties du notebook {NOTEBOOK_PATH} nettoyees.")

if __name__ == "__main__":
    clean_notebook()
