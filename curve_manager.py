import json
import os
import numpy as np

CURVES_DIR = os.path.join(os.path.dirname(__file__), 'curves')


def save_curve(name: str, freqs: np.ndarray, db: np.ndarray) -> str:
    os.makedirs(CURVES_DIR, exist_ok=True)
    path = os.path.join(CURVES_DIR, f'{name}.json')
    with open(path, 'w') as f:
        json.dump({'freqs': freqs.tolist(), 'db': db.tolist()}, f)
    return path


def load_curve(name: str):
    path = os.path.join(CURVES_DIR, f'{name}.json')
    with open(path) as f:
        data = json.load(f)
    return np.array(data['freqs']), np.array(data['db'])


def list_curves():
    if not os.path.isdir(CURVES_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(CURVES_DIR) if f.endswith('.json'))
