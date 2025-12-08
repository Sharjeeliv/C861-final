# General utility methods
import time
from typing import Dict
import os
import json

import torch


# ********************************
# EARLY STOPPING IMPLEMENTATION
# ********************************
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-3):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter < self.patience: return
            self.early_stop = True

        else:
            self.best_score = score
            self.counter = 0


# ********************************
# HELPER FUNCTIONS
# ********************************
def time_execution(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        m, s = divmod(execution_time, 60)
        ms = (execution_time - int(execution_time)) * 1000
        print(f"\033[91;1mEXECUTION TIME: {int(m):02}:{int(s):02}.{int(ms):03}\033[0m")
        return result
    return wrapper


def save_output(res: Dict, params: Dict, title: str, root):
    """
    Save evaluation results and best hyperparameters to both a text file and a JSON file.

    Args:
        res (Dict): Dictionary containing evaluation metrics (accuracy, precision, etc.)
        params (Dict): Dictionary containing best hyperparameters
        name (str): Base file name (without extension)
    """
    # Verify base path
    path_text = root / 'results' / 'text'
    os.makedirs(path_text, exist_ok=True)
    
    path_json = root / 'results' / 'json'
    os.makedirs(path_json, exist_ok=True)

    # Text Output
    with open(path_text / f"{title}.txt", "w") as f:
        f.write("=== Evaluation Results ===\n")
        for k, v in res.items(): f.write(f"{k:<15}{v:.4f}\n")

        f.write("\n=== Best Hyperparameters ===\n")
        for k, v in params.items(): f.write(f"{k:<20}{v}\n")

    # JSON output
    output_data = {
        "Evaluation_Results": res,
        "Best_Hyperparameters": params
    }
    with open(path_json / f"{title}.json", "w") as jf:
        json.dump(output_data, jf, indent=4)

    print(f"Output saved to '{title}'.txt/.json")

