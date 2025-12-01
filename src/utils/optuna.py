from optuna.trial import Trial
from pathlib import Path
import json

# Code to initialize params dictionary
ROOT = Path(__file__).parent.parent.parent
params = json.load(open(ROOT / 'config' / 'params.json'))


def trial_type(trial: Trial, name: str, config: dict):
    TYPE, START, END = 0, 1, 2
    ptype = config[TYPE]
    if ptype == 'log':     return trial.suggest_float(name, config[START], config[END], log=True)
    if ptype == 'int':     return trial.suggest_int(name, config[START], config[END])
    if ptype == 'flt':     return trial.suggest_float(name, config[START], config[END])
    if ptype == 'cat':     return trial.suggest_categorical(name, config[START:])
    raise ValueError(f"Unknown trial type: {ptype}")


def get_trial_params(trial: Trial, model_name: str):
    model_params = {}
    for param_name, param_range in params[model_name].items():
        suggest_fn = trial_type(trial, param_name, param_range)
        model_params[param_name] = suggest_fn
    return model_params
