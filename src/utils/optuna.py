from optuna.trial import Trial
from pathlib import Path
from ..models import MODELS
import torch
import json

# Code to initialize params dictionary
ROOT = Path(__file__).parent.parent.parent
params = json.load(open(ROOT / 'config' / 'params.json'))
NON_MODEL_PARAMS = {'lr', 'optimizer', 'weight_decay'}


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


def get_model(model_name, trial_params, vocab_params, val=False):
    model_cls = MODELS[model_name]
    # Filter non-model params
    model_params = {k: v for k, v in trial_params.items() 
                    if k not in NON_MODEL_PARAMS}
    # Load vocab parameters
    n_classes = vocab_params['N_CLASSES']
    key = f"N_VOCAB_{'VA' if val else 'TE'}"
    n_vocab   = vocab_params[key]
    pad_idx   = vocab_params['PAD_IDX']
    return model_cls(**model_params, num_classes=n_classes, 
                     pad_idx=pad_idx, vocab_size=n_vocab)


def get_optimizer(model, trial_params, forced_lr=0):
    # Unpack parameters
    optimizer_type = trial_params.get("optimizer", "Adam")
    lr = trial_params.get("lr", 1e-3)
    weight_decay = trial_params.get("weight_decay", 0.0)
    
    # Build optimizer for remaining models
    if forced_lr !=0: lr = forced_lr
    if optimizer_type == "SGD":
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                lr=lr, weight_decay=weight_decay)
    else: raise ValueError(f"Unknown optimizer type: {optimizer_type}")
