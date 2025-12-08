# Local
from .data.encode import seq_encode
from .data.data import get_datasets, get_val_split
from .models import MODELS
from .utils.optuna import get_trial_params, get_model, get_optimizer
from .utils.utils import EarlyStopping, save_output

# Builtin
from time import time
from pathlib import Path
import json

# External
import numpy as np
import optuna
from optuna import Trial

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss as CEL
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, accuracy_score, 
                             precision_score, recall_score)


# ********************************
# VARIABLES AND SETUP
# ********************************
P_ROOT = Path.cwd()
P_PARAM = P_ROOT / 'config' / 'vocab.json' 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_TRIALS = 10
TE_EPOCHS = 15
TR_EPOCHS = 5


# ********************************
# HELPER FUNCTIONS
# ********************************
def _metrics(all_labels, all_preds):
    a = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return float(a), float(p), float(r), float(f)

def _pack_metrics(a, p, r, f):
    return {
        'Accuracy':  a,
        'Precision': p,
        'Recall':    r,
        'F1Score':   f
    }
     

def _train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for labels, text, length in dataloader:
        labels, text, length = labels.to(device), text.to(device), length.to(device)
        optimizer.zero_grad()

        # Compute pred and cost
        predicted_labels = model(text, length)
        loss = criterion(predicted_labels, labels)
        # Backprop.
        loss.backward()
        optimizer.step()
        # Loss computation
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(dataloader.dataset)


def _evaluate(model, dataloader, criterion, val=False):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for labels, text, length in dataloader:
            labels, text, length = labels.to(device), text.to(device), length.to(device)
        
            # Compute pred and cost
            predicted_labels = model(text, length)
            loss = criterion(predicted_labels, labels)
            
            # Loss calc. and result storage
            total_loss += loss.item() * labels.size(0)
            _, predicted_classes = torch.max(predicted_labels, 1)
            all_preds.extend(predicted_classes.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    # Compute Metrics
    test_loss = total_loss / len(dataloader.dataset)
    a, p, r, f = _metrics(all_labels, all_preds)
    return test_loss, a, float(p), float(r), float(f)


def _loop(model: Module, criterion: CEL, optimizer: Optimizer, 
          tr_loader: DataLoader, te_loader: DataLoader, trial: Trial | None=None, 
          val=False, earlystop=False)->float | tuple[float, float, float, float]:
    
    EPOCHS = TR_EPOCHS if trial else TE_EPOCHS
    total_time = 0.0
    early_stopping = EarlyStopping()
    for epoch in range(1, EPOCHS + 1):
    
        start_time = time()
        # Training & Evaluation
        tr_loss = _train(model, tr_loader, criterion, optimizer)
        l, a, p, r, f = _evaluate(model, te_loader, criterion, val)
        
        # Timing & Printing
        epoch_time = time() - start_time
        total_time += epoch_time
        print(f"Epoch {epoch:02d} "
              f"| Train Loss: {tr_loss:.4f} "
              f"| {'val' if val else 'Test'} Loss: {l:.4f} "
              f"| Accuracy: {a:.4f} "
              f"| Time: {epoch_time:.2f}s")
        
        # Prune (i.e., early stopping) based on validation loss
        early_stopping(l)
        if earlystop and early_stopping.early_stop: 
            print(f"Early stop! Total Training Time: {total_time:.2f}s")
            break
        if not trial: continue
        if trial.should_prune(): raise optuna.TrialPruned()
    
    print(f"Total Training Time: {total_time:.2f}s")
    if trial: return l
    return a, p, r, f


# ********************************
# TUNE & TEST FUNCTIONS
# ********************************
def _objective(trial: Trial, model_name: str, tr_loader: DataLoader, val_loader: DataLoader):
    # Guard Clause
    err_string = f"Model {model_name} not found in models dictionary."
    if model_name not in MODELS: raise ValueError(err_string)

    # Retrieve parameters
    vocab_params = json.load(open(P_PARAM))
    trial_params = get_trial_params(trial, model_name)
    
    # Omit invalid configurations:
    try: model = get_model(model_name, trial_params, vocab_params, val=True)
    except RuntimeError: raise optuna.exceptions.TrialPruned()
    model.to(device)

    # Criterion and optimizer
    c = torch.nn.CrossEntropyLoss()
    o = get_optimizer(model, trial_params)
    # Train-val loop
    l =_loop(model, c, o, tr_loader, val_loader, trial, val=True)
    return l


# ********************************
# INTERFACE FUNCTIONS
# ********************************
def tune(model_name: str, tr_loader: DataLoader, val_loader: DataLoader):
    # Hyperparameter Tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: _objective(trial, model_name, tr_loader, val_loader),  n_trials=N_TRIALS)
    return study.best_params


def test(model_name: str, model_params, tr_loader: DataLoader, te_loader: DataLoader, es=True):
    # Model testing
    vocab_params = json.load(open(P_PARAM))
    model = get_model(model_name, model_params, vocab_params)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, model_params)
    a, p, r, f = _loop(model, criterion, optimizer, tr_loader, te_loader, earlystop=es) # type: ignore

    return model, _pack_metrics(a, p, r, f)


# ********************************
# EXPERIMENT HELPER FUNCTIONS
# ********************************
def extract_features(model, loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for y, text, length in loader:
            text, lengths = text.to(device), length.to(device)
            
            f = model.repr(text, lengths)
            features.append(f.cpu().numpy())
            labels.append(y.numpy())
    
    X = np.vstack(features)
    y = np.hstack(labels)
    return X, y


# ********************************
# MAIN FUNCTIONS
# ********************************
if __name__ == '__main__':
    RUN_ALL = False
    X_tr, y_tr, X_te, y_te = get_datasets(run_all=RUN_ALL)
    X_tn, X_va, y_tn, y_va = get_val_split(X_tr, y_tr)
    
    tr_loader, te_loader = seq_encode(X_tr, y_tr, X_te, y_te)
    tn_loader, va_loader = seq_encode(X_tn, y_tn, X_va, y_va, val=True)

    # print((X_tr.str.strip().str.len() == 0).sum())
    # print((X_te.str.strip().str.len() == 0).sum())
    # print((X_tn.str.strip().str.len() == 0).sum())
    # print((X_va.str.strip().str.len() == 0).sum())
    
    print(MODELS.keys())
    # TEST_MODELS = ['CNN', 'RNN', 'LSTM', 'H1', 'H2', 'H3', 'H4']
    TEST_MODELS = ['CNN']
    for model_name in TEST_MODELS:
        
        print(f'Tuning: {model_name}')
        params = tune(model_name, tn_loader, va_loader)
        print(f'Testing: {model_name}')
        model, res = test(model_name, params, tr_loader, te_loader)
        save_output(res, params, f"{model_name}", P_ROOT)