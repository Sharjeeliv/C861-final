import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from .data.encode import seq_encode
from .data.data import get_datasets, get_val_split
from .models import BasicRNN, BasicLSTM, TextCNN


import optuna
from optuna import Trial


# ********************************
# VARIABLES AND SETUP
# ********************************
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 5

# NOTE: This needs to be automated with optuna
# Vocab Metadata
VOCAB_PARAMS = {
    'VOCAB_SIZE': 16860,
    'PAD_IDX': 1,
    'NUM_CLASSES': 3,
}
# # Hyperparameters
# MODEL_HYPERPARAMS = {
#     'EMBED_DIM': 100,
#     'HIDDEN_DIM': 128, # Used as hidden size for RNN/LSTM and filter count for CNN
#     'LR': 1e-3
# }
MODELS = {
    "RNN": BasicRNN,
    "LSTM": BasicLSTM,
    "CNN": TextCNN,
}

# ********************************
# HELPER FUNCTIONS
# ********************************
def _metrics(all_labels, all_preds):
    a = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return a, p, r, f
    

def _train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for labels, text in dataloader:
        labels, text = labels.to(device), text.to(device)
        optimizer.zero_grad()
        
        predicted_labels = model(text)
        loss = criterion(predicted_labels, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
    
    return total_loss / len(dataloader.dataset)


def _evaluate(model, dataloader, criterion, val=False):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    
    with torch.no_grad():
        for labels, text in dataloader:
            labels, text = labels.to(device), text.to(device)
            
            predicted_labels = model(text)
            loss = criterion(predicted_labels, labels)
            
            total_loss += loss.item() * labels.size(0)
            _, predicted_classes = torch.max(predicted_labels, 1)
            
            all_preds.extend(predicted_classes.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            
    # Compute Metrics
    test_loss = total_loss / len(dataloader.dataset)
    a, p, r, f = _metrics(all_labels, all_preds)
    
    return test_loss, a, p, r, f


def _loop(model, criterion, optimizer, tr_loader, te_loader):
    for epoch in range(1, EPOCHS + 1):
        tr_loss = _train(model, tr_loader, criterion, optimizer)
        l, a, p, r, f = _evaluate(model, te_loader, criterion)
        print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} | Test Loss: {l:.4f} | Accuracy: {a:.4f}")
    return a, p, r, f



# ********************************
# HYPERPARAMETER FUNCTIONS
# ********************************

from time import time
from .utils.optuna import get_trial_params
N_TRIALS = 15
N_CLS = 3
PAD_IDX = 1
VOCAB_SIZE = 16860

NON_MODEL_PARAMS = {'lr', 'optimizer', 'weight_decay'}

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

from pathlib import Path
import json
P_ROOT = Path.cwd()
P_PARAM = P_ROOT / 'config' / 'vocab.json' 

def _objective(trial: Trial, model_name: str, tr_loader: DataLoader, val_loader: DataLoader):
    # Guard Clause
    err_string = f"Model {model_name} not found in models dictionary."
    if model_name not in MODELS: raise ValueError(err_string)
    

    vocab_params = json.load(open(P_PARAM))
    
    # Retrieve and instantiate model
    trial_params = get_trial_params(trial, model_name)
    # Omit invalid configurations:
    try: model = get_model(model_name, trial_params, vocab_params, val=True)
    except RuntimeError: raise optuna.exceptions.TrialPruned()
    model.to(device)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, trial_params)

    # Epoch Loop:
    total_time = 0.0
    for epoch in range(EPOCHS):
        start_time = time()
        # Training & Validation
        # print(f"E={epoch + 1}", end="\t")
        _train(model, tr_loader, criterion, optimizer)
        l, a, _, _, _ = _evaluate(model, val_loader, criterion, val=True)
        # Timing and reporting
        epoch_time = time() - start_time
        total_time += epoch_time
        print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {l:.4f}, Val Acc: {a*100:.2f}%, Time: {epoch_time:.2f}s")

        # Prune (i.e., early stopping) based on validation loss
        trial.report(l, epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
    print(f"Total Training Time: {total_time:.2f}s")
    return l

def _tune(model_name: str, tr_loader: DataLoader, val_loader: DataLoader):
    # Hyperparameter Tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: _objective(trial, model_name, tr_loader, val_loader),  n_trials=N_TRIALS)
    return study.best_params


def _test(model_name, model_params, tr_loader: DataLoader, te_loader: DataLoader):
    
    vocab_params = json.load(open(P_PARAM))
    model = get_model(model_name, model_params, vocab_params)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, model_params)
    _loop(model, criterion,  optimizer, tr_loader, te_loader)
    return model

# ********************************
# INTERFACE FUNCTIONS
# ********************************


def run_experiment(model_name, ModelClass, tr_loader, te_loader, hyperparams, vocab_params):
    # Initialize, train, and test a model
    print(f"Model: {model_name}")
    
    # Initialize Model
    # Note: CNN reuses HIDDEN_DIM as num_filters for the __init__ call
    model = ModelClass(
        vocab_params['VOCAB_SIZE'], 
        hyperparams['EMBED_DIM'], 
        hyperparams['HIDDEN_DIM'], 
        vocab_params['NUM_CLASSES'], 
        vocab_params['PAD_IDX']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['LR'])
    
    # Training Loop
    a, p, r, f = _loop(model, criterion, optimizer, tr_loader, te_loader)
    print(f"Accuracy: {a:.4} ({model_name})")
    return a

# ********************************
# INTERFACE FUNCTIONS
# ********************************
# def experiments():
#     # Load and process data
#     X_tr, y_tr, X_te, y_te = get_datasets()
#     tr_loader, te_loader = seq_encode(X_tr, y_tr, X_te, y_te)
    
#     # Run experiment per model
#     results = {}
#     for name, Model in MODELS.items():
#         results[name] = run_experiment(name, Model, tr_loader, te_loader, MODEL_HYPERPARAMS, VOCAB_PARAMS)
    
#     # Print summary results
#     print("\n\nSummary Results (Accuracy)")
#     for model, a in results.items(): print(f"{model}: {a:.4f}")


if __name__ == '__main__':
    # experiments()
    X_tr, y_tr, X_te, y_te = get_datasets()
    X_tn, X_va, y_tn, y_va = get_val_split(X_tr, y_tr)
    
    # print('train-full')
    # print(y_tr)
    
    # print('train-part')
    # print(y_tn)
    tr_loader, te_loader = seq_encode(X_tr, y_tr, X_te, y_te)
    tn_loader, va_loader = seq_encode(X_tn, y_tn, X_va, y_va, val=True)
    
    
    
    # import pandas as pd
    # # Assuming y_tr_new is a pandas Series

    # print("--- Training Set Class Counts ---")
    # # 1. Get raw counts
    # print(y_tn.value_counts()) 

    # print("\n--- Training Set Class Percentages ---")
    # # 2. Get normalized percentages (easier to identify severe imbalance)
    # # The result will sum to 1.0 (or 100%)
    # print(y_tn.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
    
    MODEL_NAME = 'RNN'
    print('Tuning')
    params = _tune(MODEL_NAME, tn_loader, va_loader)
    print('Testing')
    model = _test(MODEL_NAME, params, tr_loader, te_loader)