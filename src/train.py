import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from .data.encode import seq_encode
from .data.data import get_datasets
from .models import BasicRNN, BasicLSTM, TextCNN


# ********************************
# VARIABLES AND SETUP
# ********************************
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NOTE: This needs to be automated with optuna
# Vocab Metadata
VOCAB_PARAMS = {
    'VOCAB_SIZE': 16860,
    'PAD_IDX': 1,
    'NUM_CLASSES': 3,
}
# Hyperparameters
MODEL_HYPERPARAMS = {
    'EMBED_DIM': 100,
    'HIDDEN_DIM': 128, # Used as hidden size for RNN/LSTM and filter count for CNN
    'N_EPOCHS': 5,     # Quick MVP run
    'LR': 1e-3
}
MODElS = {
    "GRU RNN": BasicRNN,
    "LSTM": BasicLSTM,
    "Text CNN": TextCNN,
}

# ********************************
# HELPER FUNCTIONS
# ********************************
def _metrics(all_labels, all_preds):
    a = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, average='macro')
    r = recall_score(all_labels, all_preds, average='macro')
    f = f1_score(all_labels, all_preds, average='macro')
    return a, p, r, f
    

def _train(model, dataloader, criterion, optimizer, device):
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


def _evaluate(model, dataloader, criterion, device):
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


def _loop(model, criterion, optimizer, tr_loader, te_loader, hyperparams):
    for epoch in range(1, hyperparams['N_EPOCHS'] + 1):
        tr_loss = _train(model, tr_loader, criterion, optimizer, device)
        l, a, p, r, f = _evaluate(model, te_loader, criterion, device)
        print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} | Test Loss: {l:.4f} | Accuracy: {a:.4f}")
    return a, p, r, f


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
    a, p, r, f = _loop(model, criterion, optimizer, tr_loader, te_loader, hyperparams)
    print(f"Accuracy: {a:.4} ({model_name})")
    return a

# ********************************
# INTERFACE FUNCTIONS
# ********************************
def experiments():
    # Load and process data
    X_tr, y_tr, X_te, y_te = get_datasets()
    tr_loader, te_loader = seq_encode(X_tr, y_tr, X_te, y_te)
    
    # Run experiment per model
    results = {}
    for name, Model in MODElS.items():
        results[name] = run_experiment(name, Model, tr_loader, te_loader, MODEL_HYPERPARAMS, VOCAB_PARAMS)
    
    # Print summary results
    print("\n\nSummary Results (Accuracy)")
    for model, a in results.items(): print(f"{model}: {a:.4f}")


if __name__ == '__main__':
    experiments()