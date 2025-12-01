import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score, accuracy_score

from .data.encode import seq_encode
from .data.data import get_datasets

# NOTE: Assume seq_encode, the necessary Dataset/DataLoader setup,
# and the underlying vocab functions are defined/imported above.

# =================================================================
# I. MVP MODEL DEFINITIONS
# (These remain the same as they define the architectures)
# =================================================================

# --- A. Basic RNN (GRU) ---
class BasicRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super(BasicRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        _, hidden = self.rnn(embedded)
        hidden_last = hidden.squeeze(0) 
        return self.fc(hidden_last)

# --- B. LSTM Model ---
class BasicLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx):
        super(BasicLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)
        hidden_last = hidden.squeeze(0) 
        return self.fc(hidden_last)

# --- C. 1D CNN Model ---
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # We reuse hidden_dim as num_filters for simplicity in the MVP function call
        num_filters = hidden_dim 
        
        # Parallel 1D Convolutions for different n-gram sizes (2, 3, 4)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in [2, 3, 4] 
        ])
        
        self.fc = nn.Linear(len(self.convs) * num_filters, num_classes)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        # Permute to [batch_size, embed_dim, seq_len] for Conv1d
        embedded = self.embedding(text).permute(0, 2, 1) 
        
        # Apply convolution and then Max Pooling over the sequence dimension
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        
        # Max pool over the sequence dimension (dim=2)
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        
        # Concatenate all pooled features
        cat = torch.cat(pooled, dim=1)
        
        return self.fc(cat)

def train_model(model, dataloader, criterion, optimizer, device):
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

def evaluate_model(model, dataloader, criterion, device):
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
            
    # Metrics are calculated on the encoded integer classes (0, 1, 2, etc.)
    test_loss = total_loss / len(dataloader.dataset)
    mae = mean_absolute_error(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    acc = accuracy_score(all_labels, all_preds)
    
    return test_loss, mae, f1, acc

def run_experiment(model_name, ModelClass, tr_loader, te_loader, hyperparams, vocab_params):
    """Initializes, trains, and tests a single model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Running MVP: {model_name} on {device} ---")
    
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
    for epoch in range(1, hyperparams['N_EPOCHS'] + 1):
        train_loss = train_model(model, tr_loader, criterion, optimizer, device)
        test_loss, test_mae, test_f1, test_acc = evaluate_model(
            model, te_loader, criterion, device
        )
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test F1: {test_f1:.4f}")
    
    print(f"✅ {model_name} MVP Complete. Final Test Accuracy (Encoded): {test_acc:.4f}")
    return test_acc

# =================================================================
# III. EXECUTION BLOCK (Hooks up to your data processing)
# =================================================================

if __name__ == '__main__':
    # --- 1. MOCK DATA LOADING (REPLACE THIS WITH YOUR ACTUAL DATA CALLS) ---
    # Assume these variables are loaded here:
    # X_tr, y_tr, X_te, y_te = load_your_processed_data() 
    


    # --- 2. DATA PIPELINE EXECUTION (YOUR REQUIRED CALL) ---
    # Assuming seq_encode returns the vocab/encoding info as well
    X_tr, y_tr, X_te, y_te = get_datasets()
    tr_loader, te_loader = seq_encode(X_tr, y_tr, X_te, y_te)
    # tr_loader, te_loader, word_to_idx,  = seq_encode(X_tr, y_tr, X_te, y_te)
    
    # --- 3. HYPERPARAMETERS & VOCAB METADATA ---
    VOCAB_PARAMS = {
        'VOCAB_SIZE': 16860,
        'PAD_IDX': 1,
        'NUM_CLASSES': 3,
    }
    
    MODEL_HYPERPARAMS = {
        'EMBED_DIM': 100,
        'HIDDEN_DIM': 128, # Used as hidden size for RNN/LSTM and filter count for CNN
        'N_EPOCHS': 5,     # Quick MVP run
        'LR': 1e-3
    }
    
    # --- 4. RUN ALL MVPs ---
    models_to_run = {
        "GRU RNN": BasicRNN,
        "LSTM": BasicLSTM,
        "Text CNN": TextCNN,
    }

    results = {}
    for name, Model in models_to_run.items():
        results[name] = run_experiment(name, Model, tr_loader, te_loader, MODEL_HYPERPARAMS, VOCAB_PARAMS)
        
    print("\n\n=== FINAL MVP RESULTS (Weighted F1-Score) ===")
    for model, f1 in results.items():
        print(f"{model}: {f1:.4f}")