import torch
import torch.nn as nn
import torch.nn.functional as F


# ********************************
# BASIC RNN (GRU)
# ********************************
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


# ********************************
# BASIC LSTM
# ********************************
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


# ********************************
# BASIC 1D CNN
# ********************************
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