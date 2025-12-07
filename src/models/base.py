import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from .utils import EmbeddingModel



# ********************************
# BASE: CNN
# ********************************
class BaseCNN(EmbeddingModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout):
        
        num_filters = hidden_dim 
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout)
        
        # Unique layer: Parallel 1D Convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
            for k in [2, 3, 4] ])
        
        # FC layer input size: (number of kernels * number of filters)
        self.fc = nn.Linear(len(self.convs) * num_filters, num_classes)

    def forward(self, text, lengths):
        # text shape: [B, T]
        embedded = self.embedding(text) 
        # Permute for Conv1d: [B, E, T]
        embedded = embedded.permute(0, 2, 1) 
        
        # 1. Convolution and ReLU
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # 2. Global Max Pooling (dim=2)
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved]
        # 3. Concatenate all pooled features
        cat = torch.cat(pooled, dim=1) 
        # 4. Apply Dropout and Final FC layer
        out = self.dropout(cat)
        return self.fc(out)


# ********************************
# BASE: RNN
# ********************************
class BaseRNN(EmbeddingModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout)
        
        # Unique layer
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, lengths):
        embedded = self.embedding(text) # [B, T, E]
        
        # 1. Apply Packing (Crucial for RNNs) 
        packed = pack_padded_sequence(
            embedded, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # 2. Get final hidden state
        _, hidden = self.rnn(packed) 
        
        # hidden shape: [1, B, H]. Squeeze to [B, H]
        hidden_last = hidden.squeeze(0)

        # 3. Apply Dropout and Final FC layer
        out = self.dropout(hidden_last) 
        return self.fc(out)

# ********************************
# BASE: LSTM
# ********************************
class BaseLSTM(EmbeddingModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout)

        # Unique layer (note: recurrent dropout is often added here for stacked LSTMs)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        # FC layer input size must be 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, num_classes) 

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        
        # 1. Apply Packing
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # 2. Process with LSTM (Only capture final hidden state (h))
        # We discard the packed output and the cell state (c)
        _, (hidden, _) = self.lstm(packed) 
        
        # 3. Concatenate final forward (hidden[-2]) and backward (hidden[-1]) states
        hidden_last = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) 
                                           
        # 4. Apply Dropout and Final FC layer
        out = self.dropout(hidden_last)
        return self.fc(out)
