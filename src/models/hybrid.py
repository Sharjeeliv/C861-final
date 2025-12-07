import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SingleCNNBlock, EmbeddingModel
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

# ********************************
# HYBRID: CNN -> RNN
# ********************************
class Hybrid1(EmbeddingModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout)
        num_filters = hidden_dim
        # Input to CNN is E, Output is F
        self.cnn_block = SingleCNNBlock(embed_dim, num_filters)
        
        # Input to RNN is F, Output is H
        self.rnn = nn.GRU(num_filters, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, text, lengths, score=None):
        embedded = self.embedding(text)             # [B, T, E]
        x = self.cnn_block(embedded).transpose(1, 2) # [B, T, F]
        
        # Apply padding to the GRU input
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Get final hidden state
        _, h = self.rnn(packed)                      # h: [1, B, H]
        
        return self.fc(h.squeeze(0))
        


# ********************************
# HYBRID: RNN -> CNN
# ********************************
class Hybrid2(EmbeddingModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout)
        num_filters = hidden_dim
        # Input to RNN is E, Output is H
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # Input to CNN is H, Output is F
        self.cnn_block = SingleCNNBlock(hidden_dim, num_filters)

        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, text, lengths, score=None):
        embedded = self.embedding(text)  # [B, T, E]
        
        # Apply packing to the GRU input
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True) # rnn_out: [B, T, H]

        # Process sequence output with CNN
        cnn_out = self.cnn_block(rnn_out)  # [B, F, T]
        
        # Global Max Pooling
        pooled = torch.max(cnn_out, dim=2).values
        return self.fc(pooled)

# ********************************
# HYBRID: CNN + RNN
# ********************************
class Hybrid3(EmbeddingModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout)
        num_filters = hidden_dim
        # CNN branch
        self.cnn_block = SingleCNNBlock(embed_dim, num_filters)
        
        # RNN branch
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # Combined classifier
        self.fc = nn.Linear(hidden_dim + num_filters, num_classes)

    def forward(self, text, lengths, score=None):
        x = self.embedding(text)  # [B, T, E]

        # --- CNN branch (No packing needed here) ---
        cnn_x = self.cnn_block(x) # [B, F, T]
        cnn_vec = torch.max(cnn_x, dim=2).values  # [B, F]

        # --- RNN branch (Packing is crucial here) ---
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        rnn_vec = h.squeeze(0) # [B, H]

        # Concatenate and Classify
        combined = torch.cat([cnn_vec, rnn_vec], dim=1)
        return self.fc(combined)


# ********************************
# HYBRID: LSTM -> CNN
# ********************************
class Hybrid4(EmbeddingModel):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout)
        num_filters = hidden_dim
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # CNN input channel must be 2 * hidden_dim
        self.cnn_block = SingleCNNBlock(2 * hidden_dim, num_filters)
        
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, text, lengths, score=None):
        x = self.embedding(text) 
        
        # 1. Pack
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False) 

        # 2. Process
        packed_lstm_out, _ = self.bilstm(packed_x) 

        # 3. Unpack (for the CNN layer)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True) # [B, T, 2H]

        # Process with CNN
        cnn_out = self.cnn_block(lstm_out)  # [B, F, T]
        
        # Global Max Pool
        pooled = torch.max(cnn_out, dim=2).values
        return self.fc(pooled)
