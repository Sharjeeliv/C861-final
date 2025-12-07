import torch
import torch.nn as nn
import torch.nn.functional as F


# ********************************
# CNN + RNN
# ********************************
class CNN_RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx,
                 num_filters=128, kernel_size=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.rnn = nn.GRU(num_filters, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, text, lengths):
        x = self.embedding(text)           # [B, T, E]
        x = x.transpose(1, 2)              # [B, E, T]
        x = self.relu(self.conv(x))        # [B, F, T]
        x = x.transpose(1, 2)              # [B, T, F]
        _, h = self.rnn(x)                 # h: [1, B, H]
        return self.fc(h.squeeze(0))


# ********************************
# RNN + CNN
# ********************************
class RNN_CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx,
                 num_filters=128, kernel_size=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.fc = nn.Linear(num_filters, num_classes)
        self.relu = nn.ReLU()

    def forward(self, text, lengths):
        x = self.embedding(text)       # [B, T, E]
        # _, h_seq = self.rnn(x)         # h_seq: [1, B, H]
        # h = h_seq.squeeze(0)           # [B, H]
        # h = h.unsqueeze(-1)            # [B, H, 1]

        # # replicate across time to pseudo-sequence
        # h = h.repeat(1, 1, text.size(1))  # [B, H, T]
        # feat = self.relu(self.conv(h))    # [B, F, T]
        # pooled = torch.max(feat, dim=2).values
        # return self.fc(pooled)
        rnn_out, _ = self.rnn(x) # rnn_out: [B, T, H]

        # Transpose for CNN
        cnn_in = rnn_out.transpose(1, 2) # [B, H, T]

        # Apply CNN and pooling (same as your current code)
        feat = self.relu(self.conv(cnn_in)) # [B, F, T]
        pooled = torch.max(feat, dim=2).values
        return self.fc(pooled)
    

# ********************************
# RNN + CNN
# ********************************
class Parallel_CNN_RNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx,
                 num_filters=128, kernel_size=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # CNN branch
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

        # RNN branch
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

        # Combined classifier
        self.fc = nn.Linear(hidden_dim + num_filters, num_classes)

    def forward(self, text, lengths):
        x = self.embedding(text)  # [B, T, E]

        # --- CNN branch ---
        cnn_x = x.transpose(1, 2)
        cnn_x = self.relu(self.conv(cnn_x))
        cnn_vec = torch.max(cnn_x, dim=2).values  # global max pool

        # --- RNN branch ---
        _, h = self.rnn(x)
        rnn_vec = h.squeeze(0)

        # Concatenate
        combined = torch.cat([cnn_vec, rnn_vec], dim=1)
        return self.fc(combined)


# ********************************
# RNN + CNN
# ********************************
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
class BiLSTM_CNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx,
                 num_filters=128, kernel_size=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.conv = nn.Conv1d(
            in_channels=2 * hidden_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()

        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, text, lengths):
        x = self.embedding(text) 

        # 1. Pack
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False) 

        # 2. Process
        packed_lstm_out, _ = self.bilstm(packed_x) 

        # 3. Unpack (for the CNN layer)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True) # lstm_out: [B, T, 2H]

        # lstm_out = lstm_out.transpose(1, 2)
        x = self.embedding(text)                  # [B, T, E]
        lstm_out, _ = self.bilstm(x)              # [B, T, 2H]
        lstm_out = lstm_out.transpose(1, 2)       # [B, 2H, T]
        cnn_out = self.relu(self.conv(lstm_out))  # [B, F, T]
        pooled = torch.max(cnn_out, dim=2).values
        return self.fc(pooled)
