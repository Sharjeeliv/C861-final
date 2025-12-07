import torch.nn as nn
import torch.nn.functional as F
from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout):
        super().__init__()
    
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)


class SingleCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
    
    def forward(self, x):
        # x shape: [B, T, Channels]
        x = x.transpose(1, 2)  # [B, Channels, T]
        x = F.relu(self.conv(x))
        return x  # [B, Filters, T]
