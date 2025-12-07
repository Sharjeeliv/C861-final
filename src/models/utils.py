
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout, 
                 pretrained_weights=None, freeze=True):
        super().__init__()
        
        # Initialize standard embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        
        # --- Pre-trained Weight Handling ---
        if pretrained_weights is not None:
            # 1. Check dimensions
            assert pretrained_weights.shape == (vocab_size, embed_dim), "Pre-trained weights shape mismatch."
            
            # 2. Load the weights
            self.embedding.weight = nn.Parameter(pretrained_weights)
            
            # 3. Freeze the layer if requested
            if freeze:
                self.embedding.weight.requires_grad = False
                print("Using pre-trained embeddings and freezing weights.")
            else:
                print("Using pre-trained embeddings and fine-tuning weights.")
        else:
            print("Using randomized embeddings.")
            
        self.dropout = nn.Dropout(dropout)
        # Other layers defined by child classes...
        


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
