# Data specific utility functions, while the main data.py handles 
# direct data related functions he utils data.py handles extra data 
# function perhaps making a data folder with data and util files?

from .utils import time_execution
import pandas as pd
from nltk import word_tokenize
from collections import Counter
from itertools import chain


@time_execution
def _build_vocab(texts: pd.Series, min_freq=2, specials=['<unk>', '<pad>']):
    
    print("Building Vocabulary...")
    token_stream = (word_tokenize(text) for text in texts)
    counter = Counter(chain.from_iterable(token_stream))
    filtered_tokens = [token for token, count in counter.items() if count >= min_freq]
    
    # Create sorted list of tokens including specials
    all_tokens = specials + sorted(filtered_tokens)
    # Create the final word-to-index dictionary (the vocab)
    word_to_idx = {word: idx for idx, word in enumerate(all_tokens)}
    return word_to_idx, all_tokens



import torch
from torch.utils.data import Dataset, DataLoader

class SentimentDataset(Dataset):
    def __init__(self, texts, labels_encoded, text_pipeline, word_to_idx):
        # 1. Store the lists of cleaned text and integer labels
        self.texts = texts
        # 2. Convert integer labels to a PyTorch LongTensor (required for classification loss)
        self.labels = torch.tensor(labels_encoded, dtype=torch.long)
        self.pipeline = text_pipeline # The function that converts text string to list of IDs
        self.word_to_idx = word_to_idx

    def __len__(self):
        # Returns the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns one sample: the sequence of token IDs and the target label
        token_ids = self.pipeline(self.texts[idx], self.word_to_idx)
        return token_ids, self.labels[idx]
    
    
from torch.nn.utils.rnn import pad_sequence

# Assuming UNK_IDX and PAD_IDX are defined from your vocabulary build:
# PAD_IDX = word_to_idx['<pad>'] 

PAD_IDX = 1

def collate_batch(batch):
    """
    Function to collate data samples into a batch tensor with padding.
    """
    label_list, text_list = [], []
    
    for text_ids, label in batch:
        label_list.append(label)
        
        # Convert the list of IDs (text_ids) to a PyTorch tensor
        processed_text = torch.tensor(text_ids, dtype=torch.long)
        text_list.append(processed_text)
    
    # Pad all sequences in the batch to the length of the longest one in THIS batch.
    # batch_first=True makes the tensor shape [batch_size, sequence_length]
    padded_texts = pad_sequence(
        text_list, 
        batch_first=True, 
        padding_value=PAD_IDX
    )
    
    # Return labels and padded texts as Tensors
    return torch.stack(label_list), padded_texts