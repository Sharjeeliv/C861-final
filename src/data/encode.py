from pathlib import Path
from collections import Counter
from itertools import chain

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
from nltk import word_tokenize

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import tensor, long


# ********************************
# VARIABLE AND SETUP
# ********************************
BATCH_SIZE = 64
UNK_IDX, PAD_IDX = 0, 1
SPECIAL = ['<unk>', '<pad>']

# ********************************
# HELPER FUNCTION
# ********************************
def _build_vocab(texts: pd.Series, min_freq=2):
    print("Building Vocabulary...")
    token_stream = (word_tokenize(text) for text in texts)
    counter = Counter(chain.from_iterable(token_stream))
    filtered_tokens = [t for t, c in counter.items() if c >= min_freq]
    
    # Create sorted list of tokens including specials
    all_tokens = SPECIAL + sorted(filtered_tokens)
    # Create the final word-to-index dictionary (the vocab)
    word_to_idx = {word: idx for idx, word in enumerate(all_tokens)}
    return word_to_idx, all_tokens


def _text_pipeline(text, vocab):
        # Lookup index, defaulting to UNK_IDX if not found
        return [vocab.get(t, UNK_IDX) for t in word_tokenize(text)]


def _collate_batch(batch):
    # Function to collate data samples into a batch tensor with padding.
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


def _get_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=_collate_batch
    )

# ********************************
# SENTIMENT DATASET DEFINITION
# ********************************
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts                          # 1. Store list of cleaned texts
        self.labels = tensor(labels, dtype=long)    # 2. Convert int label to LongTensor
        self.vocab = vocab                          # 3. Convert string text to ID list

    def __len__(self):
        # Returns the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns one sample: the sequence of token IDs and the target label
        token_ids = _text_pipeline(self.texts[idx], self.vocab)
        return token_ids, self.labels[idx]


# ********************************
# MAIN INTERFACE FUNCTION
# ********************************
# TF-ID Encoding: Classical Machine Learning
def tfid_encode(X_tr, X_te, max_feat=10000, ngram_range=(1, 3)):
    print("Encoding...")
    vectorizer = TfidfVectorizer(max_features=max_feat, 
                                 ngram_range=ngram_range)
    X_tr_tfid = vectorizer.fit_transform(X_tr)
    X_te_tfid = vectorizer.transform(X_te)
    return X_tr_tfid, X_te_tfid


# Sequence Encoding: RNN, LSTM, etc.
def seq_encode(X_tr: pd.Series, y_tr: pd.Series,
               X_te: pd.Series, y_te: pd.Series):
    print("Encoding...")
    # 1. Build vocabulary (word-to-id)
    vocab, _ = _build_vocab(X_tr)
    # 2. Build datasets
    tr_dataset = SentimentDataset(X_tr, y_tr, vocab)
    te_dataset = SentimentDataset(X_te, y_te, vocab)
    # 3. Build dataloades
    tr_loader = _get_loader(tr_dataset, BATCH_SIZE, True)
    te_loader = _get_loader(te_dataset, BATCH_SIZE, False)
    
    return tr_loader, te_loader