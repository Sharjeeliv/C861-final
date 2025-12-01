from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import mean_absolute_error, accuracy_score

from .data import get_datasets
from .utils.utils import time_execution

BATCH_SIZE = 64

def tfid_encode(X_tr, X_te, max_feat=10000, ngram_range=(1, 3)):
    print("Encoding...")
    vectorizer = TfidfVectorizer(max_features=max_feat, ngram_range=ngram_range)
    X_tr_tfid = vectorizer.fit_transform(X_tr)
    X_te_tfid = vectorizer.transform(X_te)
    return X_tr_tfid, X_te_tfid
    


@time_execution
def ml_process(X_tr, y_tr, X_te, y_te):
    
    X_tr, X_te = tfid_encode(X_tr, X_te)
    print("Training...")
    # model = LogisticRegression(max_iter=1000)
    model = LinearSVC()
    model.fit(X_tr, y_tr)
    
    print("Testing...")
    y_pred = model.predict(X_te)
    res = accuracy_score(y_te, y_pred)
    print(f"Accuracy: {res}")

from nltk import word_tokenize
import pandas as pd
from .utils.data import _build_vocab

MIN_FREQ = 5



from .utils.data import SentimentDataset, collate_batch
from torch.utils.data import DataLoader



def get_loader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_batch
    )


def seq_encode(X_tr, y_tr, X_te: pd.Series, y_te):
    # 1. Build vocabulary (word-to-id)
    vocab, _ = _build_vocab(X_tr)
    # 2. Build datasets
    tr_dataset = SentimentDataset(X_tr, y_tr, vocab)
    te_dataset = SentimentDataset(X_te, y_te, vocab)
    # 3. Build dataloades
    tr_loader = get_loader(tr_dataset, BATCH_SIZE, True)
    te_loader = get_loader(te_dataset, BATCH_SIZE, False)
    
    return tr_loader, te_loader



def dl_preocess(X_tr, y_tr, X_te: pd.Series, y_te):
    
    tr_loader, te_loader = seq_encode(X_tr, y_tr, X_te, y_te)
    
    for labels, texts in tr_loader:
        print(f"Batch Labels Shape: {labels.shape}")   # e.g., torch.Size([64])
        print(f"Batch Texts Shape: {texts.shape}")    # e.g., torch.Size([64, 98])
        break
    

def test():
    X_tr, y_tr, X_te, y_te = get_datasets()
    dl_preocess(X_tr, y_tr, X_te, y_te)


if __name__ == "__main__":
    test()
