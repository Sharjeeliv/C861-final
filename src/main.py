from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import mean_absolute_error, accuracy_score

from .data import get_datasets
from .utils.utils import time_execution

@time_execution
def ml_process(X_tr, y_tr, X_te, y_te):
    
    print("Encoding...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_tr)
    X_test_tfidf = tfidf_vectorizer.transform(X_te)

    print("Training...")
    # model = LogisticRegression(max_iter=1000)
    model = LinearSVC()
    model.fit(X_train_tfidf, y_tr)
    
    print("Testing...")
    y_pred = model.predict(X_test_tfidf)
    res = accuracy_score(y_te, y_pred)
    print(f"Accuracy: {res}")

from nltk import word_tokenize
import pandas as pd
from .utils.data import _build_vocab

MIN_FREQ = 5



from .utils.data import SentimentDataset, collate_batch
from torch.utils.data import DataLoader


def dl_preocess(X_tr, y_tr, X_te: pd.Series, y_te):
    
    # 1. Build vocabulary and convert seq. to ints
    word_to_idx, all_tokens = _build_vocab(X_tr)

    # Define necessary constants
    VOCAB_SIZE = len(word_to_idx)
    PAD_IDX = word_to_idx['<pad>']
    UNK_IDX = word_to_idx['<unk>']
    
    print(VOCAB_SIZE)
    print(PAD_IDX)
    print(UNK_IDX)
    
    def text_pipeline(text, word_to_idx):
        UNK_IDX = word_to_idx['<unk>']
        # Lookup index, defaulting to UNK_IDX if not found
        return [word_to_idx.get(token, UNK_IDX) for token in word_tokenize(text)]
    
    # for text in X_te.head(10):
    #     print(text)
    #     print(text_pipeline(text, word_to_idx))
    #     print("\n")
    
    # Instantiate datasets
    # Assuming X_tr, y_tr_encoded, X_te, y_te_encoded are ready
    train_dataset = SentimentDataset(X_tr, y_tr, text_pipeline, word_to_idx)
    test_dataset = SentimentDataset(X_te, y_te, text_pipeline, word_to_idx)
    
    # --- Hyperparameter ---
    BATCH_SIZE = 64

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,          # Shuffle training data
        collate_fn=collate_batch
    )

    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,         # Do not shuffle test data
        collate_fn=collate_batch
    )

    print(f"DataLoaders created with batch size: {BATCH_SIZE}")
    
    for labels, texts in train_dataloader:
        print(f"Batch Labels Shape: {labels.shape}")   # e.g., torch.Size([64])
        print(f"Batch Texts Shape: {texts.shape}")    # e.g., torch.Size([64, 98])
        break
    
    
    # 1. Build vocabulary and convert seq. to ints
    #    a. Ensure vocab is for words > MIN_FREQ
    #    b. Other words become unk?
    # 2. Pad the sequence?
    
    # 3. Can use for cnn, rnn and lstm now>
    #    a. First layer of model will produce embeddings 
    
    pass
    
    

def test():
    X_tr, y_tr, X_te, y_te = get_datasets()
    dl_preocess(X_tr, y_tr, X_te, y_te)
    


if __name__ == "__main__":
    test()
